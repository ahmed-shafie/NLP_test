"""Audit logging: record every system action and expose it for observability.

Each event is persisted durably in the admin store and, when configured, shipped to
ELK (directly to Elasticsearch or through Logstash). The middleware captures every
HTTP request; domain code can also call :func:`record` for non-HTTP system actions
(e.g. activating a connection). Dashboard stats come from Elasticsearch when reachable
and fall back to the local store otherwise.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.admin import elk
from app.admin.schemas import AuditEvent, AuditStats
from app.admin.store import AuditEventRow, get_sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)

# Paths whose requests are not worth auditing (static assets, liveness/readiness
# probes). Skipping /health avoids container healthchecks flooding the audit log.
_SKIP_PREFIXES = ("/static", "/favicon", "/health")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _to_document(event: AuditEvent) -> dict:
    """Serialize an event for ELK (timestamp as epoch millis for date math)."""

    return {
        "timestamp": int(event.timestamp.timestamp() * 1000),
        "@timestamp": event.timestamp.isoformat(),
        "action": event.action,
        "category": event.category,
        "method": event.method,
        "path": event.path,
        "status_code": event.status_code,
        "duration_ms": event.duration_ms,
        "client_ip": event.client_ip,
        "actor": event.actor,
        "request_id": event.request_id,
        "outcome": event.outcome,
        "detail": event.detail,
    }


def _ship(event: AuditEvent) -> None:
    """Forward an event to the configured sink (best effort)."""

    if settings.audit_sink == "none":
        return
    document = _to_document(event)
    if settings.audit_sink == "logstash":
        elk.ship_event_via_logstash(json.dumps(document, default=str))
    else:
        elk.ship_event(document)


def record(
    action: str,
    *,
    category: str = "system",
    method: str | None = None,
    path: str | None = None,
    status_code: int | None = None,
    duration_ms: float | None = None,
    client_ip: str | None = None,
    actor: str | None = None,
    request_id: str | None = None,
    outcome: str = "success",
    detail: dict | None = None,
) -> None:
    """Persist a single audit event and ship it to the configured sink."""

    if not settings.audit_enabled:
        return
    event = AuditEvent(
        timestamp=_utcnow(),
        action=action,
        category=category,
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
        client_ip=client_ip,
        actor=actor,
        request_id=request_id,
        outcome=outcome,
        detail=detail or {},
    )
    try:
        with get_sessionmaker()() as session:
            session.add(
                AuditEventRow(
                    timestamp=event.timestamp,
                    action=event.action,
                    category=event.category,
                    method=event.method,
                    path=event.path,
                    status_code=event.status_code,
                    duration_ms=event.duration_ms,
                    client_ip=event.client_ip,
                    actor=event.actor,
                    request_id=event.request_id,
                    outcome=event.outcome,
                    detail=json.dumps(event.detail, default=str),
                )
            )
            session.commit()
    except Exception as exc:  # noqa: BLE001 - auditing must never break the app
        logger.warning("Failed to persist audit event %s: %s", action, exc)
    _ship(event)


class AuditMiddleware(BaseHTTPMiddleware):
    """Record an audit event for every HTTP request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if not settings.audit_enabled or path.startswith(_SKIP_PREFIXES):
            return await call_next(request)

        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        actor = request.headers.get("x-actor") or "anonymous"
        client_ip = request.client.host if request.client else None
        started = time.perf_counter()
        status_code = 500
        outcome = "error"
        try:
            response = await call_next(request)
            status_code = response.status_code
            outcome = "success" if status_code < 400 else "error"
            response.headers["x-request-id"] = request_id
            return response
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            record(
                action=f"{request.method} {path}",
                category="http",
                method=request.method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
                client_ip=client_ip,
                actor=actor,
                request_id=request_id,
                outcome=outcome,
                detail={"query": dict(request.query_params)},
            )


def _row_to_event(row: AuditEventRow) -> AuditEvent:
    try:
        detail = json.loads(row.detail or "{}")
    except json.JSONDecodeError:
        detail = {}
    return AuditEvent(
        id=row.id,
        timestamp=row.timestamp,
        action=row.action,
        category=row.category,
        method=row.method,
        path=row.path,
        status_code=row.status_code,
        duration_ms=row.duration_ms,
        client_ip=row.client_ip,
        actor=row.actor,
        request_id=row.request_id,
        outcome=row.outcome,
        detail=detail,
    )


def list_events(
    limit: int = 100,
    category: str | None = None,
    outcome: str | None = None,
    action: str | None = None,
) -> list[AuditEvent]:
    """Return recent audit events from the store (newest first)."""

    with get_sessionmaker()() as session:
        stmt = select(AuditEventRow).order_by(AuditEventRow.id.desc())
        if category:
            stmt = stmt.where(AuditEventRow.category == category)
        if outcome:
            stmt = stmt.where(AuditEventRow.outcome == outcome)
        if action:
            stmt = stmt.where(AuditEventRow.action.like(f"%{action}%"))
        rows = session.scalars(stmt.limit(limit)).all()
        return [_row_to_event(row) for row in rows]


def stats_from_store(window_minutes: int = 1440) -> AuditStats:
    """Compute dashboard aggregations directly from the local store."""

    cutoff = _utcnow().timestamp() - window_minutes * 60
    with get_sessionmaker()() as session:
        rows = session.scalars(
            select(AuditEventRow).order_by(AuditEventRow.id.desc()).limit(5000)
        ).all()

    rows = [r for r in rows if r.timestamp.replace(tzinfo=UTC).timestamp() >= cutoff]
    total = len(rows)
    durations = sorted(r.duration_ms for r in rows if r.duration_ms is not None)
    by_status: Counter = Counter()
    by_action: Counter = Counter()
    by_category: Counter = Counter()
    by_path: Counter = Counter()
    timeline: Counter = Counter()
    success = errors = 0
    bucket = "%Y-%m-%dT%H:00:00Z" if window_minutes > 60 else "%Y-%m-%dT%H:%M:00Z"
    for row in rows:
        if row.status_code is not None:
            by_status[str(row.status_code)] += 1
        by_action[row.action] += 1
        by_category[row.category] += 1
        if row.path:
            by_path[row.path] += 1
        if row.outcome == "success":
            success += 1
        else:
            errors += 1
        ts = row.timestamp.replace(tzinfo=UTC)
        timeline[ts.strftime(bucket)] += 1

    avg = sum(durations) / len(durations) if durations else 0.0
    p95 = durations[int(len(durations) * 0.95)] if durations else 0.0
    return AuditStats(
        source="store",
        total=total,
        success=success,
        errors=errors,
        avg_duration_ms=round(avg, 2),
        p95_duration_ms=round(p95, 2),
        by_status=dict(by_status),
        by_action=dict(by_action.most_common(20)),
        by_category=dict(by_category),
        top_paths=[{"path": p, "count": c} for p, c in by_path.most_common(10)],
        timeline=[{"t": t, "count": timeline[t]} for t in sorted(timeline)],
    )


def get_stats(window_minutes: int = 1440) -> AuditStats:
    """Return dashboard stats, preferring Elasticsearch, falling back to the store."""

    if settings.audit_sink == "elasticsearch":
        es_stats = elk.fetch_stats(window_minutes)
        if es_stats is not None and es_stats.total > 0:
            return es_stats
    return stats_from_store(window_minutes)
