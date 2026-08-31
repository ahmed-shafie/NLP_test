"""Authenticated read API over the turn store and the SLO catalogue.

Fail-closed: with no key configured these routes answer 503, never open. A turn
row is thin by construction, but "who is stuck on which slot, and which sessions"
is still operational detail about real customers, so it does not go out on an
unauthenticated endpoint the way the imported design had it.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel

from app.config import settings
from app.observability import alerts
from app.observability.turns import (
    TurnRecord,
    counts,
    list_turns,
    purge_older_than,
    session_ref_for,
)

OPS_KEY_HEADER = "x-ops-key"


def require_ops_key(x_ops_key: str | None = Header(default=None)) -> None:
    """Reject the request unless it carries the configured operations key."""

    if not settings.ops_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Turn observability is unauthenticated; set NLU_OPS_API_KEY.",
        )
    if not x_ops_key or not hmac.compare_digest(x_ops_key, settings.ops_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid operations key is required.",
        )


router = APIRouter(
    prefix="/ops/observability",
    tags=["observability"],
    dependencies=[Depends(require_ops_key)],
)


class TurnView(BaseModel):
    """A turn row as the API returns it."""

    id: int
    timestamp: str
    trace_id: str | None
    session_ref: str
    customer_ref: str | None
    language: str
    intent: str | None
    status: str
    pending_slot: str | None
    reason_code: str | None
    latency_ms: float | None
    slots: dict[str, str]


class SloView(BaseModel):
    """One objective and its current reading."""

    key: str
    title: str
    window_minutes: int
    severity: str
    max_ratio: float
    observed: int
    total: int
    ratio: float
    breached: bool
    rationale: str


class SloReport(BaseModel):
    """The whole catalogue, plus whether anything is breaching."""

    breaching: int
    objectives: list[SloView]


class PurgeResult(BaseModel):
    """How many rows the retention policy removed."""

    deleted: int
    retention_days: int


def _to_view(record: TurnRecord) -> TurnView:
    return TurnView(
        id=record.id,
        timestamp=record.timestamp.isoformat(),
        trace_id=record.trace_id,
        session_ref=record.session_ref,
        customer_ref=record.customer_ref,
        language=record.language,
        intent=record.intent,
        status=record.status,
        pending_slot=record.pending_slot,
        reason_code=record.reason_code,
        latency_ms=record.latency_ms,
        slots=record.slots,
    )


@router.get("/turns", response_model=list[TurnView])
def get_turns(
    limit: int = Query(default=100, ge=1, le=1000),
    session_id: str | None = Query(
        default=None,
        description="A known session id; it is hashed here, never stored or logged.",
    ),
    session_ref: str | None = Query(default=None, description="A session digest."),
    reason_code: str | None = Query(default=None),
) -> list[TurnView]:
    """Recent turns, newest first."""

    reference = session_ref_for(session_id) if session_id else session_ref
    records = list_turns(limit=limit, session_ref=reference, reason_code=reason_code)
    return [_to_view(record) for record in records]


@router.get("/slo", response_model=SloReport)
def get_slo() -> SloReport:
    """Read every objective in the catalogue against the store."""

    per_window = {window: counts(window) for window in alerts.windows()}
    measurements = alerts.evaluate(per_window)
    objectives = [
        SloView(
            key=m.slo.key,
            title=m.slo.title,
            window_minutes=m.slo.window_minutes,
            severity=m.slo.severity.value,
            max_ratio=m.slo.max_ratio,
            observed=m.observed,
            total=m.total,
            ratio=m.ratio,
            breached=m.breached,
            rationale=m.slo.rationale,
        )
        for m in measurements
    ]
    return SloReport(
        breaching=sum(1 for m in measurements if m.breached), objectives=objectives
    )


@router.post("/retention/purge", response_model=PurgeResult)
def purge_retention() -> PurgeResult:
    """Apply the retention window now."""

    days = settings.turn_retention_days
    return PurgeResult(deleted=purge_older_than(days), retention_days=days)
