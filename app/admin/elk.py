"""Elasticsearch (ELK) integration for audit observability.

Ships audit events to Elasticsearch and computes the dashboard aggregations there.
Everything degrades gracefully: if the ``elasticsearch`` client is missing or the
cluster is unreachable, shipping is skipped and the API falls back to the local store.
"""

from __future__ import annotations

import logging
import socket
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING

from app.admin.schemas import AuditStats, ElkStatus
from app.config import settings

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> Elasticsearch | None:
    """Return a cached Elasticsearch client, or ``None`` if unavailable."""

    if not settings.elk_enabled:
        return None
    try:
        from elasticsearch import Elasticsearch
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.debug("elasticsearch client not installed: %s", exc)
        return None
    auth = None
    if settings.elasticsearch_username and settings.elasticsearch_password:
        auth = (settings.elasticsearch_username, settings.elasticsearch_password)
    try:
        return Elasticsearch(
            settings.elasticsearch_url,
            basic_auth=auth,
            request_timeout=5,
            retry_on_timeout=False,
            max_retries=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build Elasticsearch client: %s", exc)
        return None


def ship_event(document: dict) -> bool:
    """Index a single audit document. Returns ``True`` on success."""

    client = get_client()
    if client is None:
        return False
    try:
        client.index(index=settings.elk_index, document=document)
        return True
    except Exception as exc:  # noqa: BLE001 - never break the request path
        logger.debug("Elasticsearch index failed: %s", exc)
        return False


def ship_event_via_logstash(line: str) -> bool:
    """Send one JSON line to Logstash over TCP (json_lines codec)."""

    try:
        with socket.create_connection(
            (settings.logstash_host, settings.logstash_port), timeout=3
        ) as sock:
            sock.sendall((line + "\n").encode("utf-8"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Logstash send failed: %s", exc)
        return False


def status() -> ElkStatus:
    """Probe the ELK pipeline for the observability page."""

    base = ElkStatus(
        enabled=settings.elk_enabled,
        sink=settings.audit_sink,
        reachable=False,
        index=settings.elk_index,
    )
    client = get_client()
    if client is None:
        base.message = "Elasticsearch client unavailable or disabled."
        return base
    try:
        info = client.info()
        base.reachable = True
        base.cluster = info.get("cluster_name")
        try:
            count = client.count(index=settings.elk_index)
            base.doc_count = int(count.get("count", 0))
        except Exception:  # noqa: BLE001 - index may not exist yet
            base.doc_count = 0
    except Exception as exc:  # noqa: BLE001
        base.message = f"Elasticsearch unreachable: {exc}"
    return base


def fetch_stats(window_minutes: int = 1440) -> AuditStats | None:
    """Compute dashboard aggregations from Elasticsearch, or ``None`` if unavailable."""

    client = get_client()
    if client is None:
        return None
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    gte = now_ms - window_minutes * 60 * 1000
    query: dict[str, dict] = {
        "range": {"timestamp": {"gte": gte, "lte": now_ms, "format": "epoch_millis"}}
    }
    aggs: dict[str, dict] = {
        "by_status": {"terms": {"field": "status_code", "size": 20}},
        "by_action": {"terms": {"field": "action", "size": 20}},
        "by_category": {"terms": {"field": "category", "size": 20}},
        "by_outcome": {"terms": {"field": "outcome", "size": 5}},
        "top_paths": {"terms": {"field": "path", "size": 10}},
        "avg_duration": {"avg": {"field": "duration_ms"}},
        "p95_duration": {"percentiles": {"field": "duration_ms", "percents": [95]}},
        "timeline": {
            "date_histogram": {
                "field": "timestamp",
                "fixed_interval": _bucket_interval(window_minutes),
                "min_doc_count": 0,
            }
        },
    }
    try:
        resp = client.search(index=settings.elk_index, query=query, aggs=aggs, size=0)
    except Exception as exc:  # noqa: BLE001 - index missing / cluster down
        logger.debug("Elasticsearch stats query failed: %s", exc)
        return None

    agg = resp.get("aggregations", {})
    total = int(resp.get("hits", {}).get("total", {}).get("value", 0))
    by_outcome = {
        b["key"]: b["doc_count"] for b in agg.get("by_outcome", {}).get("buckets", [])
    }
    p95_values = agg.get("p95_duration", {}).get("values", {})
    return AuditStats(
        source="elasticsearch",
        total=total,
        success=by_outcome.get("success", 0),
        errors=by_outcome.get("error", 0),
        avg_duration_ms=round(agg.get("avg_duration", {}).get("value") or 0.0, 2),
        p95_duration_ms=round(_first_value(p95_values), 2),
        by_status={
            str(b["key"]): b["doc_count"]
            for b in agg.get("by_status", {}).get("buckets", [])
        },
        by_action={
            b["key"]: b["doc_count"]
            for b in agg.get("by_action", {}).get("buckets", [])
        },
        by_category={
            b["key"]: b["doc_count"]
            for b in agg.get("by_category", {}).get("buckets", [])
        },
        top_paths=[
            {"path": b["key"], "count": b["doc_count"]}
            for b in agg.get("top_paths", {}).get("buckets", [])
        ],
        timeline=[
            {"t": b["key_as_string"], "count": b["doc_count"]}
            for b in agg.get("timeline", {}).get("buckets", [])
        ],
    )


def _first_value(values: dict) -> float:
    for value in values.values():
        return float(value or 0.0)
    return 0.0


def _bucket_interval(window_minutes: int) -> str:
    if window_minutes <= 60:
        return "1m"
    if window_minutes <= 1440:
        return "1h"
    return "1d"
