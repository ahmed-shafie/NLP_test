"""FastAPI routes for the admin GUI: resource connections + audit observability."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.admin import audit, connections, elk
from app.admin.schemas import (
    PROVIDER_PRESETS,
    AuditEvent,
    AuditStats,
    Connection,
    ConnectionCreate,
    ConnectionTestResult,
    ConnectionUpdate,
    ElkStatus,
)

router = APIRouter(prefix="/admin/api", tags=["admin"])


def _actor(request: Request) -> str:
    return request.headers.get("x-actor") or "anonymous"


# --------------------------------------------------------------------------- #
# Resource connections
# --------------------------------------------------------------------------- #
@router.get("/connections/providers")
def list_providers() -> list[dict]:
    """Return the supported provider presets surfaced in the GUI."""

    return PROVIDER_PRESETS


@router.get("/connections/active", response_model=Connection | None)
def active_connection() -> Connection | None:
    """Return the currently active connection, if any."""

    return connections.get_active_connection()


@router.get("/connections", response_model=list[Connection])
def list_connections() -> list[Connection]:
    """List all stored connections."""

    return connections.list_connections()


@router.post("/connections", response_model=Connection, status_code=201)
def create_connection(payload: ConnectionCreate, request: Request) -> Connection:
    """Create a new connection."""

    created = connections.create_connection(payload)
    audit.record(
        "connection.create",
        category="admin",
        actor=_actor(request),
        detail={"id": created.id, "name": created.name, "provider": created.provider},
    )
    return created


@router.post("/connections/test", response_model=ConnectionTestResult)
def test_adhoc(payload: ConnectionCreate, request: Request) -> ConnectionTestResult:
    """Test a connection from an unsaved form payload."""

    result = connections.test_connection(
        payload.url, payload.query, payload.account_param
    )
    audit.record(
        "connection.test",
        category="admin",
        actor=_actor(request),
        outcome="success" if result.ok else "error",
        detail={"provider": payload.provider, "ok": result.ok, "message": result.message},
    )
    return result


@router.get("/connections/{connection_id}", response_model=Connection)
def get_connection(connection_id: int) -> Connection:
    """Get a single connection."""

    found = connections.get_connection(connection_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return found


@router.put("/connections/{connection_id}", response_model=Connection)
def update_connection(
    connection_id: int, payload: ConnectionUpdate, request: Request
) -> Connection:
    """Update a connection."""

    updated = connections.update_connection(connection_id, payload)
    if updated is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    audit.record(
        "connection.update",
        category="admin",
        actor=_actor(request),
        detail={"id": connection_id},
    )
    return updated


@router.delete("/connections/{connection_id}", status_code=204, response_class=Response)
def delete_connection(connection_id: int, request: Request) -> Response:
    """Delete a connection."""

    if not connections.delete_connection(connection_id):
        raise HTTPException(status_code=404, detail="Connection not found")
    audit.record(
        "connection.delete",
        category="admin",
        actor=_actor(request),
        detail={"id": connection_id},
    )
    return Response(status_code=204)


@router.post("/connections/{connection_id}/activate", response_model=Connection)
def activate_connection(connection_id: int, request: Request) -> Connection:
    """Make a connection the active beneficiary provider."""

    activated = connections.activate_connection(connection_id)
    if activated is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    audit.record(
        "connection.activate",
        category="admin",
        actor=_actor(request),
        detail={"id": connection_id, "name": activated.name},
    )
    return activated


@router.post("/connections/{connection_id}/test", response_model=ConnectionTestResult)
def test_saved(
    connection_id: int,
    request: Request,
    sample_account: str | None = Query(default=None),
) -> ConnectionTestResult:
    """Test a saved connection, optionally running the lookup for a sample account."""

    found = connections.get_connection(connection_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    result = connections.test_connection(
        found.url, found.query, found.account_param, sample_account
    )
    audit.record(
        "connection.test",
        category="admin",
        actor=_actor(request),
        outcome="success" if result.ok else "error",
        detail={"id": connection_id, "ok": result.ok, "message": result.message},
    )
    return result


# --------------------------------------------------------------------------- #
# Audit observability
# --------------------------------------------------------------------------- #
@router.get("/audit/events", response_model=list[AuditEvent])
def audit_events(
    limit: int = Query(default=100, ge=1, le=1000),
    category: str | None = None,
    outcome: str | None = None,
    action: str | None = None,
) -> list[AuditEvent]:
    """Return recent audit events from the durable store."""

    return audit.list_events(
        limit=limit, category=category, outcome=outcome, action=action
    )


@router.get("/audit/stats", response_model=AuditStats)
def audit_stats(window_minutes: int = Query(default=1440, ge=1, le=43200)) -> AuditStats:
    """Return aggregated metrics for the observability charts."""

    return audit.get_stats(window_minutes)


@router.get("/audit/elk-status", response_model=ElkStatus)
def elk_status() -> ElkStatus:
    """Report ELK pipeline availability."""

    return elk.status()
