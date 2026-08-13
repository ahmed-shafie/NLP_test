"""Service layer for external resource connections (CRUD, test, activate).

Connections describe how to reach an external resource (database or datalake) for the
beneficiary lookup. They are persisted in the admin store and edited through the GUI;
activating one makes the beneficiary repository use it without any code change.
"""

from __future__ import annotations

import json
import logging
import time

from sqlalchemy import select

from app.admin.schemas import (
    Connection,
    ConnectionCreate,
    ConnectionTestResult,
    ConnectionUpdate,
)
from app.admin.store import ResourceConnection, get_sessionmaker

logger = logging.getLogger(__name__)


def _to_schema(row: ResourceConnection) -> Connection:
    try:
        column_map = json.loads(row.column_map or "{}")
    except json.JSONDecodeError:
        column_map = {}
    return Connection(
        id=row.id,
        name=row.name,
        kind=row.kind,
        provider=row.provider,
        url=row.url,
        query=row.query,
        account_param=row.account_param,
        column_map=column_map,
        is_active=row.is_active,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def list_connections() -> list[Connection]:
    """Return all stored connections, newest first."""

    with get_sessionmaker()() as session:
        rows = session.scalars(
            select(ResourceConnection).order_by(ResourceConnection.id.desc())
        ).all()
        return [_to_schema(row) for row in rows]


def get_connection(connection_id: int) -> Connection | None:
    """Return a single connection by id, or ``None``."""

    with get_sessionmaker()() as session:
        row = session.get(ResourceConnection, connection_id)
        return _to_schema(row) if row else None


def get_active_connection() -> Connection | None:
    """Return the active connection, if any."""

    with get_sessionmaker()() as session:
        row = session.scalars(
            select(ResourceConnection).where(ResourceConnection.is_active.is_(True))
        ).first()
        return _to_schema(row) if row else None


def create_connection(payload: ConnectionCreate) -> Connection:
    """Persist a new connection."""

    with get_sessionmaker()() as session:
        row = ResourceConnection(
            name=payload.name,
            kind=payload.kind,
            provider=payload.provider,
            url=payload.url,
            query=payload.query,
            account_param=payload.account_param,
            column_map=json.dumps(payload.column_map),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _to_schema(row)


def update_connection(
    connection_id: int, payload: ConnectionUpdate
) -> Connection | None:
    """Apply a partial update to a connection."""

    with get_sessionmaker()() as session:
        row = session.get(ResourceConnection, connection_id)
        if row is None:
            return None
        data = payload.model_dump(exclude_unset=True)
        if "column_map" in data and data["column_map"] is not None:
            row.column_map = json.dumps(data.pop("column_map"))
        for field, value in data.items():
            if value is not None:
                setattr(row, field, value)
        session.commit()
        session.refresh(row)
        return _to_schema(row)


def delete_connection(connection_id: int) -> bool:
    """Delete a connection. Returns ``True`` if a row was removed."""

    with get_sessionmaker()() as session:
        row = session.get(ResourceConnection, connection_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True


def activate_connection(connection_id: int) -> Connection | None:
    """Mark one connection active (deactivating the others)."""

    with get_sessionmaker()() as session:
        target = session.get(ResourceConnection, connection_id)
        if target is None:
            return None
        for row in session.scalars(select(ResourceConnection)).all():
            row.is_active = row.id == connection_id
        session.commit()
        session.refresh(target)
    # The cached beneficiary repository must be rebuilt against the new connection.
    _reset_beneficiary_repository()
    return _to_schema(target)


def _reset_beneficiary_repository() -> None:
    """Clear the cached beneficiary repository so the next lookup rebuilds it."""

    from app.db.beneficiary import get_beneficiary_repository

    get_beneficiary_repository.cache_clear()


def test_connection(
    url: str, query: str, account_param: str, sample_account: str | None = None
) -> ConnectionTestResult:
    """Open the connection and run a lightweight probe against it."""

    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:  # pragma: no cover - SQLAlchemy is a hard dep
        return ConnectionTestResult(ok=False, message=f"SQLAlchemy unavailable: {exc}")

    started = time.perf_counter()
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            if sample_account:
                result = conn.execute(text(query), {account_param: sample_account})
                columns = list(result.keys())
            else:
                # No sample account: validate connectivity with a trivial probe.
                conn.execute(text("SELECT 1"))
                columns = []
        elapsed = (time.perf_counter() - started) * 1000
        engine.dispose()
        return ConnectionTestResult(
            ok=True,
            message="Connection succeeded.",
            elapsed_ms=round(elapsed, 2),
            sample_columns=columns,
        )
    except Exception as exc:  # noqa: BLE001 - surface any driver/connection error
        elapsed = (time.perf_counter() - started) * 1000
        return ConnectionTestResult(
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
            elapsed_ms=round(elapsed, 2),
        )
