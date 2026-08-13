"""Local persistence for external resource connections and audit events.

A small SQLAlchemy store (SQLite by default, configurable via ``NLU_ADMIN_STORE_URL``)
that backs the admin GUI. It is intentionally separate from the *beneficiary* database
provider being configured — this store holds the configuration itself plus a durable
copy of every audit event, so the observability dashboards keep working even when ELK
is unavailable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for the admin store."""


class ResourceConnection(Base):
    """A configured external resource connection (database or datalake provider)."""

    __tablename__ = "resource_connections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), default="database", nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[str] = mapped_column(Text, default="", nullable=False)
    account_param: Mapped[str] = mapped_column(
        String(64), default="account_number", nullable=False
    )
    # JSON-encoded mapping of result columns -> Beneficiary fields.
    column_map: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class AuditEventRow(Base):
    """A single recorded system action."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True, nullable=False
    )
    action: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    category: Mapped[str] = mapped_column(String(64), default="http", index=True)
    method: Mapped[str | None] = mapped_column(String(16), nullable=True)
    path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    outcome: Mapped[str] = mapped_column(String(16), default="success", index=True)
    # JSON-encoded extra detail (request/response summary).
    detail: Mapped[str] = mapped_column(Text, default="{}", nullable=False)


class AppSetting(Base):
    """A single JSON-encoded configuration blob keyed by name (e.g. banking core)."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (once) the configured admin-store engine and ensure tables exist."""

    url = settings.admin_store_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker:
    """Return a configured ``sessionmaker`` bound to the admin-store engine."""

    return sessionmaker(bind=get_engine(), expire_on_commit=False)
