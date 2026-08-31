"""Durable table for conversation turns, with its own engine.

Separate from the admin store on purpose: an operations team reading turns is not
the same audience as the people configuring the bank, and a real deployment points
this at a retained database (Postgres) while the admin store stays where it is. It
defaults to the admin-store URL so a single-file demo still works.

Nothing here holds a value the bank acts on: no raw customer text, no account
number, no beneficiary name, no exact amount. See :mod:`app.observability.turns`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for the turn-observability store."""


class ConversationTurnRow(Base):
    """One customer turn as the conversation layer decided it."""

    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True, nullable=False
    )
    # Correlates this turn with the HTTP request in the audit log and the logs.
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Salted digests, never the identifiers themselves: enough to follow one
    # conversation or count returning customers, not enough to name anybody.
    session_ref: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    customer_ref: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    language: Mapped[str] = mapped_column(String(8), nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    pending_slot: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # The code the engine attached to this decision (``ReasonCode``), not a
    # description inferred afterwards from the status.
    reason_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    # JSON: per slot, whether it is filled and where it came from — no values.
    slots_masked: Mapped[str] = mapped_column(Text, default="{}", nullable=False)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (once) the turn-store engine and ensure the table exists."""

    url = settings.turn_store_url or settings.admin_store_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker:
    """Return a ``sessionmaker`` bound to the turn-store engine."""

    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def reset_engine() -> None:
    """Drop the cached engine so a new store URL takes effect (tests)."""

    get_sessionmaker.cache_clear()
    get_engine.cache_clear()
