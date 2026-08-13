"""Persistence for the Memory Brain.

Two layers, combined by :class:`MemoryStore`:

* a durable **SQL** store (SQLAlchemy; any provider, SQLite by default) that is the
  source of truth for each user's habits and shortcuts, and
* a fast **cache** (Redis, or a process-local dict) holding the serialized
  :class:`~app.memory.schemas.UserMemory` so repeat reads avoid the database.

Both layers degrade gracefully: if Redis is unavailable the cache falls back to an
in-memory dict, and the SQL store always remains authoritative.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime
from functools import lru_cache
from typing import Protocol

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings
from app.memory.schemas import Habits, Shortcut, UserMemory

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "nlu:memory:"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for the memory store."""


class UserHabitsRow(Base):
    """Learned habits for a single user."""

    __tablename__ = "user_habits"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    preferred_currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    preferred_source_account: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    preferred_language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    favorite_recipient: Mapped[str | None] = mapped_column(String(200), nullable=True)
    last_recipient: Mapped[str | None] = mapped_column(String(200), nullable=True)
    last_currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    total_transfers: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # JSON-encoded {recipient: count} and [amount, ...].
    recipient_counts: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    # JSON-encoded {"recipient|amount|currency": count}.
    combo_counts: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    common_amounts: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class UserShortcutRow(Base):
    """A user-defined shortcut (named transfer template)."""

    __tablename__ = "user_shortcuts"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_shortcut"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    amount: Mapped[str | None] = mapped_column(String(64), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    recipient: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_account: Mapped[str | None] = mapped_column(String(64), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    url = settings.memory_store_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


@lru_cache(maxsize=1)
def _get_sessionmaker() -> sessionmaker:
    return sessionmaker(bind=_get_engine(), expire_on_commit=False)


# --------------------------------- cache ----------------------------------- #


class MemoryCache(Protocol):
    def get(self, user_id: str) -> UserMemory | None: ...

    def set(self, memory: UserMemory) -> None: ...

    def invalidate(self, user_id: str) -> None: ...


class InMemoryCache:
    """Process-local TTL cache."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, user_id: str) -> UserMemory | None:
        with self._lock:
            entry = self._data.get(user_id)
            if entry is None:
                return None
            expires_at, raw = entry
            if expires_at < time.monotonic():
                self._data.pop(user_id, None)
                return None
        return UserMemory.model_validate_json(raw)

    def set(self, memory: UserMemory) -> None:
        with self._lock:
            self._data[memory.user_id] = (
                time.monotonic() + self._ttl,
                memory.model_dump_json(),
            )

    def invalidate(self, user_id: str) -> None:
        with self._lock:
            self._data.pop(user_id, None)


class RedisCache:
    """Shared cache backed by Redis with a per-user TTL."""

    def __init__(self, client, ttl_seconds: int) -> None:
        self._client = client
        self._ttl = ttl_seconds

    def get(self, user_id: str) -> UserMemory | None:
        raw = self._client.get(_CACHE_PREFIX + user_id)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return UserMemory.model_validate_json(raw)

    def set(self, memory: UserMemory) -> None:
        self._client.set(
            _CACHE_PREFIX + memory.user_id, memory.model_dump_json(), ex=self._ttl
        )

    def invalidate(self, user_id: str) -> None:
        self._client.delete(_CACHE_PREFIX + user_id)


def _build_cache() -> MemoryCache:
    ttl = settings.memory_cache_ttl_seconds
    if settings.memory_cache_backend == "redis":
        try:
            import redis  # noqa: PLC0415 - optional dependency, imported lazily

            client = redis.Redis.from_url(settings.redis_url)
            client.ping()
            logger.info("Memory Brain cache using Redis at %s", settings.redis_url)
            return RedisCache(client, ttl)
        except Exception as exc:  # noqa: BLE001 - any failure falls back to memory
            logger.warning(
                "Redis unavailable for memory cache (%s); using in-memory cache", exc
            )
    return InMemoryCache(ttl)


# ------------------------------ row mapping -------------------------------- #


def _habits_from_row(row: UserHabitsRow | None) -> Habits:
    if row is None:
        return Habits()
    return Habits(
        preferred_currency=row.preferred_currency,
        preferred_source_account=row.preferred_source_account,
        preferred_language=row.preferred_language,  # type: ignore[arg-type]
        favorite_recipient=row.favorite_recipient,
        last_recipient=row.last_recipient,
        last_currency=row.last_currency,
        total_transfers=row.total_transfers,
        recipient_counts=json.loads(row.recipient_counts or "{}"),
        combo_counts=json.loads(row.combo_counts or "{}"),
        common_amounts=json.loads(row.common_amounts or "[]"),
    )


def _shortcut_from_row(row: UserShortcutRow) -> Shortcut:
    return Shortcut(
        name=row.name,
        amount=row.amount,  # type: ignore[arg-type]
        currency=row.currency,
        recipient=row.recipient,
        source_account=row.source_account,
        note=row.note,
    )


class MemoryStore:
    """Combines the durable SQL store with the read cache."""

    def __init__(self) -> None:
        self._sessionmaker = _get_sessionmaker()
        self._cache = _build_cache()

    # ------------------------------- reads -------------------------------- #

    def get(self, user_id: str) -> UserMemory:
        cached = self._cache.get(user_id)
        if cached is not None:
            return cached
        memory = self._load_from_db(user_id)
        self._cache.set(memory)
        return memory

    def _load_from_db(self, user_id: str) -> UserMemory:
        with self._sessionmaker() as session:
            habits_row = session.get(UserHabitsRow, user_id)
            shortcut_rows = (
                session.execute(
                    select(UserShortcutRow).where(UserShortcutRow.user_id == user_id)
                )
                .scalars()
                .all()
            )
        return UserMemory(
            user_id=user_id,
            habits=_habits_from_row(habits_row),
            shortcuts=[_shortcut_from_row(r) for r in shortcut_rows],
        )

    def list_memories(self) -> list[UserMemory]:
        """Return memory for every known user (read straight from SQL).

        Combines users that have habits, shortcuts, or both. This bypasses the
        per-user cache since it is a full scan used by the monitoring dashboard.
        """

        with self._sessionmaker() as session:
            habit_rows = session.execute(select(UserHabitsRow)).scalars().all()
            shortcut_rows = session.execute(select(UserShortcutRow)).scalars().all()

        habits_by_user = {row.user_id: row for row in habit_rows}
        shortcuts_by_user: dict[str, list[Shortcut]] = {}
        for row in shortcut_rows:
            shortcuts_by_user.setdefault(row.user_id, []).append(
                _shortcut_from_row(row)
            )

        user_ids = sorted(set(habits_by_user) | set(shortcuts_by_user))
        return [
            UserMemory(
                user_id=user_id,
                habits=_habits_from_row(habits_by_user.get(user_id)),
                shortcuts=shortcuts_by_user.get(user_id, []),
            )
            for user_id in user_ids
        ]

    # ------------------------------- writes ------------------------------- #

    def save_habits(self, user_id: str, habits: Habits) -> None:
        with self._sessionmaker() as session:
            row = session.get(UserHabitsRow, user_id)
            if row is None:
                row = UserHabitsRow(user_id=user_id)
                session.add(row)
            row.preferred_currency = habits.preferred_currency
            row.preferred_source_account = habits.preferred_source_account
            row.preferred_language = (
                habits.preferred_language.value if habits.preferred_language else None
            )
            row.favorite_recipient = habits.favorite_recipient
            row.last_recipient = habits.last_recipient
            row.last_currency = habits.last_currency
            row.total_transfers = habits.total_transfers
            row.recipient_counts = json.dumps(habits.recipient_counts)
            row.combo_counts = json.dumps(habits.combo_counts)
            row.common_amounts = json.dumps([str(a) for a in habits.common_amounts])
            session.commit()
        self._cache.invalidate(user_id)

    def upsert_shortcut(self, user_id: str, shortcut: Shortcut) -> None:
        with self._sessionmaker() as session:
            row = session.execute(
                select(UserShortcutRow).where(
                    UserShortcutRow.user_id == user_id,
                    UserShortcutRow.name == shortcut.name,
                )
            ).scalar_one_or_none()
            if row is None:
                row = UserShortcutRow(user_id=user_id, name=shortcut.name)
                session.add(row)
            row.amount = str(shortcut.amount) if shortcut.amount is not None else None
            row.currency = shortcut.currency
            row.recipient = shortcut.recipient
            row.source_account = shortcut.source_account
            row.note = shortcut.note
            session.commit()
        self._cache.invalidate(user_id)

    def delete_shortcut(self, user_id: str, name: str) -> bool:
        with self._sessionmaker() as session:
            result = session.execute(
                delete(UserShortcutRow).where(
                    UserShortcutRow.user_id == user_id,
                    UserShortcutRow.name == name,
                )
            )
            session.commit()
            removed = result.rowcount > 0
        self._cache.invalidate(user_id)
        return removed


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    """Return the process-wide memory store (created once)."""

    return MemoryStore()
