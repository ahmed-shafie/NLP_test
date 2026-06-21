"""Session storage for conversation state: Redis with an in-memory fallback."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Protocol

from app.config import settings
from app.conversation.state import ConversationState

logger = logging.getLogger(__name__)

_KEY_PREFIX = "nlu:conversation:"


class SessionStore(Protocol):
    def load(self, session_id: str) -> ConversationState | None: ...

    def save(self, state: ConversationState) -> None: ...

    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """Process-local store. Suitable for a single instance / local development."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()

    def load(self, session_id: str) -> ConversationState | None:
        with self._lock:
            raw = self._data.get(session_id)
        if raw is None:
            return None
        return ConversationState.model_validate_json(raw)

    def save(self, state: ConversationState) -> None:
        with self._lock:
            self._data[state.session_id] = state.model_dump_json()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


class RedisSessionStore:
    """Shared store backed by Redis, with a per-session TTL."""

    def __init__(self, client, ttl_seconds: int) -> None:
        self._client = client
        self._ttl = ttl_seconds

    def load(self, session_id: str) -> ConversationState | None:
        raw = self._client.get(_KEY_PREFIX + session_id)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return ConversationState.model_validate_json(raw)

    def save(self, state: ConversationState) -> None:
        self._client.set(
            _KEY_PREFIX + state.session_id,
            state.model_dump_json(),
            ex=self._ttl,
        )

    def delete(self, session_id: str) -> None:
        self._client.delete(_KEY_PREFIX + session_id)


def _build_redis_store() -> SessionStore | None:
    try:
        import redis  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError:
        logger.warning("redis package not installed; using in-memory session store")
        return None
    try:
        client = redis.Redis.from_url(settings.redis_url)
        client.ping()
    except Exception as exc:  # noqa: BLE001 - any connection error falls back
        logger.warning("Redis unavailable (%s); using in-memory session store", exc)
        return None
    logger.info("Using Redis session store at %s", settings.redis_url)
    return RedisSessionStore(client, settings.session_ttl_seconds)


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """Return the configured session store, falling back to in-memory."""

    if settings.session_backend == "redis":
        store = _build_redis_store()
        if store is not None:
            return store
    return InMemorySessionStore()
