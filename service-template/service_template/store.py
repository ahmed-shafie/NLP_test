"""Session store — persists ``ConversationState`` between turns, keyed by id.

The production app persists state to SQLite via the "Memory Brain"
(``app/conversation/store.py``). To keep the template dependency-free this uses
a simple in-process dict. Swap ``InMemorySessionStore`` for a Redis/SQL-backed
implementation with the same two methods (``load`` / ``save``) for production.
"""

from __future__ import annotations

from functools import lru_cache

from service_template.state import ConversationState


class InMemorySessionStore:
    """Process-local store. State is lost on restart (fine for demos/tests)."""

    def __init__(self) -> None:
        self._data: dict[str, ConversationState] = {}

    def load(self, session_id: str) -> ConversationState | None:
        state = self._data.get(session_id)
        # Return a copy so the engine mutates a fresh object each turn; only an
        # explicit ``save`` persists changes (mirrors the DB-backed store).
        return state.model_copy(deep=True) if state is not None else None

    def save(self, state: ConversationState) -> None:
        self._data[state.session_id] = state.model_copy(deep=True)


@lru_cache(maxsize=1)
def get_session_store() -> InMemorySessionStore:
    """Return the process-wide singleton store."""

    return InMemorySessionStore()
