"""Semantic recipient/contact matching.

Embeds contact names into a FAISS index so a spoken/typed recipient (in either
Arabic or English) can be resolved to an address-book entry, including across
scripts (e.g. "Ahmed" -> "أحمد حسن"). When embeddings are unavailable it falls
back to fuzzy string matching via :mod:`difflib`.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from functools import lru_cache

from app.config import DEMO_CONTACTS, settings
from app.embeddings import get_embedder
from app.schemas import Contact, ContactMatch
from app.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)


class ContactMatcher:
    """Resolve a recipient name to contacts via embeddings (or fuzzy fallback)."""

    def __init__(self, contacts: list[Contact]) -> None:
        self._contacts = contacts
        self._embedder = get_embedder()
        self._store: FaissVectorStore[Contact] | None = None

        if self._embedder is not None and contacts:
            self._store = FaissVectorStore(self._embedder.dimension)
            vectors = self._embedder.encode([c.name for c in contacts])
            self._store.add(vectors, contacts)

    def _fuzzy(self, name: str, top_k: int) -> list[ContactMatch]:
        scored = [
            ContactMatch(
                contact=c,
                score=SequenceMatcher(None, name.lower(), c.name.lower()).ratio(),
            )
            for c in self._contacts
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def resolve(
        self, name: str, top_k: int = 3
    ) -> tuple[ContactMatch | None, list[ContactMatch]]:
        """Return ``(best_match_or_None, ranked_candidates)`` for ``name``.

        ``best_match`` is the top candidate only if its score clears
        ``contact_match_threshold``.
        """

        if not self._contacts:
            return None, []

        if self._embedder is not None and self._store is not None:
            query = self._embedder.encode_one(name)
            hits = self._store.search(query, top_k)
            candidates = [
                ContactMatch(contact=hit.payload, score=max(hit.score, 0.0))
                for hit in hits
            ]
        else:
            candidates = self._fuzzy(name, top_k)

        best = candidates[0] if candidates else None
        if best is not None and best.score < settings.contact_match_threshold:
            best = None
        return best, candidates


def _to_contacts(raw: list[dict[str, str]]) -> list[Contact]:
    return [Contact(**item) for item in raw]


@lru_cache(maxsize=1)
def get_default_matcher() -> ContactMatcher:
    """Build and cache a matcher over the demo address book."""

    return ContactMatcher(_to_contacts(DEMO_CONTACTS))
