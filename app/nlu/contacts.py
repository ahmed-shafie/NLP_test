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
from app.data_loader import transliterations
from app.embeddings import get_embedder
from app.nlu.normalize import normalize, normalize_tokens
from app.schemas import Contact, ContactMatch
from app.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)

# Candidates the name gate is applied to. Wider than the caller's ``top_k``
# because the ranking is only a shortlist: the right record can sit behind its
# own cross-script duplicate.
_GATE_WINDOW = 10


def _variants(token: str) -> set[str]:
    """The token plus its spelling in the other script ("ahmed" -> "احمد")."""

    key = normalize(token)
    if not key:
        return set()
    return {key} | transliterations(key)


def _name_is_contained(name: str, contact: Contact) -> bool:
    """True when every name the customer typed appears in ``contact``'s name.

    Embedding similarity ranks candidates but cannot decide identity: in this
    address book "محمد نور" sits 0.81 from the record of *محمد علي* - closer than
    "Ahmed" sits to his own record - so a score alone would hand a transfer to a
    different person. Requiring the typed tokens to be present keeps the useful
    matches (a first name, or the other script's spelling) and refuses the
    look-alikes; a missing token means "ask", not "guess". Typos are deliberately
    not tolerated here - one edit separates "احمد" from "محمد" - they are already
    normalised upstream by the name gazetteer.
    """

    contact_tokens = frozenset(normalize_tokens(contact.name))
    if not contact_tokens:
        return False
    tokens = normalize_tokens(name)
    return bool(tokens) and all(_variants(token) & contact_tokens for token in tokens)


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

        ``best_match`` is set only when a single person clears
        ``contact_match_threshold`` *and* actually carries the name the customer
        typed. Several people carrying it (two Ahmeds) also gives ``None``: which
        one is a question for the customer, not for a tie-break.

        ``ranked_candidates`` stays the raw similarity ranking so callers can show
        what was considered; it is not a list of accepted matches.
        """

        if not self._contacts:
            return None, []

        window = max(top_k, _GATE_WINDOW)
        if self._embedder is not None and self._store is not None:
            query = self._embedder.encode_one(name)
            hits = self._store.search(query, window)
            ranked = [
                ContactMatch(contact=hit.payload, score=min(max(hit.score, 0.0), 1.0))
                for hit in hits
            ]
        else:
            ranked = self._fuzzy(name, window)

        accepted = [
            m
            for m in ranked
            if m.score >= settings.contact_match_threshold
            and _name_is_contained(name, m.contact)
        ]
        # The same person is listed once per script, so identity is the account.
        people = {m.contact.account or m.contact.id for m in accepted}
        best = accepted[0] if len(people) == 1 else None
        return best, ranked[:top_k]


def _to_contacts(raw: list[dict[str, str]]) -> list[Contact]:
    return [Contact(**item) for item in raw]


@lru_cache(maxsize=1)
def get_default_matcher() -> ContactMatcher:
    """Build and cache a matcher over the demo address book."""

    return ContactMatcher(_to_contacts(DEMO_CONTACTS))
