"""Load reference datasets (SADAD billers, name gazetteer) from committed CSVs.

Both datasets are small enough to live entirely in memory:
- ``sadad_billers.csv`` -> a gazetteer (exact name/alias -> biller) plus an
  optional FAISS index for fuzzy/semantic fallback.
- ``names.csv`` -> a normalized name set + an Arabic<->English transliteration
  map, used as an extraction aid for the recipient slot.

All lookups go through :mod:`app.nlu.normalize` so matching is robust to Arabic
diacritics, letter-form variants, and casing.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process

from app.config import settings
from app.nlu.normalize import normalize, normalize_tokens

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_BILLERS_CSV = _DATA_DIR / "sadad_billers.csv"
_NAMES_CSV = _DATA_DIR / "names.csv"

# Generic words customers use mapped to the SADAD billers they could mean,
# canonical/most-common first. A term with several candidates (e.g. "electricity"
# -> Saudi Electric Company *or* Marafiq) is ambiguous: the conversation engine
# asks the customer which one instead of silently guessing. Terms that map to a
# single code resolve directly.
_GENERIC_BILLER_GROUPS: dict[str, tuple[str, ...]] = {
    "electricity": ("002", "004"),
    "power": ("002", "004"),
    "كهرباء": ("002", "004"),
    "الكهرباء": ("002", "004"),
    "water": ("015", "138"),
    "مياه": ("015", "138"),
    "المياه": ("015", "138"),
    "مية": ("015", "138"),
    "gas": ("148",),
    "غاز": ("148",),
    "الغاز": ("148",),
}


@dataclass(frozen=True)
class BillerRecord:
    """A single SADAD biller."""

    biller_code: str
    name_en: str
    name_ar: str
    category: str


@lru_cache(maxsize=1)
def load_billers() -> tuple[BillerRecord, ...]:
    """Read the SADAD biller catalogue from the committed CSV."""

    if not _BILLERS_CSV.exists():
        logger.warning("SADAD billers CSV missing at %s", _BILLERS_CSV)
        return ()
    records: list[BillerRecord] = []
    with _BILLERS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("biller_code") or "").strip()
            if not code:
                continue
            records.append(
                BillerRecord(
                    biller_code=code,
                    name_en=(row.get("name_en") or "").strip(),
                    name_ar=(row.get("name_ar") or "").strip(),
                    category=(row.get("category") or "").strip(),
                )
            )
    return tuple(records)


@lru_cache(maxsize=1)
def _biller_by_code() -> dict[str, BillerRecord]:
    return {r.biller_code: r for r in load_billers()}


@lru_cache(maxsize=1)
def _biller_name_index() -> list[tuple[tuple[str, ...], BillerRecord]]:
    """Normalized name-token tuples paired with their biller, longest first.

    Matching a contiguous token sub-sequence (rather than a raw substring)
    avoids short codes like "go" matching unrelated words.
    """

    index: list[tuple[tuple[str, ...], BillerRecord]] = []
    for rec in load_billers():
        for name in (rec.name_en, rec.name_ar):
            tokens = tuple(normalize_tokens(name))
            if tokens:
                index.append((tokens, rec))
    # Longer names first so the most specific match wins.
    index.sort(key=lambda item: len(item[0]), reverse=True)
    return index


def _contains_subsequence(haystack: list[str], needle: tuple[str, ...]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    for i in range(len(haystack) - n + 1):
        if tuple(haystack[i : i + n]) == needle:
            return True
    return False


def resolve_biller_gazetteer(text: str) -> BillerRecord | None:
    """Exact resolution via biller names then curated generic aliases.

    For an ambiguous generic term the canonical (first) candidate is returned, so
    the stateless single-result path stays deterministic; the conversation engine
    uses :func:`resolve_biller_candidates` to offer the full choice instead.
    """

    candidates = _gazetteer_candidates(text)
    return candidates[0] if candidates else None


def _gazetteer_candidates(text: str) -> list[BillerRecord]:
    """Exact name match (one record) or a generic-term group (one or several)."""

    tokens = normalize_tokens(text)
    if not tokens:
        return []
    for name_tokens, rec in _biller_name_index():
        if _contains_subsequence(tokens, name_tokens):
            return [rec]
    by_code = _biller_by_code()
    token_set = set(tokens)
    for term, codes in _GENERIC_BILLER_GROUPS.items():
        if normalize(term) in token_set:
            recs = [by_code[c] for c in codes if c in by_code]
            if recs:
                return recs
    return []


def resolve_biller_candidates(
    text: str, *, allow_semantic: bool = False
) -> list[BillerRecord]:
    """Return every SADAD biller ``text`` could mean (ordered, most likely first).

    Returns ``[]`` when nothing matches, ``[rec]`` for an unambiguous hit (exact
    name, single-candidate generic term, or a FAISS fallback when
    ``allow_semantic`` is set), and several records when a generic term is
    ambiguous (e.g. "electricity" -> Saudi Electric Company or Marafiq) so the
    caller can ask the customer to choose.
    """

    if not settings.biller_catalog_enabled:
        return []
    candidates = _gazetteer_candidates(text)
    if candidates:
        return candidates
    if allow_semantic:
        rec = resolve_biller_semantic(text)
        if rec is not None:
            return [rec]
    return []


# ---- FAISS fuzzy/semantic fallback for billers --------------------------- #

_biller_store = None
_biller_store_built = False


def _build_biller_store():
    from app.embeddings import get_embedder
    from app.vectorstore import FaissVectorStore

    embedder = get_embedder()
    records = load_billers()
    if embedder is None or not records:
        return None
    texts = [f"{r.name_en} | {r.name_ar}" for r in records]
    store: FaissVectorStore[BillerRecord] = FaissVectorStore(embedder.dimension)
    store.add(embedder.encode(texts), list(records))
    return store


def resolve_biller_semantic(text: str) -> BillerRecord | None:
    """Nearest-neighbour biller match above the configured threshold."""

    global _biller_store, _biller_store_built
    if not _biller_store_built:
        try:
            _biller_store = _build_biller_store()
        except Exception as exc:  # noqa: BLE001 - degrade to gazetteer-only
            logger.warning("Biller FAISS index unavailable (%s).", exc)
            _biller_store = None
        _biller_store_built = True
    if _biller_store is None:
        return None
    from app.embeddings import get_embedder

    embedder = get_embedder()
    if embedder is None:
        return None
    hits = _biller_store.search(embedder.encode_one(normalize(text)), 1)
    if hits and hits[0].score >= settings.biller_match_threshold:
        return hits[0].payload
    return None


def resolve_biller(text: str, *, allow_semantic: bool = False) -> BillerRecord | None:
    """Resolve a biller: exact gazetteer first, then optional FAISS fallback.

    The FAISS fallback is only consulted when ``allow_semantic`` is set (i.e. we
    already know we're in a bill context). This keeps arbitrary chit-chat like
    "hi" from being mis-resolved to a near-neighbour biller.
    """

    if not settings.biller_catalog_enabled:
        return None
    gazetteer = resolve_biller_gazetteer(text)
    if gazetteer is not None:
        return gazetteer
    if allow_semantic:
        return resolve_biller_semantic(text)
    return None


# ---- Name gazetteer ------------------------------------------------------ #


@lru_cache(maxsize=1)
def _load_names() -> dict[str, str]:
    """Return ``normalized name form -> canonical display name`` (same script).

    English and Arabic forms each map to their own display surface, so spelling
    correction never silently transliterates an Arabic name into English (and
    vice-versa); it only fixes the spelling within the input's own script.
    """

    known: dict[str, str] = {}
    if not _NAMES_CSV.exists():
        logger.warning("Names CSV missing at %s", _NAMES_CSV)
        return known
    with _NAMES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            en = (row.get("name_en") or "").strip()
            ar = (row.get("name_ar") or "").strip()
            if en:
                known.setdefault(normalize(en), en)
            if ar:
                known.setdefault(normalize(ar), ar)
    return known


@lru_cache(maxsize=1)
def _name_keys() -> list[str]:
    return list(_load_names().keys())


@lru_cache(maxsize=1)
def _transliteration_map() -> dict[str, frozenset[str]]:
    """``normalized name form -> its equivalents in the other script``.

    Built from the English/Arabic pairs in ``names.csv`` so a person name written
    in one script can be matched against the same name written in the other
    (e.g. "mohammed" <-> "محمد").
    """

    pairs: dict[str, set[str]] = {}
    if not _NAMES_CSV.exists():
        return {}
    with _NAMES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            en = normalize((row.get("name_en") or "").strip())
            ar = normalize((row.get("name_ar") or "").strip())
            if en and ar:
                pairs.setdefault(en, set()).add(ar)
                pairs.setdefault(ar, set()).add(en)
    return {key: frozenset(values) for key, values in pairs.items()}


def transliterations(token: str) -> set[str]:
    """Return the normalized cross-script equivalents of a single name token.

    Empty when the gazetteer is disabled or the token is not a recognised name.
    """

    if not settings.names_gazetteer_enabled:
        return set()
    key = normalize(token)
    if not key:
        return set()
    return set(_transliteration_map().get(key, frozenset()))


def lookup_name(token: str) -> str | None:
    """Canonical (same-script) name for a single token: exact then fuzzy typo."""

    if not settings.names_gazetteer_enabled:
        return None
    known = _load_names()
    key = normalize(token)
    if not key:
        return None
    if key in known:
        return known[key]
    if len(key) >= 3:
        match = process.extractOne(
            key, _name_keys(), scorer=fuzz.ratio, score_cutoff=settings.name_match_score
        )
        if match is not None:
            return known[match[0]]
    return None


def is_known_name(text: str) -> bool:
    """True when any token of ``text`` is a recognised given name."""

    return any(lookup_name(tok) is not None for tok in normalize_tokens(text))


def canonicalize_recipient(candidate: str) -> str:
    """Correct each token of a recipient against the gazetteer where possible.

    Unknown tokens are kept as-is so unusual names still pass through; only the
    spelling of recognised given names is normalised.
    """

    if not settings.names_gazetteer_enabled:
        return candidate
    out: list[str] = []
    for raw in candidate.split():
        canonical = lookup_name(raw)
        out.append(canonical if canonical is not None else raw)
    return " ".join(out) if out else candidate
