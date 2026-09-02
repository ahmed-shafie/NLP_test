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
from rapidfuzz.distance import Levenshtein

from app.config import settings
from app.nlu.normalize import (
    normalize,
    normalize_digits,
    normalize_tokens,
    strip_diacritics,
    strip_proclitic,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_BILLERS_CSV = _DATA_DIR / "sadad_billers.csv"
_NAMES_CSV = _DATA_DIR / "names.csv"

# Generic words customers use mapped to the SADAD billers they could mean,
# canonical/most-common first. Used for utility sub-types (electricity / water /
# gas) which all share the single coarse "Utilities" SADAD category, so the
# category alone can't separate them. A term with several candidates (e.g.
# "electricity" -> Saudi Electric Company *or* Marafiq) is ambiguous: the engine
# asks the customer which one instead of silently guessing.
_GENERIC_BILLER_GROUPS: dict[str, tuple[str, ...]] = {
    "electricity": ("002", "004"),
    "power": ("002", "004"),
    "كهرباء": ("002", "004"),
    "الكهرباء": ("002", "004"),
    "water": ("015", "138"),
    "مياه": ("015", "138"),
    "المياه": ("015", "138"),
    "مية": ("015", "138"),
    # Colloquial Gulf spellings of "مياه".
    "موية": ("015", "138"),
    "الموية": ("015", "138"),
    "مويه": ("015", "138"),
    "المويه": ("015", "138"),
    "gas": ("148",),
    "غاز": ("148",),
    "الغاز": ("148",),
}

# Generic words mapped to a whole SADAD *category*: every biller in that category
# is offered. E.g. "internet" -> all "Telecom & Internet" billers (STC, Mobily,
# Zain, ...) so the customer picks, instead of only matching names that literally
# contain the word "internet".
_GENERIC_CATEGORY_TERMS: dict[str, str] = {
    "internet": "Telecom & Internet",
    "wifi": "Telecom & Internet",
    "نت": "Telecom & Internet",
    "النت": "Telecom & Internet",
    "انترنت": "Telecom & Internet",
    "إنترنت": "Telecom & Internet",
    "mobile": "Telecom & Internet",
    "phone": "Telecom & Internet",
    "موبايل": "Telecom & Internet",
    "الموبايل": "Telecom & Internet",
    "جوال": "Telecom & Internet",
    "الجوال": "Telecom & Internet",
    "insurance": "Insurance",
    "تأمين": "Insurance",
    "التأمين": "Insurance",
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
def _billers_by_category() -> dict[str, tuple[BillerRecord, ...]]:
    """``normalized category -> billers in it`` (ordered by code)."""

    groups: dict[str, list[BillerRecord]] = {}
    for rec in load_billers():
        if rec.category:
            groups.setdefault(normalize(rec.category), []).append(rec)
    return {
        key: tuple(sorted(recs, key=lambda r: r.biller_code))
        for key, recs in groups.items()
    }


@lru_cache(maxsize=1)
def biller_categories() -> tuple[str, ...]:
    """The distinct SADAD categories in the catalogue, alphabetically."""

    return tuple(sorted({r.category for r in load_billers() if r.category}))


def _as_biller_code(token: str) -> str | None:
    """Return the SADAD code a short numeric token denotes, else ``None``.

    SADAD codes are 1-3 digit, zero-padded to three in the catalogue ("001",
    "153"). Longer digit runs are reference numbers, never codes.
    """

    digits = "".join(ch for ch in normalize_digits(token) if ch.isdigit())
    if not digits or len(digits) > 3:
        return None
    by_code = _biller_by_code()
    for candidate in (digits, digits.zfill(3)):
        if candidate in by_code:
            return candidate
    return None


def resolve_biller_by_code(token: str) -> BillerRecord | None:
    """Resolve a short numeric token (1-3 digits) to its SADAD biller."""

    if not settings.biller_catalog_enabled:
        return None
    code = _as_biller_code(token)
    return _biller_by_code().get(code) if code else None


# Extra spellings customers use for a specific biller, keyed by SADAD code.
# Mostly Latin brand names written out letter-by-letter in Arabic ("اس تي سي"),
# which no amount of normalization derives from the catalogue name.
_BILLER_ALIASES: dict[str, tuple[str, ...]] = {
    "001": ("اس تي سي", "اس تي سى", "الاتصالات السعودية", "stc"),
    "005": ("موبايلي", "mobily"),
    "044": ("زين", "زين السعودية", "zain"),
    "151": ("فيرجن", "فيرجن موبايل"),
    "207": ("stc pay", "اس تي سي باي"),
}


@lru_cache(maxsize=1)
def _biller_name_index() -> list[tuple[tuple[str, ...], BillerRecord]]:
    """Normalized name-token tuples paired with their biller, longest first.

    Matching a contiguous token sub-sequence (rather than a raw substring)
    avoids short codes like "go" matching unrelated words.
    """

    index: list[tuple[tuple[str, ...], BillerRecord]] = []
    for rec in load_billers():
        names = (rec.name_en, rec.name_ar, *_BILLER_ALIASES.get(rec.biller_code, ()))
        for name in names:
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


def _unprefixed_tokens(tokens: list[str]) -> list[str]:
    """Drop a fused Arabic proclitic from each token (keeping 2-letter words)."""

    return [strip_proclitic(token) for token in tokens]


def _gazetteer_candidates(text: str) -> list[BillerRecord]:
    """Exact name match (one record) or a generic-term group (one or several)."""

    tokens = normalize_tokens(text)
    if not tokens:
        return []
    index = _biller_name_index()
    for name_tokens, rec in index:
        if _contains_subsequence(tokens, name_tokens):
            return [rec]
    # Retry once with fused Arabic proclitics removed.
    stripped = _unprefixed_tokens(tokens)
    if stripped != tokens:
        for name_tokens, rec in index:
            if _contains_subsequence(stripped, name_tokens):
                return [rec]
    by_code = _biller_by_code()
    token_set = set(tokens) | set(stripped)
    for term, codes in _GENERIC_BILLER_GROUPS.items():
        if normalize(term) in token_set:
            recs = [by_code[c] for c in codes if c in by_code]
            if recs:
                return recs
    by_category = _billers_by_category()
    for term, category in _GENERIC_CATEGORY_TERMS.items():
        if normalize(term) in token_set:
            recs = list(by_category.get(normalize(category), ()))
            if recs:
                return recs
    return []


def resolve_biller_candidates(
    text: str, *, allow_semantic: bool = False
) -> list[BillerRecord]:
    """Return every SADAD biller ``text`` could mean (ordered, most likely first).

    Returns ``[]`` when nothing matches, ``[rec]`` for an unambiguous hit (exact
    name, single-candidate generic term, or a typo/semantic fallback when
    ``allow_semantic`` is set), and several records when a generic term is
    ambiguous (e.g. "electricity" -> Saudi Electric Company or Marafiq, or
    "internet" -> all Telecom & Internet billers) so the caller can ask the
    customer to choose.
    """

    if not settings.biller_catalog_enabled:
        return []
    candidates = _gazetteer_candidates(text)
    if candidates:
        return candidates
    if allow_semantic:
        misspelt = _misspelt_generic_candidates(text)
        if misspelt:
            return misspelt
        rec = resolve_biller_fuzzy(text)
        if rec is None and settings.biller_semantic_enabled:
            rec = resolve_biller_semantic(text)
        if rec is not None:
            return [rec]
    return []


# ---- rapidfuzz typo-tolerant biller matching ---------------------------- #

# Stopwords stripped from a query before fuzzy matching so they can't typo-match
# a biller name (e.g. "pay" should never fuzzy-resolve to a biller).
_BILLER_QUERY_STOPWORDS: frozenset[str] = frozenset(
    {
        "pay",
        "bill",
        "bills",
        "my",
        "the",
        "a",
        "an",
        "your",
        "our",
        "this",
        "for",
        "فاتورة",
        "فواتير",
        "ادفع",
        "دفع",
    }
)
# Minimum token length considered for a fuzzy match (shorter tokens are too
# collision-prone for a single-edit typo rule).
_FUZZY_MIN_LEN = 4


# Longest edit distance tolerated between a token and a generic bill term, so a
# single slipped letter ("قهرباء" for "كهرباء") still reaches the utility.
_GENERIC_TERM_MAX_DISTANCE = 1


@lru_cache(maxsize=1)
def _generic_terms() -> tuple[str, ...]:
    """Normalized generic bill words ("كهرباء", "internet", ...)."""

    terms = {normalize(t) for t in _GENERIC_BILLER_GROUPS}
    terms |= {normalize(t) for t in _GENERIC_CATEGORY_TERMS}
    return tuple(sorted(t for t in terms if len(t) >= _FUZZY_MIN_LEN))


def _misspelt_generic_candidates(text: str) -> list[BillerRecord]:
    """Candidates for a generic bill word the customer misspelt.

    "قهرباء" is "كهرباء" with one slipped letter, and the customer means the
    electricity bill; only tokens long enough for a single edit to stay
    unambiguous are corrected, and the correction still goes through the normal
    generic-term path, so an ambiguous word keeps asking which biller.
    """

    terms = _generic_terms()
    if not terms:
        return []
    for token in normalize_tokens(text):
        if len(token) < _FUZZY_MIN_LEN or token in terms:
            continue
        near = [
            term
            for term in terms
            if Levenshtein.distance(token, term) <= _GENERIC_TERM_MAX_DISTANCE
        ]
        if len(near) == 1:
            candidates = _gazetteer_candidates(near[0])
            if candidates:
                return candidates
    return []


@lru_cache(maxsize=1)
def _biller_fuzzy_index() -> tuple[tuple[str, BillerRecord], ...]:
    """``(normalized full name, record)`` pairs for typo matching.

    Only whole names are indexed (not sub-tokens) so a common word shared by many
    billers ("saudi", "company") can't fuzzy-match an unrelated biller; a query
    only matches a biller whose *entire* name is close to it.
    """

    index: list[tuple[str, BillerRecord]] = []
    seen: set[tuple[str, str]] = set()
    for rec in load_billers():
        for name in (rec.name_en, rec.name_ar):
            key = normalize(name)
            if key and (key, rec.biller_code) not in seen:
                index.append((key, rec))
                seen.add((key, rec.biller_code))
    return tuple(index)


def resolve_biller_fuzzy(text: str) -> BillerRecord | None:
    """Best typo-tolerant biller match, or ``None``.

    Accepts a candidate (the stopword-stripped phrase, or any single token) when
    its edit distance to a biller name is within
    ``settings.biller_fuzzy_max_distance`` or its rapidfuzz ratio clears
    ``settings.biller_fuzzy_min_ratio``. Closest match wins.
    """

    if not (settings.biller_catalog_enabled and settings.biller_fuzzy_enabled):
        return None
    tokens = [t for t in normalize_tokens(text) if t not in _BILLER_QUERY_STOPWORDS]
    candidates = {t for t in tokens if len(t) >= _FUZZY_MIN_LEN and not t.isdigit()}
    phrase = " ".join(tokens)
    if len(phrase) >= _FUZZY_MIN_LEN:
        candidates.add(phrase)
    if not candidates:
        return None
    max_distance = settings.biller_fuzzy_max_distance
    best: BillerRecord | None = None
    best_rank: tuple[int, float] = (max_distance + 1, 0.0)
    for candidate in candidates:
        for key, rec in _biller_fuzzy_index():
            distance = Levenshtein.distance(candidate, key)
            ratio = fuzz.ratio(candidate, key)
            if distance <= max_distance or ratio >= settings.biller_fuzzy_min_ratio:
                rank = (min(distance, max_distance), -ratio)
                if rank < best_rank:
                    best_rank = rank
                    best = rec
    return best


# ---- FAISS semantic fallback for billers (off by default) --------------- #

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
    """Resolve a single biller: exact gazetteer first, then a typo fallback.

    The typo (and optional semantic) fallback is only consulted when
    ``allow_semantic`` is set (i.e. we already know we're in a bill context).
    This keeps arbitrary chit-chat like "hi" from being mis-resolved to a
    near-neighbour biller.
    """

    if not settings.biller_catalog_enabled:
        return None
    gazetteer = resolve_biller_gazetteer(text)
    if gazetteer is not None:
        return gazetteer
    if allow_semantic:
        misspelt = _misspelt_generic_candidates(text)
        if misspelt:
            return misspelt[0]
        fuzzy = resolve_biller_fuzzy(text)
        if fuzzy is not None:
            return fuzzy
        if settings.biller_semantic_enabled:
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
        return _unambiguous_name_match(key, known)
    return None


def _unambiguous_name_match(key: str, known: dict[str, str]) -> str | None:
    """The one name ``key`` is a typo of, or ``None`` when that is unclear.

    A typo correction may fix a spelling but must never change *who* is meant,
    and a near-miss on a name list of this size is usually several different
    names at once: "noura" is one edit from nouran, nora, nour and nura. So the
    winner has to stand clear of the runner-up by ``name_match_margin``;
    otherwise the customer's own spelling is kept and the name stays whatever
    they typed.
    """

    matches = process.extract(
        key,
        _name_keys(),
        scorer=fuzz.ratio,
        score_cutoff=settings.name_match_score,
        limit=3,
    )
    if not matches:
        return None
    winner = known[str(matches[0][0])]
    rivals = [
        score
        for candidate, score, _ in matches[1:]
        # Two keys that canonicalise to the same name (احمد / أحمد) do not
        # disagree about who is meant.
        if known[str(candidate)] != winner
    ]
    if rivals and matches[0][1] - rivals[0] < settings.name_match_margin:
        return None
    return winner


def is_known_name(text: str) -> bool:
    """True when any token of ``text`` is a recognised given name."""

    return any(lookup_name(tok) is not None for tok in normalize_tokens(text))


def _preferred_spelling(raw: str, canonical: str) -> str:
    """Choose between the customer's spelling and the gazetteer's canonical one.

    The gazetteer restores hamza and fixes typos (احمد -> أحمد), which we want,
    but some of its entries are the worse spelling of a pair: they carry
    tashkeel (عادل -> عَادِل) or swap a final haa for taa marbuta
    (عبدالله -> عبداللة). Those two we decline, since the name is echoed back
    to the customer.
    """

    canonical = strip_diacritics(canonical)
    ends_haa = {raw[-1:], canonical[-1:]} <= {"ه", "ة"}
    if ends_haa and raw[:-1] == canonical[:-1]:
        return raw
    return canonical


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
        out.append(_preferred_spelling(raw, canonical) if canonical else raw)
    return " ".join(out) if out else candidate
