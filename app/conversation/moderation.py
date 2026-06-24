"""Deterministic profanity / abuse ("ribaldry") detection.

Loads a small bilingual blocklist (English + Arabic) with a per-term severity
and flags user messages so the conversation engine can refuse gracefully, steer
the customer back to banking, and never let abusive text leak into a slot.

Matching runs on normalized tokens (case-, diacritic-, elongation- and
digit-form-insensitive via :func:`app.nlu.normalize.normalize`) with light
leetspeak folding so "sh1t" / "stup1d" still match. Single words are matched per
token; multi-word entries ("shut up") are matched as a normalized substring.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.nlu.normalize import normalize

_BLOCKLIST_CSV = Path(__file__).resolve().parent.parent / "data" / "blocklist.csv"

# Digit -> letter folding for leetspeak ("sh1t" -> "shit"). Symbol leet (@, $)
# is dropped by ``normalize`` as punctuation, so only digit forms are folded.
_LEET_MAP = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
)

# Severity ranking so a single severe hit dominates a mixed message.
_SEVERITY_RANK = {"mild": 1, "severe": 2}


@dataclass(frozen=True)
class ModerationResult:
    """Outcome of a moderation check for one message."""

    flagged: bool
    severity: str | None = None
    terms: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _blocklist() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(words, phrases)`` mapping normalized term -> severity."""

    words: dict[str, str] = {}
    phrases: dict[str, str] = {}
    if not _BLOCKLIST_CSV.exists():
        return words, phrases
    with _BLOCKLIST_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            term = normalize(row.get("term", ""))
            severity = (row.get("severity") or "mild").strip().lower()
            if not term or severity not in _SEVERITY_RANK:
                continue
            target = phrases if " " in term else words
            # Keep the most severe label if a term appears more than once.
            if _SEVERITY_RANK[severity] >= _SEVERITY_RANK.get(target.get(term, ""), 0):
                target[term] = severity
    return words, phrases


def _fold(token: str) -> set[str]:
    """Return the token plus its leetspeak-folded form."""

    return {token, token.translate(_LEET_MAP)}


def detect(text: str) -> ModerationResult:
    """Flag ``text`` if it contains blocklisted (abusive) language."""

    if not settings.moderation_enabled or not text:
        return ModerationResult(False)

    words, phrases = _blocklist()
    if not words and not phrases:
        return ModerationResult(False)

    hits: list[tuple[str, str]] = []  # (display term, severity)

    # Single words: match each original token by its normalized / leet-folded form
    # so we can surface the user's own spelling for the UI highlight.
    for raw in text.split():
        normalized = normalize(raw)
        if not normalized:
            continue
        for candidate in _fold(normalized):
            severity = words.get(candidate)
            if severity is not None:
                hits.append((raw.strip("\"'.,!؟،:;()[]"), severity))
                break

    # Multi-word entries: substring match on the normalized message.
    normalized_text = normalize(text)
    for phrase, severity in phrases.items():
        if phrase and phrase in normalized_text:
            hits.append((phrase, severity))

    if not hits:
        return ModerationResult(False)

    severity = (
        "severe"
        if any(sev == "severe" for _, sev in hits)
        else "mild"
    )
    # De-duplicate display terms while preserving order.
    seen: set[str] = set()
    terms: list[str] = []
    for term, _ in hits:
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return ModerationResult(True, severity, tuple(terms))


def is_clean(text: str) -> bool:
    """True when ``text`` contains no blocklisted language."""

    return not detect(text).flagged


def _reset_cache() -> None:
    """Drop the cached blocklist (used by tests after patching the CSV)."""

    _blocklist.cache_clear()


__all__ = ["ModerationResult", "detect", "is_clean"]
