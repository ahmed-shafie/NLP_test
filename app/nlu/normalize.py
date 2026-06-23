"""Shared Arabic/English text normalization for matching.

Used by the biller resolver and the name gazetteer so that lookups are robust to
diacritics, letter-form variants, elongations, and casing. Applied identically at
ingest time (when building the gazetteers) and at query time.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits to ASCII.
_DIGIT_MAP = {ord(c): str(i % 10) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")}

# Arabic diacritics / tashkeel and the tatweel elongation character.
_TASHKEEL = "".join(
    [
        "\u0610",
        "\u0611",
        "\u0612",
        "\u0613",
        "\u0614",
        "\u0615",
        "\u0616",
        "\u0617",
        "\u0618",
        "\u0619",
        "\u061a",
        "\u064b",
        "\u064c",
        "\u064d",
        "\u064e",
        "\u064f",
        "\u0650",
        "\u0651",
        "\u0652",
        "\u0653",
        "\u0654",
        "\u0655",
        "\u0656",
        "\u0657",
        "\u0658",
        "\u0670",
    ]
)
_TASHKEEL_RE = re.compile(f"[{_TASHKEEL}\u0640]")

# Letter-form unifications so spelling variants collapse to one key.
_LETTER_MAP = {
    ord("أ"): "ا",
    ord("إ"): "ا",
    ord("آ"): "ا",
    ord("ٱ"): "ا",
    ord("ى"): "ي",
    ord("ئ"): "ي",
    ord("ؤ"): "و",
    ord("ة"): "ه",
    ord("ـ"): None,  # tatweel
}

_REPEAT_RE = re.compile(r"(.)\1{2,}")  # 3+ identical chars -> 1 ("yesss" -> "yes")
_PUNCT_RE = re.compile(r"[^\w\u0600-\u06ff]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic digits to ASCII and normalise decimal separators."""

    return text.translate(_DIGIT_MAP).replace("٫", ".").replace("٬", ",")


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel and the tatweel elongation character."""

    return _TASHKEEL_RE.sub("", text)


def normalize(text: str) -> str:
    """Return a canonical matching key for ``text`` (EN + AR aware).

    Lowercases, strips Arabic diacritics, unifies alef/ya/hamza/ta-marbuta,
    removes tatweel, normalises digits, collapses 3+ char repeats, and reduces
    punctuation/whitespace to single spaces.
    """

    text = unicodedata.normalize("NFKC", text)
    text = normalize_digits(text)
    text = strip_diacritics(text)
    text = text.translate(_LETTER_MAP)
    text = text.lower()
    text = _REPEAT_RE.sub(r"\1", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_tokens(text: str) -> list[str]:
    """Return the normalized whitespace-separated tokens of ``text``."""

    normalized = normalize(text)
    return normalized.split() if normalized else []
