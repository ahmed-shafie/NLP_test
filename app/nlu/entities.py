"""Language-agnostic slot extraction helpers (amount, currency, recipient).

These regex-based extractors work without any downloaded model and are used as a
baseline. The per-language modules (:mod:`app.nlu.english`, :mod:`app.nlu.arabic`)
augment recipient detection with spaCy/Stanza named-entity recognition.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.config import CURRENCY_SYMBOLS, SUPPORTED_CURRENCIES
from app.schemas import Language

# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digit translation to ASCII.
_DIGIT_MAP = {ord(c): str(i % 10) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")}

# Multipliers for shorthand magnitudes (English + Arabic).
_MULTIPLIERS: dict[str, int] = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "الف": 1_000,
    "ألف": 1_000,
    "آلاف": 1_000,
    "مليون": 1_000_000,
}

_AMOUNT_RE = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<mult>k|m|thousand|million|الف|ألف|آلاف|مليون)?",
    re.IGNORECASE,
)

# "to John", "to my friend Sara"
_EN_RECIPIENT_RE = re.compile(
    r"\bto\s+(?:my\s+)?(?:friend\s+|account\s+|number\s+)?([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,2})",
)
# "إلى أحمد" / "الى احمد" / "لأحمد" (ل only when preceded by whitespace/start)
_AR_RECIPIENT_RE = re.compile(
    r"(?:(?:إلى|الى)\s+|(?:(?<=\s)|^)ل)([^\d،,.]{2,40}?)(?:\s+(?:مبلغ|بمبلغ)|$|[،,.])"
)

_EN_SOURCE_RE = re.compile(
    r"\bfrom\s+(?:my\s+)?([\w'’-]+(?:\s+[\w'’-]+){0,2}?)\s*account\b", re.IGNORECASE
)
_AR_SOURCE_RE = re.compile(r"من\s+حساب(?:ي)?\s*(?:ال)?([^\d،,.]{0,20})")


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic digits to ASCII and normalise the decimal separator."""

    return text.translate(_DIGIT_MAP).replace("٫", ".").replace("٬", ",")


def extract_amount(text: str) -> Decimal | None:
    """Return the first monetary amount found, applying magnitude multipliers."""

    normalized = normalize_digits(text)
    match = _AMOUNT_RE.search(normalized)
    if not match:
        return None
    raw = match.group("num").replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    mult = match.group("mult")
    if mult:
        value *= _MULTIPLIERS[mult.lower()]
    return value


def extract_currency(text: str) -> str | None:
    """Return the ISO-4217 code referenced in ``text``, if any."""

    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code

    lowered = normalize_digits(text).lower()
    tokens = set(re.findall(r"[^\W\d_]+", lowered, re.UNICODE))
    for code, aliases in SUPPORTED_CURRENCIES.items():
        for alias in aliases:
            alias_l = alias.lower()
            if " " in alias_l:
                if alias_l in lowered:
                    return code
            elif alias_l in tokens:
                return code
    return None


def extract_recipient(text: str, language: Language) -> str | None:
    """Return the beneficiary name via language-specific surface patterns."""

    pattern = _AR_RECIPIENT_RE if language is Language.AR else _EN_RECIPIENT_RE
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip(" ,،.") or None


def extract_source_account(text: str, language: Language) -> str | None:
    """Return the source account hint (e.g. "savings"), if mentioned."""

    pattern = _AR_SOURCE_RE if language is Language.AR else _EN_SOURCE_RE
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip(" ,،.") or None
