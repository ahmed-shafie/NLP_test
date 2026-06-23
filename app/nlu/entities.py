"""Language-agnostic slot extraction helpers (amount, currency, recipient).

These regex-based extractors work without any downloaded model and are used as a
baseline. The per-language modules (:mod:`app.nlu.english`, :mod:`app.nlu.arabic`)
augment recipient detection with spaCy/Stanza named-entity recognition.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.config import BILLER_CATEGORIES, CURRENCY_SYMBOLS, SUPPORTED_CURRENCIES
from app.schemas import BillEntities, Language

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


# ---- Bill-payment slot extraction --------------------------------------- #

# A reference number following an explicit cue ("bill 778899", "ref 4455", "رقم ٩٩").
_REF_CUE_RE = re.compile(
    r"(?:\b(?:ref|reference|number|no|bill|account|invoice)\b|رقم|مرجع|فاتورة)"
    r"\s*[:#\-]?\s*(\d{2,})",
    re.IGNORECASE,
)
# An amount following an explicit cue ("amount 320", "بمبلغ ٣٢٠").
_AMOUNT_CUE_RE = re.compile(
    r"(?:\bamount\b|بمبلغ|مبلغ)\s*[:#]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
_DIGITS_RUN_RE = re.compile(r"\d{2,}")
_BILL_WORD_RE = re.compile(r"\bbills?\b|فاتورة|فواتير", re.IGNORECASE)
# Free-text biller before the word "bill" (e.g. "City Power Co bill").
_EN_BILLER_RE = re.compile(
    r"([A-Za-z][\w&'’.-]*(?:\s+[A-Za-z][\w&'’.-]*){0,3})\s+bills?\b", re.IGNORECASE
)
# Free-text biller after "فاتورة" (e.g. "فاتورة شركة الكهرباء").
_AR_BILLER_RE = re.compile(r"فاتورة\s+([^\d،,.]{2,30})")
_BILLER_STOPWORDS = {"my", "the", "a", "an", "your", "our", "this", "pay"}


def _strip_biller_stopwords(value: str) -> str:
    words = [w for w in value.split() if w.lower() not in _BILLER_STOPWORDS]
    return " ".join(words).strip(" ,،.")


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _amount_digits(amount: Decimal | None) -> str | None:
    if amount is None:
        return None
    return f"{amount.normalize():f}"


def _reference_via_cue(normalized: str) -> str | None:
    match = _REF_CUE_RE.search(normalized)
    return match.group(1) if match else None


def _bill_amount(
    normalized: str, currency: str | None, reference: str | None
) -> Decimal | None:
    """Amount for a bill: only via an explicit cue or adjacent to a currency.

    A bare number with no currency/cue is treated as the reference, not the
    amount, so "electricity bill 778899" doesn't read 778899 as the amount.
    """

    cue = _AMOUNT_CUE_RE.search(normalized)
    if cue is not None:
        return _to_decimal(cue.group(1))
    if currency:
        for run in re.findall(r"\d+(?:\.\d+)?", normalized):
            if reference is None or run != reference:
                return _to_decimal(run)
    return None


def extract_biller(text: str, language: Language) -> tuple[str | None, str | None]:
    """Return ``(biller, category)`` for a bill utterance.

    ``category`` is a canonical name from :data:`BILLER_CATEGORIES` when a keyword
    matches; otherwise the free-text biller before "bill"/after "فاتورة" is used.
    """

    lowered = normalize_digits(text).lower()
    for category, keywords in BILLER_CATEGORIES.items():
        if any(kw in lowered for kw in keywords):
            return category, category
    match = _EN_BILLER_RE.search(text) or _AR_BILLER_RE.search(text)
    if match:
        biller = _strip_biller_stopwords(match.group(1))
        return (biller or None), None
    return None, None


def extract_reference_number(text: str) -> str | None:
    """Return a bill reference: an explicit-cue number, else the first digit run."""

    normalized = normalize_digits(text)
    cue = _reference_via_cue(normalized)
    if cue is not None:
        return cue
    run = _DIGITS_RUN_RE.search(normalized)
    return run.group(0) if run else None


def has_bill_word(text: str) -> bool:
    return bool(_BILL_WORD_RE.search(text))


def extract_bill_entities(text: str, language: Language) -> BillEntities:
    """Extract all bill slots (biller, reference, amount, currency) from ``text``."""

    normalized = normalize_digits(text)
    biller, category = extract_biller(text, language)
    currency = extract_currency(normalized)
    reference = _reference_via_cue(normalized)
    amount = _bill_amount(normalized, currency, reference)
    if reference is None and (biller is not None or has_bill_word(text)):
        amount_digits = _amount_digits(amount)
        for run in _DIGITS_RUN_RE.findall(normalized):
            if run != amount_digits:
                reference = run
                break
    return BillEntities(
        biller=biller,
        biller_category=category,
        reference_number=reference,
        amount=amount,
        currency=currency,
    )
