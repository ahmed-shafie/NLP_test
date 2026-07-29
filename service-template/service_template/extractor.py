"""Language detection, intent detection, and slot extraction.

This is the deliberately *simple* NLU layer of the template. It uses regexes and
keyword sets so the template runs with no ML dependencies. The production app
replaces this with a semantic classifier + FAISS retrieval + an LLM fallback
(see ``app/nlu`` and ``app/orchestration.py``) — but the *interface* is the
same: text in, structured signals out. You can upgrade this file in isolation
without touching the engine.

Everything here is pure functions (no state), which makes them trivial to unit
test.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from service_template.config import DEFAULT_CURRENCY
from service_template.schemas import ActionSlots, Intent, Language

# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
# Arabic Unicode block. If the text contains any Arabic letter we treat the
# whole utterance as Arabic (good enough for a demo; the main app uses a proper
# language detector).
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> Language:
    return Language.AR if _ARABIC_RE.search(text) else Language.EN


# Arabic diacritics (tashkeel) + tatweel. We strip them so "حوّل" matches the
# cue "حول": real NLU always normalises before matching.
_TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")


def _normalize(text: str) -> str:
    return _TASHKEEL_RE.sub("", text)


# --------------------------------------------------------------------------- #
# Intent detection (keyword based)
# --------------------------------------------------------------------------- #
# >>> EDIT PER CASE: add the trigger words for your new case and return its
#     Intent below. Order matters — check the most specific case first.
_TRANSFER_CUES = {
    # English
    "send",
    "transfer",
    "pay",
    "wire",
    "remit",
    # Arabic
    "حول",
    "حوّل",
    "ارسل",
    "أرسل",
    "تحويل",
}
_GREETING_CUES = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank",
    "مرحبا",
    "شكرا",
    "السلام",
}

# Amounts written as digits, optionally with a currency code/symbol.
_AMOUNT_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")
# Arabic-Indic digits → ASCII, so "٢٠٠" is parsed as 200.
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_CURRENCY_WORDS = {
    "sar": "SAR",
    "riyal": "SAR",
    "ريال": "SAR",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "دولار": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "يورو": "EUR",
    "gbp": "GBP",
    "pound": "GBP",
    "aed": "AED",
    "egp": "EGP",
}

# Recipient after "to"/"for" (EN) or "إلى"/"الى"/"ل" (AR). Captures a short name.
_RECIPIENT_RE = re.compile(
    r"(?:to|for|إلى|الى|لـ|ل)\s+([A-Za-z\u0600-\u06FF][\w\u0600-\u06FF]*"
    r"(?:\s+[A-Za-z\u0600-\u06FF][\w\u0600-\u06FF]*)?)",
    re.IGNORECASE,
)
# Source account: "from my savings", "from current account", "from account ...1234".
_SOURCE_RE = re.compile(
    r"from(?:\s+my)?\s+(savings|current|credit|salary|account(?:\s+ending)?\s*\d+)",
    re.IGNORECASE,
)

# Words that must never be treated as a recipient name — includes the action
# verbs so "I want to send money" does not capture "send" as the recipient.
_STOPWORDS = {
    "my",
    "the",
    "a",
    "an",
    "account",
    "please",
    "savings",
    "current",
    "send",
    "transfer",
    "pay",
    "wire",
    "remit",
    "money",
    "some",
    "cash",
}


def detect_intent(text: str) -> Intent:
    """Very small keyword classifier. Replace with a real one in production."""

    tokens = _tokens(_normalize(text))
    if tokens & _TRANSFER_CUES:
        return Intent.TRANSFER_MONEY
    # >>> EDIT PER CASE: add ``if tokens & _YOUR_CUES: return Intent.YOUR_CASE``
    if tokens & _GREETING_CUES:
        return Intent.SMALL_TALK
    return Intent.FALLBACK


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE))


# --------------------------------------------------------------------------- #
# Slot extraction
# --------------------------------------------------------------------------- #
def extract_slots(text: str) -> ActionSlots:
    """Pull whatever slots we can from a single utterance.

    Returns a *partial* ``ActionSlots``; the engine merges it into the running
    state without overwriting slots already filled on earlier turns.
    """

    normalized = _normalize(text).translate(_ARABIC_DIGITS)
    return ActionSlots(
        amount=_extract_amount(normalized),
        currency=_extract_currency(normalized),
        recipient=_extract_recipient(normalized),
        source_account=_extract_source(normalized),
    )


def _extract_amount(text: str) -> Decimal | None:
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def _extract_currency(text: str) -> str | None:
    for word in _tokens(text):
        if word in _CURRENCY_WORDS:
            return _CURRENCY_WORDS[word]
    return None


def _extract_recipient(text: str) -> str | None:
    match = _RECIPIENT_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    # Drop trailing stopwords that the greedy regex may have captured.
    parts = [p for p in name.split() if p.lower() not in _STOPWORDS]
    return " ".join(parts) or None


def _extract_source(text: str) -> str | None:
    match = _SOURCE_RE.search(text)
    return match.group(1).strip() if match else None


def apply_defaults(slots: ActionSlots) -> ActionSlots:
    """Fill in sensible defaults (e.g. currency) once the rest is known.

    Called by the engine after merging. Keeping defaults here (not in
    extraction) means "send 500 to Ahmed" only defaults the currency to SAR once
    we're sure the user didn't state one.
    """

    if slots.amount is not None and not slots.currency:
        slots.currency = DEFAULT_CURRENCY
    return slots
