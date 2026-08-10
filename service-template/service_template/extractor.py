"""Language detection, intent detection, and slot extraction.

This is the NLU layer of the template. It has two tiers and **no LLM**:

* a **semantic tier** — a FAISS + multilingual-embeddings intent classifier
  (``semantic_intents.py``) and spaCy PERSON NER for the recipient; and
* a **deterministic tier** — regex/keyword logic that always works.

The semantic tier is preferred when available and confident; otherwise the
deterministic tier takes over. Both are controlled by ``settings`` flags and
degrade gracefully if a model/dependency is missing, so the template always
runs. This mirrors ``app/nlu`` + ``app/orchestration.py`` (which additionally
has an LLM fallback — intentionally omitted here).

The functions stay pure/stateless so they are trivial to unit test.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import TYPE_CHECKING

from service_template.config import DEFAULT_CURRENCY, settings
from service_template.schemas import ActionSlots, Intent, Language

if TYPE_CHECKING:
    from spacy.language import Language as SpacyPipeline

    from service_template.semantic_intents import SemanticIntentClassifier

logger = logging.getLogger(__name__)

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
    """Classify the utterance's intent.

    Two-tier, no LLM: try the FAISS semantic classifier first; if it is
    unavailable or not confident (returns ``FALLBACK``), defer to the
    deterministic keyword classifier. This keeps behaviour sensible whether or
    not the embedding model is installed.
    """

    if settings.use_semantic_intent:
        classifier = _get_semantic()
        if classifier is not None:
            intent, _confidence = classifier.classify(text)
            if intent is not Intent.FALLBACK:
                return intent
    return _keyword_intent(text)


def _get_semantic() -> SemanticIntentClassifier | None:
    # Imported lazily so importing the extractor never forces faiss/embeddings
    # to load (and so tests can disable the semantic tier via settings).
    from service_template.semantic_intents import get_semantic_classifier

    return get_semantic_classifier()


def _keyword_intent(text: str) -> Intent:
    """Deterministic keyword classifier — the always-available fallback."""

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


@lru_cache(maxsize=1)
def _load_spacy() -> SpacyPipeline | None:
    """Load and cache the spaCy model, or ``None`` if unavailable.

    Install the model once with ``python -m spacy download en_core_web_sm``.
    """

    if not settings.use_spacy_ner:
        return None
    try:
        import spacy

        return spacy.load(settings.spacy_model)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to regex
        logger.warning(
            "spaCy model '%s' unavailable (%s); using regex for recipients.",
            settings.spacy_model,
            exc,
        )
        return None


def _extract_recipient(text: str) -> str | None:
    """Prefer a spaCy PERSON entity for the recipient; fall back to the regex.

    spaCy NER catches names the regex misses (e.g. no "to"/"for" cue, or names
    the greedy pattern would mangle). The English model won't tag Arabic names,
    so Arabic falls through to the regex path below.
    """

    nlp = _load_spacy()
    if nlp is not None:
        people = [ent.text.strip() for ent in nlp(text).ents if ent.label_ == "PERSON"]
        if people:
            return people[0]
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
