"""Intent detection.

v1 recognises a single business intent (``transfer_money``) using keyword
patterns for English and Arabic. The scoring is structured so a trained
classifier can replace :func:`classify_intent` without changing callers.
"""

from __future__ import annotations

from app.nlu.entities import extract_amount
from app.nlu.normalize import normalize
from app.schemas import Intent, Language

# Trigger keywords per language. Arabic forms include common dialectal variants.
_TRANSFER_KEYWORDS: dict[Language, set[str]] = {
    Language.EN: {"transfer", "send", "wire", "remit", "pay", "move"},
    Language.AR: {
        "حول",
        "حوّل",
        "حولي",
        "تحويل",
        "ارسل",
        "أرسل",
        "ابعت",
        "ادفع",
        "حوللي",
    },
}


# Intents that move money or change the beneficiary list. Retrieval alone must
# not open one of these: a mixed-language aside ("thanks ممكن talk معاك دقيقة
# for a bit") can land near the transfer cluster and used to be answered with
# "how much should I send?".
WRITE_INTENTS: frozenset[Intent] = frozenset(
    {Intent.TRANSFER_MONEY, Intent.PAY_BILL, Intent.ADD_BENEFICIARY}
)

# Words that say the customer wants an action, not a chat. Written in the
# normalized alphabet (``normalize``: alef/ya/ta-marbuta unified, no diacritics)
# and matched as substrings so inflected and prefixed forms hit too.
_MONEY_CUES: frozenset[str] = frozenset(
    {
        # English
        "transfer",
        "send",
        "wire",
        "remit",
        "pay",
        "move",
        "top up",
        "topup",
        "recharge",
        "bill",
        "invoice",
        "beneficiary",
        "payee",
        "iban",
        "sadad",
        "sar",
        "riyal",
        # Arabic
        "حول",
        "حويل",
        "حواله",
        "رسل",
        "ابعت",
        "دفع",
        "سدد",
        "سداد",
        "فاتور",
        "فواتير",
        "مستفيد",
        "فلوس",
        "مبلغ",
        "ريال",
        "ايبان",
    }
)


def has_money_cue(text: str) -> bool:
    """Whether ``text`` itself asks for a money action (word or amount)."""

    normalized = normalize(text)
    if any(cue in normalized for cue in _MONEY_CUES):
        return True
    return extract_amount(text) is not None


def classify_intent(text: str, language: Language) -> tuple[Intent, float]:
    """Return the best ``(intent, confidence)`` for ``text``.

    Confidence grows with the number of distinct trigger keywords matched and is
    capped at ``0.95`` to leave room for a future probabilistic model.
    """

    lowered = text.lower()
    keywords = _TRANSFER_KEYWORDS.get(language, set())
    hits = sum(1 for kw in keywords if kw in lowered)
    if hits == 0:
        return Intent.FALLBACK, 0.0

    confidence = min(0.6 + 0.2 * (hits - 1), 0.95)
    return Intent.TRANSFER_MONEY, confidence
