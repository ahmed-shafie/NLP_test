"""Intent detection.

v1 recognises a single business intent (``transfer_money``) using keyword
patterns for English and Arabic. The scoring is structured so a trained
classifier can replace :func:`classify_intent` without changing callers.
"""

from __future__ import annotations

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
