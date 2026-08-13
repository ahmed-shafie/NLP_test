"""Tests for intent classification."""

from app.nlu.intents import classify_intent
from app.schemas import Intent, Language


def test_en_transfer():
    intent, conf = classify_intent("transfer 100 dollars to John", Language.EN)
    assert intent == Intent.TRANSFER_MONEY
    assert conf >= 0.6


def test_ar_transfer():
    intent, conf = classify_intent("حوّل ألف جنيه إلى أحمد", Language.AR)
    assert intent == Intent.TRANSFER_MONEY
    assert conf >= 0.6


def test_en_fallback():
    intent, conf = classify_intent("what is my balance", Language.EN)
    assert intent == Intent.FALLBACK
    assert conf == 0.0


def test_ar_fallback():
    intent, conf = classify_intent("ما هو رصيدي", Language.AR)
    assert intent == Intent.FALLBACK
    assert conf == 0.0
