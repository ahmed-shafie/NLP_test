"""Tests for language detection."""

from app.nlu.lang import detect_language
from app.schemas import Language


def test_english_text():
    assert detect_language("Transfer 500 dollars to John") == Language.EN


def test_arabic_text():
    assert detect_language("حوّل مبلغ ٥٠٠ جنيه إلى أحمد") == Language.AR


def test_mixed_mostly_arabic():
    assert detect_language("حوّل 500 إلى Ahmed") == Language.AR


def test_mixed_mostly_english():
    assert detect_language("send money to أحمد right now") == Language.EN
