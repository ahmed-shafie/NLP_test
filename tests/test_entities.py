"""Tests for entity extraction helpers."""

from decimal import Decimal

import pytest

from app.nlu.entities import (
    extract_amount,
    extract_currency,
    extract_recipient,
    extract_source_account,
    normalize_digits,
)
from app.schemas import Language


def test_normalize_arabic_indic_digits():
    assert normalize_digits("٥٠٠") == "500"


def test_extract_amount_simple():
    assert extract_amount("send 200 dollars") == Decimal("200")


def test_extract_amount_with_comma():
    assert extract_amount("transfer 1,500 egp") == Decimal("1500")


def test_extract_amount_with_multiplier():
    assert extract_amount("send 5k to John") == Decimal("5000")


def test_extract_amount_arabic():
    assert extract_amount("حوّل ٥٠٠ جنيه") == Decimal("500")


def test_extract_currency_symbol():
    assert extract_currency("$500") == "USD"
    assert extract_currency("500€") == "EUR"


def test_extract_currency_word_en():
    assert extract_currency("transfer 50 dollars") == "USD"


def test_extract_currency_word_ar():
    assert extract_currency("حوّل ٥٠٠ جنيه") == "EGP"


def test_extract_recipient_en():
    result = extract_recipient("send money to John", Language.EN)
    assert result == "John"


def test_extract_recipient_ar():
    result = extract_recipient("حوّل ألف جنيه إلى أحمد", Language.AR)
    assert result == "أحمد"


@pytest.mark.parametrize(
    "text",
    [
        "حولي الي سارة 500",  # colloquial "الي", amount last
        "حولي الى سارة 500",
        "حوّل إلى سارة ٥٠٠",  # Arabic-Indic digits
        "حول 500 الى سارة",  # amount first (already worked)
        "حول 100 لسارة",  # attached lam
    ],
)
def test_extract_recipient_ar_before_amount(text):
    """The name is readable whether the amount leads or trails it."""

    assert extract_recipient(text, Language.AR) == "ساره"


def test_extract_recipient_ar_keeps_full_name():
    assert extract_recipient("حولي الي خالد فهد 500", Language.AR) == "خالد فهد"


def test_extract_source_account_en():
    result = extract_source_account("transfer from my savings account", Language.EN)
    assert result == "savings"


def test_no_amount():
    assert extract_amount("send money to Ahmed") is None
