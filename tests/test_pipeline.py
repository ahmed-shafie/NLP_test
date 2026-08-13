"""Integration tests for the NLU pipeline."""

from decimal import Decimal

from app.nlu.pipeline import parse, validate_transfer
from app.schemas import Intent, Language


def test_parse_english_transfer():
    result = parse("transfer 500 dollars to John")
    assert result.language == Language.EN
    assert result.intent == Intent.TRANSFER_MONEY
    assert result.entities.amount == Decimal("500")
    assert result.entities.currency == "USD"


def test_parse_arabic_transfer():
    result = parse("حوّل ألف جنيه إلى أحمد")
    assert result.language == Language.AR
    assert result.intent == Intent.TRANSFER_MONEY
    assert result.entities.currency == "EGP"


def test_parse_fallback():
    result = parse("what is the weather today")
    assert result.intent == Intent.FALLBACK


def test_validate_complete():
    result = validate_transfer(
        amount=Decimal("500"),
        currency="USD",
        recipient="John",
    )
    assert result.valid is True
    assert result.transfer is not None
    assert result.transfer.amount == Decimal("500")


def test_validate_missing_amount():
    result = validate_transfer(amount=None, currency="USD", recipient="John")
    assert result.valid is False
    assert "amount" in result.missing


def test_validate_bad_currency():
    result = validate_transfer(amount=Decimal("50"), currency="XYZ", recipient="John")
    assert result.valid is False
    assert any(e.field == "currency" for e in result.errors)
