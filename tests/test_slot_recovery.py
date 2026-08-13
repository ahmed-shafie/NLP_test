"""Slots the extractor used to drop: a payee with no "to", a cue-less bill amount."""

from decimal import Decimal

import pytest

from app.nlu import entities
from app.schemas import Language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pay Mona 40 riyals", "Mona"),
        ("pay mona 40 riyals", "mona"),
        ("send Ahmed 200 sar", "Ahmed"),
        ("transfer khalid fahad 1200 sar", "khalid fahad"),
        ("give sara 50", "sara"),
    ],
)
def test_a_payee_named_before_the_amount_is_read(text: str, expected: str) -> None:
    """No "to" anchor: the name sits between the verb and the amount."""

    assert entities.extract_recipient(text, Language.EN) == expected


@pytest.mark.parametrize(
    "text",
    [
        "pay stc 100",
        "pay mobily 100",
        "pay my electricity bill 200",
        "pay the water bill 4455 amount 120",
        "pay customer service 100",
        "send someone 50",
        "pay my savings 500",
        "transfer money 300",
        "pay usd 300",
    ],
)
def test_a_biller_or_a_thing_never_becomes_a_payee(text: str) -> None:
    assert entities.extract_recipient(text, Language.EN) is None


@pytest.mark.parametrize(
    ("text", "amount", "reference"),
    [
        # A biller named without the word "bill": the number is money.
        ("pay mobiley 100", Decimal("100"), None),
        ("pay stc 100", Decimal("100"), None),
        ("ادفع موبايلي ١٠٠", Decimal("100"), None),
        # The number comes before the bill word, so it cannot be its reference.
        ("pay 320 for my phone bill", Decimal("320"), None),
        # "for 210" is an amount cue, and the reference keeps its own cue.
        ("settle my Mobily bill reference 4455 for 210", Decimal("210"), "4455"),
    ],
)
def test_a_cueless_number_is_the_amount_when_it_cannot_be_a_reference(
    text: str, amount: Decimal, reference: str | None
) -> None:
    lang = Language.AR if any("\u0600" <= c <= "\u06ff" for c in text) else Language.EN
    bill = entities.extract_bill_entities(text, lang)

    assert bill.amount == amount
    assert bill.reference_number == reference


@pytest.mark.parametrize(
    ("text", "reference"),
    [
        ("pay my gas bill 5566", "5566"),
        ("سدد فاتورة الغاز ٥٥٦٦", "5566"),
        # A bare answer to "which bill?" names the bill, it doesn't pay an amount.
        ("Water Services 5512", "5512"),
        ("STC 778899", "778899"),
    ],
)
def test_a_number_that_follows_the_bill_stays_the_reference(
    text: str, reference: str
) -> None:
    lang = Language.AR if any("\u0600" <= c <= "\u06ff" for c in text) else Language.EN
    bill = entities.extract_bill_entities(text, lang)

    assert bill.reference_number == reference
    assert bill.amount is None


def test_a_number_with_no_biller_at_all_fills_nothing() -> None:
    """ "ادفع المخالفة 12345" is not a bill we know: guess neither slot."""

    bill = entities.extract_bill_entities("ادفع المخالفة 12345", Language.AR)

    assert bill.biller is None
    assert bill.amount is None
    assert bill.reference_number is None
