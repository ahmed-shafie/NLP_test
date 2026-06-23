"""Tests for the pay-bill flow, the Transfer/Pay-bill chooser, and bill entities."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.nlu import pipeline
from app.nlu.entities import (
    extract_bill_entities,
    extract_biller,
    extract_reference_number,
)
from app.schemas import Intent, Language


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


# ------------------------------ entity units ------------------------------ #


def test_extract_biller_category_en():
    assert extract_biller("pay my electricity bill", Language.EN) == (
        "electricity",
        "electricity",
    )


def test_extract_biller_category_ar():
    assert extract_biller("ادفع فاتورة الكهرباء", Language.AR) == (
        "electricity",
        "electricity",
    )


def test_extract_biller_freetext_fallback():
    biller, category = extract_biller("pay my Acme Telecom bill", Language.EN)
    assert biller == "Acme Telecom"
    assert category is None


def test_extract_reference_number_with_cue():
    assert extract_reference_number("ref 778899") == "778899"


def test_extract_reference_number_keeps_leading_zeros():
    # The reference is a string, so leading zeros survive.
    assert extract_reference_number("bill 007788") == "007788"


def test_extract_reference_number_bare_digits():
    assert extract_reference_number("4455123") == "4455123"


def test_bill_entities_disambiguates_amount_vs_reference():
    # Currency cues 320 as the amount; the post-"bill" run is the reference.
    ent = extract_bill_entities("pay 320 EGP electricity bill 778899", Language.EN)
    assert ent.amount == Decimal("320")
    assert ent.reference_number == "778899"
    assert ent.currency == "EGP"
    assert ent.biller == "electricity"


def test_bill_entities_amount_cue():
    ent = extract_bill_entities("pay utility bill 5512 amount 150", Language.EN)
    assert ent.amount == Decimal("150")
    assert ent.reference_number == "5512"


def test_bill_entities_bare_reference_no_amount():
    # No currency / amount cue -> the digit run is the reference, amount stays None.
    ent = extract_bill_entities("electricity bill 778899", Language.EN)
    assert ent.reference_number == "778899"
    assert ent.amount is None


def test_bill_entities_arabic_digits():
    ent = extract_bill_entities("ادفع فاتورة الكهرباء 778899 بمبلغ 320", Language.AR)
    assert ent.biller == "electricity"
    assert ent.amount == Decimal("320")
    assert ent.reference_number == "778899"


# ------------------------------ validation -------------------------------- #


def test_validate_bill_payment_ok():
    payment, missing, errors = pipeline.validate_bill_payment(
        biller="electricity",
        reference_number="778899",
        amount=Decimal("320"),
        currency="EGP",
    )
    assert not missing and not errors
    assert payment is not None
    assert payment.amount == Decimal("320")
    assert payment.currency == "EGP"


def test_validate_bill_payment_missing_slots():
    payment, missing, _errors = pipeline.validate_bill_payment(
        biller=None, reference_number=None, amount=None, currency=None
    )
    assert payment is None
    assert set(missing) == {"biller", "reference_number", "amount", "currency"}


def test_validate_bill_payment_bad_currency():
    payment, _missing, errors = pipeline.validate_bill_payment(
        biller="electricity",
        reference_number="778899",
        amount=Decimal("320"),
        currency="ZZZ",
    )
    assert payment is None
    assert any(e.field == "currency" for e in errors)


def test_validate_bill_payment_non_positive_amount():
    payment, _missing, errors = pipeline.validate_bill_payment(
        biller="electricity",
        reference_number="778899",
        amount=Decimal("0"),
        currency="EGP",
    )
    assert payment is None
    assert any(e.field == "amount" for e in errors)


# --------------------------- one-shot bill flow --------------------------- #


def test_one_shot_bill_reaches_confirmation(engine: ConversationEngine):
    result = engine.handle("pay 320 EGP electricity bill 778899", "ob1")
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller == "electricity"
    assert result.state.slots.reference_number == "778899"
    assert result.state.slots.amount == Decimal("320")
    assert result.state.slots.currency == "EGP"


def test_one_shot_bill_confirm_emits_payload(engine: ConversationEngine):
    engine.handle("pay 320 EGP electricity bill 778899", "ob2")
    result = engine.handle("yes", "ob2")
    assert result.state.status is ConversationStatus.COMPLETED
    assert result.bill is not None
    assert result.bill.biller == "electricity"
    assert result.bill.reference_number == "778899"
    assert result.bill.amount == Decimal("320")
    assert result.bill.currency == "EGP"


# --------------------------- multi-turn bill flow -------------------------- #


def test_multi_turn_bill_flow(engine: ConversationEngine):
    r1 = engine.handle("I want to pay my internet bill", "mt1")
    assert r1.state.intent is Intent.PAY_BILL
    assert r1.state.pending_slot == "reference_number"

    r2 = engine.handle("4455123", "mt1")
    assert r2.state.slots.reference_number == "4455123"
    assert r2.state.pending_slot == "amount"

    r3 = engine.handle("250 EGP", "mt1")
    assert r3.state.status is ConversationStatus.CONFIRMING
    assert r3.state.slots.amount == Decimal("250")

    r4 = engine.handle("yes", "mt1")
    assert r4.state.status is ConversationStatus.COMPLETED
    assert r4.bill is not None
    assert r4.bill.reference_number == "4455123"


# ------------------------------- chooser ---------------------------------- #


def test_vague_message_shows_chooser(engine: ConversationEngine):
    result = engine.handle("hi I want to do something", "ch1")
    assert result.state.status is ConversationStatus.SELECTING
    assert "(1)" in result.reply and "(2)" in result.reply


def test_chooser_choose_bill_then_flow(engine: ConversationEngine):
    engine.handle("hi", "ch2")
    r = engine.handle("2", "ch2")
    assert r.state.intent is Intent.PAY_BILL
    assert r.state.pending_slot == "biller"

    r = engine.handle("water 5512", "ch2")
    assert r.state.slots.biller == "water"
    assert r.state.slots.reference_number == "5512"

    r = engine.handle("80 EGP", "ch2")
    assert r.state.status is ConversationStatus.CONFIRMING

    r = engine.handle("yes", "ch2")
    assert r.state.status is ConversationStatus.COMPLETED
    assert r.bill is not None


def test_chooser_choose_transfer(engine: ConversationEngine):
    engine.handle("hello", "ch3")
    r = engine.handle("1", "ch3")
    assert r.state.intent is Intent.TRANSFER_MONEY


def test_unrecognised_choice_reasks(engine: ConversationEngine):
    engine.handle("hmm", "ch4")
    r = engine.handle("maybe later", "ch4")
    assert r.state.status is ConversationStatus.SELECTING


# ----------------------------- disambiguation ----------------------------- #


def test_clear_transfer_skips_chooser(engine: ConversationEngine):
    result = engine.handle("send 500 dollars to Ahmed Nassar", "ds1")
    assert result.state.intent is Intent.TRANSFER_MONEY
    assert result.state.status is ConversationStatus.CONFIRMING


def test_pay_a_person_is_transfer_not_bill(engine: ConversationEngine):
    # "pay <amount> to <name>" with no biller/bill keyword -> transfer flow.
    result = engine.handle("pay 75 to Ahmed", "ds2")
    assert result.state.intent is Intent.TRANSFER_MONEY


def test_clear_bill_skips_chooser(engine: ConversationEngine):
    result = engine.handle("pay my gas bill", "ds3")
    assert result.state.intent is Intent.PAY_BILL


# -------------------------------- Arabic ---------------------------------- #


def test_arabic_one_shot_bill(engine: ConversationEngine):
    result = engine.handle("ادفع فاتورة الكهرباء 778899 بمبلغ 320", "ar1")
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller == "electricity"
    assert result.state.slots.amount == Decimal("320")
    confirmed = engine.handle("نعم", "ar1")
    assert confirmed.state.status is ConversationStatus.COMPLETED
    assert confirmed.bill is not None
    assert confirmed.bill.reference_number == "778899"


def test_arabic_bill_confirm_prompt_is_arabic(engine: ConversationEngine):
    result = engine.handle("ادفع فاتورة الغاز 1122 بمبلغ 50", "ar2")
    assert "تأكيد" in result.reply
