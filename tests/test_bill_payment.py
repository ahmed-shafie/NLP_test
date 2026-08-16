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
    # "electricity" resolves to the SADAD utility (biller_code 002).
    assert extract_biller("pay my electricity bill", Language.EN) == (
        "Saudi Electric Company",
        "Utilities",
        "002",
    )


def test_extract_biller_category_ar():
    assert extract_biller("ادفع فاتورة الكهرباء", Language.AR) == (
        "الشركة السعودية للكهرباء",
        "Utilities",
        "002",
    )


def test_extract_biller_freetext_fallback():
    # An unknown biller with no gazetteer/semantic hit is kept as free text.
    biller, category, code = extract_biller("pay my Acme Telecom bill", Language.EN)
    assert biller == "Acme Telecom"
    assert category is None
    assert code is None


@pytest.mark.parametrize(
    "text",
    [
        "I want to pay a bill",
        "I need to pay a bill",
        "I want to pay bill",
        "I would like to pay a bill",
        "can you help me pay a bill",
    ],
)
def test_extract_biller_ignores_request_preamble(text: str):
    # The verb phrase of a bare "pay a bill" request is not a biller name, so the
    # engine still asks *which* bill instead of jumping to the reference number.
    assert extract_biller(text, Language.EN) == (None, None, None)


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
    assert ent.biller == "Saudi Electric Company"
    assert ent.biller_code == "002"


def test_bill_entities_amount_cue():
    ent = extract_bill_entities("pay utility bill 5512 amount 150", Language.EN)
    assert ent.amount == Decimal("150")
    assert ent.reference_number == "5512"


def test_bill_entities_bare_reference_no_amount():
    # No currency / amount cue -> the digit run is the reference, amount stays None.
    ent = extract_bill_entities("electricity bill 778899", Language.EN)
    assert ent.reference_number == "778899"
    assert ent.amount is None


def test_a_priced_number_after_the_bill_word_is_the_amount():
    """ "bill 100 sar": the cue says reference, the customer said riyals."""

    ent = extract_bill_entities("pay my mobily bill 100 sar", Language.EN)
    assert ent.amount == Decimal("100")
    assert ent.currency == "SAR"
    assert ent.reference_number is None

    arabic = extract_bill_entities("ادفع فاتورة موبايلي ١٠٠ ريال", Language.AR)
    assert arabic.amount == Decimal("100")
    assert arabic.reference_number is None


def test_an_unpriced_number_after_the_bill_word_stays_the_reference():
    """Reading it as money would put a figure the customer never quoted on the
    confirmation; read as a reference, the worst case is being asked the amount."""

    ent = extract_bill_entities("pay my mobily bill 100", Language.EN)
    assert ent.reference_number == "100"
    assert ent.amount is None


def test_a_reference_and_a_priced_amount_keep_their_own_slots():
    ent = extract_bill_entities("pay stc bill 4455 210 sar", Language.EN)
    assert ent.reference_number == "4455"
    assert ent.amount == Decimal("210")


def test_bill_entities_arabic_digits():
    ent = extract_bill_entities("ادفع فاتورة الكهرباء 778899 بمبلغ 320", Language.AR)
    assert ent.biller == "الشركة السعودية للكهرباء"
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
    # A named (unambiguous) biller goes straight to confirmation.
    result = engine.handle("pay 320 EGP STC bill 778899", "ob1")
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller == "STC"
    assert result.state.slots.biller_code == "001"
    assert result.state.slots.reference_number == "778899"
    assert result.state.slots.amount == Decimal("320")
    assert result.state.slots.currency == "EGP"


def test_one_shot_bill_confirm_emits_payload(engine: ConversationEngine):
    engine.handle("pay 320 EGP STC bill 778899", "ob2")
    result = engine.handle("yes", "ob2")
    assert result.state.status is ConversationStatus.COMPLETED
    assert result.bill is not None
    assert result.bill.biller == "STC"
    assert result.bill.biller_code == "001"
    assert result.bill.biller_name == "STC"
    assert result.bill.reference_number == "778899"
    assert result.bill.amount == Decimal("320")
    assert result.bill.currency == "EGP"


# --------------------------- multi-turn bill flow -------------------------- #


def test_multi_turn_bill_flow(engine: ConversationEngine):
    r1 = engine.handle("I want to pay my internet bill", "mt1")
    assert r1.state.intent is Intent.PAY_BILL
    # "internet" maps to the whole Telecom & Internet category, so the bot lists
    # the SADAD billers (STC, Mobily, ...) and asks the customer to choose.
    assert r1.state.status is ConversationStatus.DISAMBIGUATING
    assert r1.state.pending_slot == "biller"
    assert "(1)" in r1.reply

    r1b = engine.handle("1", "mt1")  # pick the first listed biller (STC)
    assert r1b.state.slots.biller_code == "001"
    assert r1b.state.pending_slot == "reference_number"

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


# --------------------------- unknown biller ------------------------------- #


def test_unknown_biller_is_rejected_one_shot(engine: ConversationEngine):
    # Only SADAD-catalogue billers can be paid: free text is not accepted.
    result = engine.handle("pay my Acme Telecom bill", "ub1")
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.slots.biller is None
    assert result.state.pending_slot == "biller"
    assert "isn't in our list of billers" in result.reply
    assert "Acme Telecom" in result.reply


def test_unknown_biller_is_rejected_when_asked(engine: ConversationEngine):
    engine.handle("I want to pay a bill", "ub2")
    result = engine.handle("Acme Telecom", "ub2")
    assert result.state.slots.biller is None
    assert result.state.pending_slot == "biller"
    assert "Acme Telecom" in result.reply


def test_unknown_biller_then_valid_one_resumes(engine: ConversationEngine):
    engine.handle("I want to pay a bill", "ub3")
    engine.handle("Acme Telecom", "ub3")
    result = engine.handle("STC", "ub3")
    assert result.state.slots.biller == "STC"
    assert result.state.slots.biller_code == "001"
    assert result.state.pending_slot == "reference_number"


def test_arabic_letter_spelled_biller_starts_bill_flow(engine: ConversationEngine):
    # "اس تي سي" is STC; the semantic classifier must not divert this to the
    # read-only beneficiary listing.
    result = engine.handle("ادفع ل اس تي سي", "ls1")
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.slots.biller_code == "001"
    assert result.state.pending_slot == "reference_number"


def test_unknown_biller_rejected_in_arabic(engine: ConversationEngine):
    engine.handle("ابغى ادفع فاتورة", "ub4")
    result = engine.handle("شركة نجم", "ub4")
    assert result.state.slots.biller is None
    assert "غير موجود في قائمة المزوّدين" in result.reply
    assert "شركة نجم" in result.reply


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

    r = engine.handle("Water Services 5512", "ch2")
    assert r.state.slots.biller == "Water Services"
    assert r.state.slots.biller_code == "015"
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
    # A named (unambiguous) Arabic biller goes straight to confirmation.
    result = engine.handle(
        "ادفع فاتورة الشركة السعودية للكهرباء 778899 بمبلغ 320", "ar1"
    )
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller == "الشركة السعودية للكهرباء"
    assert result.state.slots.amount == Decimal("320")
    confirmed = engine.handle("نعم", "ar1")
    assert confirmed.state.status is ConversationStatus.COMPLETED
    assert confirmed.bill is not None
    assert confirmed.bill.reference_number == "778899"


def test_arabic_bill_confirm_prompt_is_arabic(engine: ConversationEngine):
    result = engine.handle("ادفع فاتورة الغاز 1122 بمبلغ 50", "ar2")
    assert "تأكيد" in result.reply


# ------------------------- biller disambiguation (C2) --------------------- #


def test_ambiguous_biller_asks_which_one(engine: ConversationEngine):
    # "electricity" maps to two SADAD billers -> the engine asks instead of guessing.
    result = engine.handle("pay my electricity bill 778899 amount 200", "dz1")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert result.state.pending_slot == "biller"
    assert len(result.state.biller_options) == 2
    codes = {opt.code for opt in result.state.biller_options}
    assert codes == {"002", "004"}
    # Other slots stated in the same turn are retained for after the choice.
    assert result.state.slots.reference_number == "778899"
    assert result.state.slots.amount == Decimal("200")
    assert result.state.slots.currency == "SAR"


def test_disambiguation_pick_by_number(engine: ConversationEngine):
    engine.handle("pay my electricity bill 778899 amount 200", "dz2")
    result = engine.handle("2", "dz2")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller_code == "004"
    assert result.state.slots.biller == "Marafiq"


def test_disambiguation_pick_by_name(engine: ConversationEngine):
    engine.handle("pay my electricity bill 778899 amount 200", "dz3")
    result = engine.handle("Saudi Electric Company", "dz3")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller_code == "002"


def test_disambiguation_unrecognised_reask(engine: ConversationEngine):
    engine.handle("pay my electricity bill 778899 amount 200", "dz4")
    result = engine.handle("not sure", "dz4")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert "(1)" in result.reply and "(2)" in result.reply


def test_disambiguation_arabic_pick_by_indic_digit(engine: ConversationEngine):
    engine.handle("ادفع فاتورة الكهرباء 778899 بمبلغ 200", "dz5")
    result = engine.handle("١", "dz5")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller_code == "002"


def test_named_biller_skips_disambiguation(engine: ConversationEngine):
    # Naming the specific biller resolves directly, no question asked.
    result = engine.handle("pay Marafiq bill 5566 amount 310", "dz6")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller_code == "004"


# ------------------- category / code / typo bill resolution --------------- #


def test_internet_category_lists_telecom_billers(engine: ConversationEngine):
    result = engine.handle("pay my internet bill 778899 amount 200", "cat1")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    codes = {opt.code for opt in result.state.biller_options}
    assert "001" in codes and "005" in codes  # STC + Mobily offered
    # Slots from the same turn survive the pause.
    assert result.state.slots.reference_number == "778899"
    assert result.state.slots.currency == "SAR"


def test_disambiguation_pick_by_sadad_code(engine: ConversationEngine):
    engine.handle("pay my internet bill 778899 amount 200", "cat2")
    result = engine.handle("005", "cat2")  # pick by SADAD code, not list index
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller_code == "005"
    assert result.state.slots.biller == "Mobily"


def test_numeric_code_as_biller_answer(engine: ConversationEngine):
    # Choosing "pay a bill", then sending a SADAD code resolves to its name.
    engine.handle("hi", "code1")
    r = engine.handle("2", "code1")
    assert r.state.pending_slot == "biller"
    r = engine.handle("153", "code1")
    assert r.state.slots.biller_code == "153"
    assert r.state.slots.biller == "Ejar"


def test_bill_amount_not_misread_as_biller_code(engine: ConversationEngine):
    # 200 is a real SADAD code (Mawhiba), but here it's the amount -> it must
    # stay the amount; the biller is resolved by name, never from the amount.
    r = engine.handle("pay STC bill amount 200 ref 778899", "code2")
    assert r.state.slots.amount == Decimal("200")
    assert r.state.slots.biller_code == "001"  # STC, by name
    assert r.state.slots.biller == "STC"


def test_typo_biller_name_resolves(engine: ConversationEngine):
    # "egar" is a single-letter typo of the SADAD biller "Ejar".
    result = engine.handle("pay egar bill 4455 amount 300", "typo1")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.biller_code == "153"
    assert result.state.slots.biller == "Ejar"
