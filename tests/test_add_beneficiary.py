"""Standalone add-beneficiary flow: IBAN validation and confirm-before-write.

Adding a beneficiary is a *write*, so these tests pin the two safety rules:
the account must be a genuinely valid IBAN/account, and the Banking Core call
only happens after an explicit "yes".
"""

from __future__ import annotations

import pytest

import app.banking_core_client as bcc
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.nlu.accounts import analyze_iban_typo, validate_account
from app.schemas import Intent

VALID_IBAN = "SA0380000000608010167519"
VALID_IBAN_SPACED = "SA03 8000 0000 6080 1016 7519"
# Same IBAN with the last two characters transposed, and with one digit changed.
SWAPPED_IBAN = "SA0380000000608010167591"
MISTYPED_IBAN = "SA0380000000608010167516"


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


@pytest.fixture()
def writes(monkeypatch) -> list[dict[str, str]]:
    """Capture every Banking Core add_beneficiary call."""

    calls: list[dict[str, str]] = []

    def _add(**kwargs: str) -> dict[str, object]:
        calls.append(kwargs)
        return {"ok": True, "beneficiary": {"id": "new-1"}}

    monkeypatch.setattr(bcc, "add_beneficiary", _add)
    return calls


# ------------------------------ IBAN validation ----------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        (VALID_IBAN, VALID_IBAN),
        (VALID_IBAN_SPACED, VALID_IBAN),
        (VALID_IBAN.lower(), VALID_IBAN),
        ("SA03-8000-0000-6080-1016-7519", VALID_IBAN),
        ("0000123456", "0000123456"),  # domestic account number
    ],
)
def test_valid_accounts(raw: str, expected: str):
    assert validate_account(raw) == (expected, None)


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("1234", "too_short"),
        ("abcd", "not_an_account"),
        ("", "not_an_account"),
        ("SA1122330000007777", "iban_length"),  # 18 chars, SA needs 24
        ("SA0380000000608010167518", "iban_checksum"),  # last digit tampered
        ("SA555444333", "not_an_account"),
    ],
)
def test_invalid_accounts(raw: str, reason: str):
    account, got = validate_account(raw)
    assert account is None
    assert got == reason


# --------------------------- standalone add flow ---------------------------- #


def test_standalone_add_collects_name_then_account(engine, writes):
    """ "add a beneficiary" is its own intent, not a listing dead end."""

    first = engine.handle("add a new beneficiary", "add-1")
    assert first.state.intent is Intent.ADD_BENEFICIARY
    assert first.state.status is ConversationStatus.COLLECTING
    assert first.state.pending_slot == "beneficiary_name"

    named = engine.handle("Sara Ali", "add-1")
    assert named.state.pending_add_name == "Sara Ali"
    assert named.state.pending_slot == "beneficiary_account"

    quoted = engine.handle(VALID_IBAN, "add-1")
    assert quoted.state.status is ConversationStatus.CONFIRMING
    assert quoted.state.pending_add_account == VALID_IBAN
    assert not writes  # nothing written until confirmed

    done = engine.handle("yes", "add-1")
    assert done.state.status is ConversationStatus.COMPLETED
    assert len(writes) == 1
    assert writes[0]["name"] == "Sara Ali"
    assert writes[0]["account"] == VALID_IBAN


def test_standalone_add_arabic(engine, writes):
    first = engine.handle("اضف مستفيد جديد", "add-ar")
    assert first.state.intent is Intent.ADD_BENEFICIARY
    assert "اسم" in first.reply

    named = engine.handle("سارة علي", "add-ar")
    assert named.state.pending_add_name == "سارة علي"

    # A bare IBAN carries no language signal, so the reply stays Arabic.
    quoted = engine.handle(VALID_IBAN, "add-ar")
    assert quoted.state.status is ConversationStatus.CONFIRMING
    assert "نعم/لا" in quoted.reply

    done = engine.handle("نعم", "add-ar")
    assert len(writes) == 1
    assert done.state.status is ConversationStatus.COMPLETED


@pytest.mark.parametrize("phrase", ["ادفع المخالفة", "ادفع فاتورة", "pay a bill"])
def test_payment_requests_never_enter_the_add_flow(engine, writes, phrase):
    """A payment phrasing must not be pulled into add/list by the classifier."""

    result = engine.handle(phrase, f"no-add-{abs(hash(phrase))}")
    assert result.state.pending_add_name is None
    assert result.state.intent is not Intent.ADD_BENEFICIARY
    assert "آيبان" not in result.reply
    assert "IBAN" not in result.reply
    assert not writes


def test_arabic_stays_arabic_through_invalid_iban(engine, writes):
    """An account-shaped reply carries no language signal, valid or not."""

    engine.handle("اضف مستفيد جديد", "add-ar2")
    engine.handle("نورة سعد", "add-ar2")
    rejected = engine.handle("SA9820000001234567891234", "add-ar2")  # bad checksum
    assert "الآيبان" in rejected.reply
    assert rejected.state.language.value == "ar"


def test_one_shot_splits_name_and_account(engine, writes):
    """Name and IBAN in one message land in separate slots, not one blob."""

    result = engine.handle(f"add beneficiary Sara Ali {VALID_IBAN}", "add-2")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.pending_add_name == "Sara Ali"
    assert result.state.pending_add_account == VALID_IBAN


def test_one_shot_arabic_name_only(engine, writes):
    result = engine.handle("ضيف سارة علي كمستفيد", "add-3")
    assert result.state.pending_add_name == "سارة علي"
    assert result.state.pending_slot == "beneficiary_account"


@pytest.mark.parametrize(
    "bad, hint",
    [
        ("1234", "too short"),
        ("SA1122330000007777", "24 characters"),
        ("SA0380000000608010167518", "checksum"),
    ],
)
def test_invalid_account_keeps_flow_open(engine, writes, bad: str, hint: str):
    session = f"add-4-{bad}"  # the session store is shared across parametrized runs
    engine.handle("add a beneficiary", session)
    engine.handle("Khaled Otaibi", session)
    rejected = engine.handle(bad, session)
    assert hint in rejected.reply
    assert rejected.state.pending_slot == "beneficiary_account"
    assert rejected.state.pending_add_account is None
    assert not writes

    accepted = engine.handle(VALID_IBAN, session)
    assert accepted.state.status is ConversationStatus.CONFIRMING


def test_declining_confirmation_writes_nothing(engine, writes):
    engine.handle(f"add beneficiary Sara Ali {VALID_IBAN}", "add-5")
    result = engine.handle("no", "add-5")
    assert result.state.status is ConversationStatus.CANCELLED
    assert result.state.pending_add_account is None
    assert not writes


def test_unrecognised_confirmation_reasks(engine, writes):
    engine.handle(f"add beneficiary Sara Ali {VALID_IBAN}", "add-6")
    result = engine.handle("maybe later", "add-6")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert "SA••7519" in result.reply
    assert not writes


def test_banking_core_failure_is_surfaced(engine, monkeypatch):
    monkeypatch.setattr(
        bcc,
        "add_beneficiary",
        lambda **k: {
            "ok": False,
            "message": "A beneficiary with that account already exists.",
        },
    )
    engine.handle(f"add beneficiary Sara Ali {VALID_IBAN}", "add-7")
    result = engine.handle("yes", "add-7")
    assert "already exists" in result.reply
    assert "✅" not in result.reply  # never claim success on a rejected write


# --------------------- a failed checksum is a typo, not a wall -------------- #


def test_typo_hint_finds_a_transposition():
    hint = analyze_iban_typo(SWAPPED_IBAN)
    assert hint.swapped == (23, 24)
    assert hint.is_located


def test_typo_hint_stays_silent_when_the_spot_is_ambiguous():
    """Several single-character edits repair it, so no position is claimed."""

    hint = analyze_iban_typo(MISTYPED_IBAN)
    assert hint.swapped is None
    assert len(hint.positions) > 1
    assert not hint.is_located


def test_typo_hint_is_empty_for_a_valid_iban():
    assert analyze_iban_typo(VALID_IBAN) == analyze_iban_typo("not an iban")


def test_checksum_failure_offers_a_way_through(engine, writes):
    engine.handle("add a beneficiary", "typo-1")
    engine.handle("Khaled Otaibi", "typo-1")
    rejected = engine.handle(MISTYPED_IBAN, "typo-1")

    assert "checksum" in rejected.reply
    assert "I'm sure" in rejected.reply
    assert rejected.state.pending_add_account is None  # not accepted yet
    assert rejected.state.pending_unchecked_account == MISTYPED_IBAN
    assert not writes


def test_the_customer_can_insist_and_the_account_is_used_verbatim(engine, writes):
    engine.handle("add a beneficiary", "typo-2")
    engine.handle("Khaled Otaibi", "typo-2")
    engine.handle(MISTYPED_IBAN, "typo-2")

    insisted = engine.handle("I'm sure", "typo-2")
    assert insisted.state.status is ConversationStatus.CONFIRMING
    assert insisted.state.pending_add_account == MISTYPED_IBAN
    assert insisted.state.account_checksum_overridden
    assert "SA••7516" in insisted.reply
    assert "checksum" in insisted.reply  # the warning is restated before the write
    assert not writes

    engine.handle("yes", "typo-2")
    assert writes[0]["account"] == MISTYPED_IBAN


def test_a_corrected_iban_clears_the_override(engine, writes):
    engine.handle("add a beneficiary", "typo-3")
    engine.handle("Khaled Otaibi", "typo-3")
    engine.handle(MISTYPED_IBAN, "typo-3")

    fixed = engine.handle(VALID_IBAN, "typo-3")
    assert fixed.state.pending_add_account == VALID_IBAN
    assert not fixed.state.account_checksum_overridden
    assert fixed.state.pending_unchecked_account is None


@pytest.mark.parametrize("bad", ["SA1122330000007777", "1234", "abcd"])
def test_only_a_checksum_failure_can_be_overridden(engine, writes, bad: str):
    """A wrong length or shape is not a typo the customer can vouch for."""

    session = f"typo-4-{bad}"
    engine.handle("add a beneficiary", session)
    engine.handle("Khaled Otaibi", session)
    engine.handle(bad, session)

    insisted = engine.handle("I'm sure", session)
    assert insisted.state.pending_add_account is None
    assert insisted.state.status is ConversationStatus.COLLECTING
    assert not writes


def test_declining_the_flagged_iban_cancels(engine, writes):
    engine.handle("add a beneficiary", "typo-5")
    engine.handle("Khaled Otaibi", "typo-5")
    engine.handle(MISTYPED_IBAN, "typo-5")

    result = engine.handle("no", "typo-5")
    assert result.state.status is ConversationStatus.CANCELLED
    assert result.state.pending_unchecked_account is None
    assert not writes


def test_the_flagged_iban_prompt_is_what_an_aside_resumes(engine, writes):
    """A balance question mid-flow must not turn a later "yes" into a blind ok."""

    engine.handle("اضف مستفيد جديد", "typo-6")
    engine.handle("نورة سعد", "typo-6")
    engine.handle(MISTYPED_IBAN, "typo-6")

    aside = engine.handle("كم رصيدي", "typo-6")
    assert "الآيبان" in aside.reply
    assert "أنا متأكد" in aside.reply

    insisted = engine.handle("نعم", "typo-6")
    assert insisted.state.account_checksum_overridden
    assert insisted.state.language.value == "ar"


def test_add_request_is_not_a_listing_request(engine, writes):
    """The listing cue check must not swallow "add a beneficiary"."""

    result = engine.handle("add a new beneficiary", "add-8")
    assert result.state.intent is Intent.ADD_BENEFICIARY
    assert result.state.status is not ConversationStatus.COMPLETED
