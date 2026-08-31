"""A fresh instruction mid-flow is not the answer to the question we asked.

The defect these cover: while a bill payment waited for its reference number,
"transfer 250 riyals to Sara Adel" was read as reference number 250 — a figure
the customer never gave for that bill. The flow now stops and asks which of the
two requests to serve, and picking the new one replays their own message.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.conversation.engine import ConversationEngine
from app.conversation.reasons import ReasonCode
from app.conversation.state import ConversationStatus
from app.conversation.turn_role import TurnRole, classify_turn_role
from app.schemas import Intent, Language


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


def _pending_bill_reference(engine: ConversationEngine, session: str) -> None:
    """Drive a bill payment to the point where it is waiting for the reference."""

    result = engine.handle("ادفع فاتورة stc", session_id=session)
    assert result.state.pending_slot == "reference_number"


# --------------------------------------------------------------------------- #
# The role decision itself (pure, no dialogue).
# --------------------------------------------------------------------------- #


def test_bare_value_answers_the_prompt() -> None:
    evidence = classify_turn_role(
        active_intent=Intent.PAY_BILL,
        pending_slot="reference_number",
        requested_flow=None,
        is_instruction=False,
        is_question=False,
        control_only=False,
    )
    assert evidence.role is TurnRole.ANSWER


def test_instruction_for_another_flow_is_a_new_request() -> None:
    evidence = classify_turn_role(
        active_intent=Intent.PAY_BILL,
        pending_slot="reference_number",
        requested_flow=Intent.TRANSFER_MONEY,
        is_instruction=True,
        is_question=False,
        control_only=False,
    )
    assert evidence.role is TurnRole.NEW_REQUEST
    assert evidence.requested_flow is Intent.TRANSFER_MONEY


def test_instruction_for_the_same_flow_still_answers() -> None:
    """Restating the flow we are already in is not a switch."""

    evidence = classify_turn_role(
        active_intent=Intent.PAY_BILL,
        pending_slot="reference_number",
        requested_flow=Intent.PAY_BILL,
        is_instruction=True,
        is_question=False,
        control_only=False,
    )
    assert evidence.role is TurnRole.ANSWER


def test_menu_pick_is_never_a_new_request() -> None:
    evidence = classify_turn_role(
        active_intent=Intent.TRANSFER_MONEY,
        pending_slot="amount",
        requested_flow=Intent.PAY_BILL,
        is_instruction=True,
        is_question=False,
        control_only=True,
    )
    assert evidence.role is TurnRole.ANSWER


def test_question_mid_flow_is_an_aside() -> None:
    evidence = classify_turn_role(
        active_intent=Intent.TRANSFER_MONEY,
        pending_slot="recipient",
        requested_flow=None,
        is_instruction=False,
        is_question=True,
        control_only=False,
    )
    assert evidence.role is TurnRole.ASIDE


def test_no_flow_in_progress_is_left_alone() -> None:
    evidence = classify_turn_role(
        active_intent=None,
        pending_slot=None,
        requested_flow=Intent.TRANSFER_MONEY,
        is_instruction=True,
        is_question=False,
        control_only=False,
    )
    assert evidence.role is TurnRole.ANSWER


# --------------------------------------------------------------------------- #
# The dialogue: the transfer request that used to become a reference number.
# --------------------------------------------------------------------------- #


def test_transfer_request_is_not_consumed_as_a_bill_reference(
    engine: ConversationEngine,
) -> None:
    _pending_bill_reference(engine, "sw-1")

    result = engine.handle("حول 250 ريال لسارة عادل", session_id="sw-1")

    assert result.reason is ReasonCode.FLOW_SWITCH_REQUIRED
    # Nothing from the transfer leaked into the bill.
    assert result.state.slots.reference_number is None
    assert result.state.slots.amount is None
    assert result.state.pending_slot == "reference_number"
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.pending_switch is not None
    assert result.state.pending_switch.intent is Intent.TRANSFER_MONEY


def test_choosing_the_transfer_replays_the_customers_own_message(
    engine: ConversationEngine,
) -> None:
    _pending_bill_reference(engine, "sw-2")
    engine.handle("حول 250 ريال لسارة عادل", session_id="sw-2")

    result = engine.handle("تحويل", session_id="sw-2")

    assert result.state.intent is Intent.TRANSFER_MONEY
    assert result.state.slots.amount == Decimal("250")
    assert result.state.slots.recipient is not None
    assert "سارة" in result.state.slots.recipient
    # The bill's own slots are gone with the flow it belonged to.
    assert result.state.slots.biller is None
    assert result.state.pending_switch is None


def test_choosing_the_bill_resumes_it_untouched(engine: ConversationEngine) -> None:
    _pending_bill_reference(engine, "sw-3")
    engine.handle("حول 250 ريال لسارة عادل", session_id="sw-3")

    result = engine.handle("الفاتورة", session_id="sw-3")

    assert result.state.intent is Intent.PAY_BILL
    assert result.state.pending_slot == "reference_number"
    assert result.state.slots.reference_number is None
    assert result.state.slots.amount is None
    assert result.reason is ReasonCode.SLOT_REQUIRED
    assert result.state.pending_switch is None


def test_yes_at_the_switch_prompt_starts_the_new_request(
    engine: ConversationEngine,
) -> None:
    _pending_bill_reference(engine, "sw-4")
    engine.handle("حول 250 ريال لسارة عادل", session_id="sw-4")

    result = engine.handle("نعم", session_id="sw-4")

    assert result.state.intent is Intent.TRANSFER_MONEY
    assert result.state.slots.amount == Decimal("250")


def test_an_unrecognised_reply_asks_again_instead_of_guessing(
    engine: ConversationEngine,
) -> None:
    _pending_bill_reference(engine, "sw-5")
    engine.handle("حول 250 ريال لسارة عادل", session_id="sw-5")

    result = engine.handle("مش عارف", session_id="sw-5")

    assert result.reason is ReasonCode.FLOW_SWITCH_REQUIRED
    assert result.state.pending_switch is not None
    assert result.state.intent is Intent.PAY_BILL
    assert result.state.slots.reference_number is None


def test_english_transfer_request_mid_bill(engine: ConversationEngine) -> None:
    result = engine.handle("pay my stc bill", session_id="sw-6")
    assert result.state.pending_slot == "reference_number"

    result = engine.handle("send 300 sar to mona ali", session_id="sw-6")

    assert result.reason is ReasonCode.FLOW_SWITCH_REQUIRED
    assert result.state.slots.reference_number is None


def test_cancel_still_wins_over_a_held_request(engine: ConversationEngine) -> None:
    _pending_bill_reference(engine, "sw-7")
    engine.handle("حول 250 ريال لسارة عادل", session_id="sw-7")

    result = engine.handle("الغاء", session_id="sw-7")

    assert result.state.status is ConversationStatus.CANCELLED
    assert result.reason is ReasonCode.CANCELLED_BY_CUSTOMER
    assert result.state.pending_switch is None


# --------------------------------------------------------------------------- #
# Answers must keep working: the fix must not strand a flow.
# --------------------------------------------------------------------------- #


def test_a_bill_named_without_a_verb_is_still_a_new_request(
    engine: ConversationEngine,
) -> None:
    """A bill noun plus a number is not a payee name for the open transfer."""

    result = engine.handle("حول 250 ريال", session_id="reverse-1")
    assert result.state.pending_slot == "recipient"

    result = engine.handle("فاتوره 53535", session_id="reverse-1")

    assert result.state.intent is Intent.TRANSFER_MONEY
    assert result.state.slots.recipient is None
    assert result.state.slots.amount == Decimal("250")
    assert result.state.pending_switch is not None
    assert result.state.pending_switch.intent is Intent.PAY_BILL
    assert result.reason is ReasonCode.FLOW_SWITCH_REQUIRED


def test_a_bare_reference_number_still_answers(engine: ConversationEngine) -> None:
    _pending_bill_reference(engine, "keep-1")

    result = engine.handle("778899", session_id="keep-1")

    assert result.state.slots.reference_number == "778899"
    assert result.state.pending_switch is None


def test_a_bare_recipient_still_answers(engine: ConversationEngine) -> None:
    result = engine.handle("حول 500 ريال", session_id="keep-2")
    assert result.state.pending_slot == "recipient"

    result = engine.handle("منى علي", session_id="keep-2")

    assert result.state.slots.recipient is not None
    assert result.state.pending_switch is None


def test_an_amount_answer_naming_no_one_still_answers(
    engine: ConversationEngine,
) -> None:
    result = engine.handle("حول لمنى علي", session_id="keep-3")
    assert result.state.pending_slot == "amount"

    result = engine.handle("500 ريال", session_id="keep-3")

    assert result.state.slots.amount == Decimal("500")
    assert result.state.pending_switch is None


def test_more_bill_detail_mid_bill_is_not_a_switch(engine: ConversationEngine) -> None:
    """The same flow restated carries on filling slots."""

    _pending_bill_reference(engine, "keep-4")

    result = engine.handle("ادفع فاتورة stc 778899", session_id="keep-4")

    assert result.state.pending_switch is None
    assert result.state.slots.reference_number == "778899"


def test_a_first_turn_transfer_request_is_unaffected(
    engine: ConversationEngine,
) -> None:
    result = engine.handle("حول 250 ريال لسارة عادل", session_id="keep-5")

    assert result.state.intent is Intent.TRANSFER_MONEY
    assert result.state.pending_switch is None
    assert result.reason is not ReasonCode.FLOW_SWITCH_REQUIRED


def test_language_of_the_switch_prompt_follows_the_conversation(
    engine: ConversationEngine,
) -> None:
    _pending_bill_reference(engine, "sw-8")

    result = engine.handle("حول 250 ريال لسارة عادل", session_id="sw-8")

    assert result.state.language is Language.AR
