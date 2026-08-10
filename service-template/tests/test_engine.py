"""End-to-end tests for the template engine's happy paths and edge cases.

These double as living documentation of the expected behaviour. Run with::

    pytest service-template/tests -q
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from service_template.engine import ConversationEngine
from service_template.schemas import Language
from service_template.state import ConversationStatus


@pytest.fixture()
def engine() -> ConversationEngine:
    # A fresh engine per test; the in-memory store is keyed by session id so
    # unique session ids keep tests isolated.
    return ConversationEngine()


def test_one_shot_transfer_then_confirm(engine: ConversationEngine) -> None:
    """A fully-specified request with a unique name goes straight to confirm."""

    first = engine.handle("send 500 SAR to Mona", "s1")
    assert first.state.status is ConversationStatus.CONFIRMING
    assert first.state.slots.amount == Decimal("500")
    assert first.state.slots.currency == "SAR"
    assert first.state.slots.recipient == "Mona Ali"  # resolved from directory

    done = engine.handle("yes", "s1")
    assert done.state.status is ConversationStatus.COMPLETED
    assert done.action is not None
    assert done.action.recipient == "Mona Ali"
    assert done.action.amount == Decimal("500")


def test_slot_by_slot_collection(engine: ConversationEngine) -> None:
    """Missing slots are asked for one at a time, without clobbering."""

    r1 = engine.handle("I want to send money", "s2")
    assert r1.state.status is ConversationStatus.COLLECTING
    assert r1.state.pending_slot == "amount"

    r2 = engine.handle("300", "s2")
    # Amount filled (currency defaults to SAR); next missing is recipient.
    assert r2.state.slots.amount == Decimal("300")
    assert r2.state.pending_slot == "recipient"

    r3 = engine.handle("to Mona", "s2")
    assert r3.state.status is ConversationStatus.CONFIRMING


def test_disambiguation_on_shared_first_name(engine: ConversationEngine) -> None:
    """Several 'Ahmed's trigger a disambiguation turn; a numeric pick resolves it."""

    r1 = engine.handle("send 100 SAR to Ahmed", "s3")
    assert r1.state.status is ConversationStatus.DISAMBIGUATING
    assert len(r1.state.candidates) == 3

    r2 = engine.handle("2", "s3")
    assert r2.state.status is ConversationStatus.CONFIRMING
    assert r2.state.slots.recipient == "Ahmed Khaled"


def test_cancel_resets(engine: ConversationEngine) -> None:
    engine.handle("send 100 SAR to Ahmed", "s4")
    r = engine.handle("cancel", "s4")
    assert r.state.status is ConversationStatus.CANCELLED
    assert r.state.slots.recipient is None


def test_arabic_language_detected(engine: ConversationEngine) -> None:
    r = engine.handle("حوّل ٢٠٠ ريال إلى منى", "s5")
    assert r.state.language is Language.AR
    # 200 parsed from Arabic-Indic digits, Mona resolved, straight to confirm.
    assert r.state.slots.amount == Decimal("200")
    assert r.state.status is ConversationStatus.CONFIRMING


def test_no_flow_started_on_greeting(engine: ConversationEngine) -> None:
    r = engine.handle("hello", "s6")
    assert r.state.status is ConversationStatus.COLLECTING
    assert r.state.intent is None  # small talk does not start an action
