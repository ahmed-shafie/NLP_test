"""Retrieval alone must not open a money flow.

The example index holds tens of thousands of rows, so a mixed-language aside
("thanks ممكن talk معاك دقيقة for a bit") can retrieve transfer neighbours and
used to be answered with "how much should I send?". A write intent now needs the
turn itself to ask for an action — a verb or an amount.
"""

from __future__ import annotations

import pytest

from app.conversation.engine import ConversationEngine
from app.nlu.intents import has_money_cue
from app.schemas import Intent, Language


@pytest.fixture
def engine() -> ConversationEngine:
    return ConversationEngine()


@pytest.mark.parametrize(
    "text",
    [
        "حول 500 لعمر",
        "ابغي احول فلوس",
        "ابغي اسدد فاتورة الكهرباء",
        "اضف نورة سعد كمستفيد",
        "i want to transfer money",
        "pay my stc bill",
        "500",
    ],
)
def test_an_action_request_carries_a_cue(text: str) -> None:
    assert has_money_cue(text)


@pytest.mark.parametrize(
    "text",
    [
        "thanks ممكن talk معاك دقيقة for a bit",
        "ممكن اتكلم معاك دقيقة",
        "can i talk to you for a bit",
        "صباح الخير",
        "شكرا جزيلا",
    ],
)
def test_chit_chat_carries_no_cue(text: str) -> None:
    assert not has_money_cue(text)


def test_a_mixed_language_aside_does_not_open_a_transfer(
    engine: ConversationEngine,
) -> None:
    result = engine.handle("thanks ممكن talk معاك دقيقة for a bit", "cue-aside")

    assert result.state.intent is not Intent.TRANSFER_MONEY
    assert result.state.pending_slot is None
    assert result.state.slots.amount is None


@pytest.mark.parametrize(
    ("text", "language", "intent"),
    [
        ("ابغي احول فلوس", Language.AR, Intent.TRANSFER_MONEY),
        ("i want to transfer money", Language.EN, Intent.TRANSFER_MONEY),
        ("ابغي اسدد فاتورة الكهرباء", Language.AR, Intent.PAY_BILL),
        ("اضف نورة سعد كمستفيد", Language.AR, Intent.ADD_BENEFICIARY),
    ],
)
def test_a_stated_request_still_opens_its_flow(
    engine: ConversationEngine, text: str, language: Language, intent: Intent
) -> None:
    result = engine.handle(text, f"cue-{intent.value}-{language.value}", language)

    assert result.state.intent is intent
