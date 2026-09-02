"""Two payees in one sentence: ask, never pair a name with a guessed amount.

"حول 500 لمحمد و300 لعمر" used to reach a confirmation for "500 SAR إلى محمد و"
— the conjunction glued onto the payee's name, the second transfer dropped in
silence. Which amount belongs to which name is the customer's to state.
"""

from __future__ import annotations

import pytest

from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.nlu import entities
from app.schemas import Intent, Language


@pytest.fixture
def engine() -> ConversationEngine:
    return ConversationEngine()


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("حول 500 لمحمد و300 لعمر", ["محمد", "عمر"]),
        ("حوّل ٥٠٠ لمحمد و٣٠٠ لعمر", ["محمد", "عمر"]),
        ("حول لمحمد وعمر 500", ["محمد", "عمر"]),
    ],
)
def test_both_payees_are_read(text: str, names: list[str]) -> None:
    assert entities.extract_recipients(text, Language.AR) == names


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("حول 500 لمحمد و300 لعمر", "محمد"),
        ("حول لمحمد وعمر 500", "محمد"),
        # A given name that opens with the conjunction letter stays whole.
        ("حوّل 500 لوليد", "وليد"),
        ("حول لمحمد نور 500", "محمد نور"),
    ],
)
def test_the_conjunction_is_never_part_of_the_name(text: str, name: str) -> None:
    assert entities.extract_recipient(text, Language.AR) == name


def test_two_payees_are_asked_about_instead_of_confirmed(
    engine: ConversationEngine,
) -> None:
    result = engine.handle("حول 500 لمحمد و300 لعمر", "payees-ar")

    assert result.state.status is ConversationStatus.COLLECTING
    assert result.state.pending_slot == "recipient"
    assert result.state.slots.recipient is None
    assert result.state.slots.amount is None
    assert "محمد" in result.reply and "عمر" in result.reply


def test_naming_one_of_them_continues_that_transfer_alone(
    engine: ConversationEngine,
) -> None:
    engine.handle("حول 500 لمحمد و300 لعمر", "payees-pick")

    picked = engine.handle("محمد", "payees-pick")

    assert picked.state.intent is Intent.TRANSFER_MONEY
    assert picked.state.slots.recipient == "محمد"
    assert picked.state.pending_slot == "amount"
    assert picked.state.slots.amount is None


def test_two_payees_in_english_are_asked_about_too(
    engine: ConversationEngine,
) -> None:
    result = engine.handle(
        "transfer 500 to Omar and 300 to Sara", "payees-en", language=Language.EN
    )

    assert result.state.pending_slot == "recipient"
    assert result.state.slots.recipient is None
    assert "Omar" in result.reply and "Sara" in result.reply


def test_a_single_payee_still_reaches_its_confirmation(
    engine: ConversationEngine,
) -> None:
    result = engine.handle("حوّل ٤٣٢ لعمر", "payees-single")

    assert result.state.slots.recipient == "عمر"
    assert "432" in result.reply
