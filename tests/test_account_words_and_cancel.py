"""An account is not a payee, its digits are not money, and "كنسل" cancels.

From a live Arabic session: "حول الي ايبان ٠٦٦٦٣٥" was read as a transfer of
66,635 SAR to a person called "ايان", and neither "بزاف", "كنسل" nor "cansel"
would take the confirmation screen down.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.nlu import entities
from app.schemas import Language


@pytest.fixture
def engine() -> ConversationEngine:
    return ConversationEngine()


ACCOUNT_UTTERANCES = [
    "حول الي ايبان ٠٦٦٦٣٥",
    "حول الي هذا الحساب ٠١١١٥٥٥٢٤٢",
    "حوّل الى الايبان SA0380000000608010167519",
    "حوّل لرقم الحساب ١٢٣٤٥٦٧٨٩٠",
]


@pytest.mark.parametrize("text", ACCOUNT_UTTERANCES)
def test_an_account_is_neither_a_name_nor_an_amount(text: str) -> None:
    assert entities.extract_recipient(text, Language.AR) is None
    assert entities.extract_amount(text) is None


@pytest.mark.parametrize(
    ("text", "recipient", "amount"),
    [
        ("حوّل ٥٠٠ لمحمد نور", "محمد نور", Decimal("500")),
        ("حوّل الى سارة 500", "سارة", Decimal("500")),
        ("ابعت لمنى ٣٠٠ ريال", "منى", Decimal("300")),
        ("حوّل ٥٠٠ ريال ل عبدالله", "عبدالله", Decimal("500")),
    ],
)
def test_ordinary_arabic_transfers_still_read(
    text: str, recipient: str, amount: Decimal
) -> None:
    assert entities.extract_recipient(text, Language.AR) == recipient
    assert entities.extract_amount(text) == amount


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("حوّل ألف جنيه إلى محمد", Decimal("1000")),
        ("ابعت خمسة آلاف ريال لسارة", Decimal("5000")),
        ("حوّل خمسمية ريال لمنى", Decimal("500")),
        ("send five hundred dollars to Sara", Decimal("500")),
        ("send money to one of my beneficiaries", None),
    ],
)
def test_amounts_written_in_words_are_read_deterministically(
    text: str, expected: Decimal | None
) -> None:
    assert entities.extract_amount(text) == expected


@pytest.mark.parametrize("word", ["كنسل", "بطل", "الغي", "أوقف", "cansel", "cancle"])
def test_cancel_words_take_down_the_confirmation(
    engine: ConversationEngine, word: str
) -> None:
    engine.handle("حوّل ٥٠٠ لمحمد نور", f"cancel-{word}")
    result = engine.handle(word, f"cancel-{word}")
    assert result.state.status is ConversationStatus.CANCELLED
    assert result.state.slots.amount is None


def test_cancelling_in_english_keeps_an_arabic_conversation_arabic(
    engine: ConversationEngine,
) -> None:
    engine.handle("حوّل ٥٠٠ لمحمد نور", "cancel-lang")
    result = engine.handle("cansel", "cancel-lang")
    assert result.state.language is Language.AR
    assert not any(c.isascii() and c.isalpha() for c in result.reply)
