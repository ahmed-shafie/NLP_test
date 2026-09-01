"""A resolved recipient is not authorization: which account, and what for.

From a live Saudi session: picking a beneficiary jumped straight to "confirm —
432 SAR", never asking which of the customer's two accounts the money leaves or
what the transfer is for. Both are asked now, in that order, and the numbers the
customer types at those prompts are picks from the printed list — never an
amount.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app import banking_core_client
from app.banking_core_client import AccountInfo
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.nlu import entities
from app.schemas import Language

CURRENT = AccountInfo(
    account_id="acc-current",
    account_type="current",
    number="SA1234567890",
    currency="SAR",
    balance=Decimal("12300.00"),
    status="active",
)
SAVINGS = AccountInfo(
    account_id="acc-savings",
    account_type="savings",
    number="SA1234568877",
    currency="SAR",
    balance=Decimal("5000.00"),
    status="active",
)


@pytest.fixture
def engine() -> ConversationEngine:
    return ConversationEngine()


@pytest.fixture
def two_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        banking_core_client, "list_accounts", lambda owner_user: [CURRENT, SAVINGS]
    )


@pytest.fixture
def one_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        banking_core_client, "list_accounts", lambda owner_user: [CURRENT]
    )


def test_a_named_recipient_is_asked_which_account_then_what_for(
    engine: ConversationEngine, two_accounts: None
) -> None:
    account = engine.handle("حوّل ٤٣٢ لعمر", "gates-ar")
    assert "من أي حساب" in account.reply

    purpose = engine.handle("1", "gates-ar")
    assert "غرض التحويل" in purpose.reply
    assert "شخصي" in purpose.reply

    confirm = engine.handle("1", "gates-ar")
    assert "432" in confirm.reply


def test_the_account_prompt_prints_the_cores_own_numbers(
    engine: ConversationEngine, two_accounts: None
) -> None:
    result = engine.handle("حوّل ٤٣٢ لعمر", "gates-numbers")

    assert "من أي حساب" in result.reply
    # Balances, masks and currency are printed exactly as the Core reported them.
    assert "SA••7890 — 12300.00 SAR" in result.reply
    assert "SA••8877 — 5000.00 SAR" in result.reply


def test_the_account_pick_is_not_read_as_an_amount(
    engine: ConversationEngine, two_accounts: None
) -> None:
    engine.handle("حوّل ٤٣٢ لعمر", "gates-amount")

    purpose = engine.handle("1", "gates-amount")

    assert "غرض التحويل" in purpose.reply
    state = engine._store.load("gates-amount")
    assert state is not None
    assert state.slots.amount == Decimal("432")
    assert state.slots.source_account == "acc-current"


def test_the_purpose_pick_completes_the_gates_and_confirms(
    engine: ConversationEngine, two_accounts: None
) -> None:
    engine.handle("حوّل ٤٣٢ لعمر", "gates-confirm")
    engine.handle("2", "gates-confirm")

    confirm = engine.handle("4", "gates-confirm")

    state = engine._store.load("gates-confirm")
    assert state is not None
    assert state.status is ConversationStatus.CONFIRMING
    assert state.slots.source_account == "acc-savings"
    assert state.slots.transfer_purpose == "rent"
    # The amount and currency are untouched by either gate.
    assert state.slots.amount == Decimal("432")
    assert "432" in confirm.reply


def test_an_unreadable_account_pick_re_asks_the_same_list(
    engine: ConversationEngine, two_accounts: None
) -> None:
    first = engine.handle("حوّل ٤٣٢ لعمر", "gates-bad-account")

    again = engine.handle("9", "gates-bad-account")

    assert again.reply == first.reply
    state = engine._store.load("gates-bad-account")
    assert state is not None
    assert state.slots.source_account is None


def test_an_unreadable_purpose_pick_re_asks_the_purpose(
    engine: ConversationEngine, two_accounts: None
) -> None:
    engine.handle("حوّل ٤٣٢ لعمر", "gates-bad-purpose")
    prompt = engine.handle("1", "gates-bad-purpose")

    again = engine.handle("42", "gates-bad-purpose")

    assert again.reply == prompt.reply
    state = engine._store.load("gates-bad-purpose")
    assert state is not None
    assert state.slots.transfer_purpose is None
    assert state.slots.amount == Decimal("432")


def test_a_customer_with_one_account_is_not_asked_to_choose(
    engine: ConversationEngine, one_account: None
) -> None:
    result = engine.handle("حوّل ٤٣٢ لعمر", "gates-single")

    assert "من أي حساب" not in result.reply
    assert "غرض التحويل" in result.reply
    state = engine._store.load("gates-single")
    assert state is not None
    assert state.slots.source_account == "acc-current"


def test_an_account_can_be_picked_by_name_or_last_digits(
    engine: ConversationEngine, two_accounts: None
) -> None:
    engine.handle("حوّل ٤٣٢ لعمر", "gates-by-name")
    engine.handle("التوفير", "gates-by-name")

    state = engine._store.load("gates-by-name")
    assert state is not None
    assert state.slots.source_account == "acc-savings"

    engine.handle("حوّل ٤٣٢ لعمر", "gates-by-digits")
    engine.handle("7890", "gates-by-digits")

    state = engine._store.load("gates-by-digits")
    assert state is not None
    assert state.slots.source_account == "acc-current"


def test_the_english_prompts_are_asked_in_english(
    engine: ConversationEngine, two_accounts: None
) -> None:
    account = engine.handle("transfer 432 to omar", "gates-en", language=Language.EN)
    assert "Which account" in account.reply
    assert "Current Account SA••7890" in account.reply

    purpose = engine.handle("1", "gates-en", language=Language.EN)
    assert "Family support" in purpose.reply

    engine.handle("Rent", "gates-en", language=Language.EN)
    state = engine._store.load("gates-en")
    assert state is not None
    assert state.slots.transfer_purpose == "rent"


def test_a_purpose_is_never_inferred_from_the_transfer_sentence(
    engine: ConversationEngine, one_account: None
) -> None:
    result = engine.handle("حوّل ٤٣٢ لعمر إيجار", "gates-no-infer")

    assert "غرض التحويل" in result.reply
    state = engine._store.load("gates-no-infer")
    assert state is not None
    assert state.slots.transfer_purpose is None


def test_no_account_list_asks_neither_gate(
    engine: ConversationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unreachable or disabled Core means "no list to choose from", never
    # "the customer has no money": there is no account to pick and no
    # instruction to label, so the flow reaches confirmation as it always did.
    monkeypatch.setattr(banking_core_client, "list_accounts", lambda owner_user: [])

    result = engine.handle("حوّل ٤٣٢ لعمر", "gates-no-core")

    assert "من أي حساب" not in result.reply
    assert "غرض التحويل" not in result.reply
    state = engine._store.load("gates-no-core")
    assert state is not None
    assert state.status is ConversationStatus.CONFIRMING


@pytest.mark.parametrize(
    ("text", "recipient"),
    [
        ("حوّل ٤٣٢ لعمر إيجار", "عمر"),
        ("حوّل ٤٣٢ لعمر ايجار", "عمر"),
        ("حوّل ٥٠٠ لسارة تعليم", "سارة"),
        ("حوّل ٥٠٠ لعبدالله راتب", "عبدالله"),
    ],
)
def test_a_purpose_word_is_not_part_of_the_payees_name(
    text: str, recipient: str
) -> None:
    # "حوّل ٤٣٢ لعمر إيجار" used to name a payee "عمر أجار" — the speller turned
    # the purpose noun into a given name — and offered to save that person.
    assert entities.extract_recipient(text, Language.AR) == recipient


def test_a_purpose_word_alone_is_not_a_payee() -> None:
    assert entities.extract_recipient("حوّل ٤٣٢ لإيجار", Language.AR) is None
