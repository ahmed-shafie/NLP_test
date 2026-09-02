"""The answer to "who should I send it to?" has to be a person.

The recipient prompt takes the whole message as the answer, so anything typed
there used to become a payee name: an amount ("100"), an IBAN, even "no" — which
then matched *Moham**mmed No**ur* on a substring and locked their account.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import app.db.directory as directory
from app.config import settings
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.memory import service, store
from app.schemas import TransferRequest
from tests.test_beneficiary_directory import _seed_directory


@pytest.fixture()
def memory(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(settings, "memory_store_url", f"sqlite:///{tmp_path}/mem.db")
    monkeypatch.setattr(settings, "memory_cache_backend", "memory")
    monkeypatch.setattr(settings, "memory_enabled", True)
    store._get_engine.cache_clear()
    store._get_sessionmaker.cache_clear()
    store.get_memory_store.cache_clear()
    service._brain = None
    yield service.get_memory_brain()
    store._get_engine.cache_clear()
    store._get_sessionmaker.cache_clear()
    store.get_memory_store.cache_clear()
    service._brain = None


@pytest.fixture()
def directory_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "beneficiary_lookup_enabled", True)
    monkeypatch.setattr(
        settings, "beneficiary_db_url", _seed_directory(tmp_path / "d.db")
    )
    directory.get_beneficiary_directory.cache_clear()
    yield
    directory.get_beneficiary_directory.cache_clear()


def _ask_who(engine: ConversationEngine, session: str) -> None:
    """Get the conversation to the "who should I send it to?" prompt."""

    result = engine.handle("send 30 sar", session, user_id="demo")
    assert result.state.pending_slot == "recipient"


@pytest.mark.parametrize(
    "answer",
    ["100", "١٠٠", "sa1122330000002211", "SA11 2233 0000 0022 11", "no", "idk"],
)
def test_an_answer_that_is_not_a_name_names_nobody(
    memory, directory_db, answer: str
) -> None:
    engine = ConversationEngine()
    session = f"sid-{answer}"
    _ask_who(engine, session)

    result = engine.handle(answer, session, user_id="demo")

    assert result.state.slots.recipient is None
    assert result.state.slots.account_number is None
    assert result.state.pending_slot == "recipient"
    assert result.state.status is ConversationStatus.COLLECTING


@pytest.mark.parametrize(
    "answer",
    [
        "حياك الله",
        "صباح الخير",
        "السلام عليكم",
        "شكرا",
        "good morning",
        "thanks a lot",
    ],
)
def test_a_greeting_at_the_prompt_names_nobody(
    memory, directory_db, answer: str
) -> None:
    """A greeting is not a payee, so it must not reach the confirmation."""

    engine = ConversationEngine()
    session = f"greet-{answer}"
    _ask_who(engine, session)

    result = engine.handle(answer, session, user_id="demo")

    assert result.state.slots.recipient is None
    assert result.state.pending_slot == "recipient"
    assert result.state.status is ConversationStatus.COLLECTING


@pytest.mark.parametrize(
    "answer", ["اخبار الطقس", "الطقس", "the weather", "اخبار", "كم الساعة"]
)
def test_a_topic_at_the_prompt_names_nobody(memory, directory_db, answer: str) -> None:
    """The weather is a subject, not a payee: it must never reach an account."""

    engine = ConversationEngine()
    session = f"topic-{answer}"
    _ask_who(engine, session)

    result = engine.handle(answer, session, user_id="demo")

    assert result.state.slots.recipient is None
    assert result.state.slots.account_number is None
    assert result.state.status is not ConversationStatus.CONFIRMING


def test_a_name_typed_at_the_prompt_still_works(memory, directory_db) -> None:
    engine = ConversationEngine()
    _ask_who(engine, "sid-name")

    result = engine.handle("mona ali", "sid-name", user_id="demo")

    assert result.state.slots.recipient == "Mona Ali"
    assert result.state.slots.account_number == "SA1122330000003333"
    assert result.state.status is ConversationStatus.CONFIRMING


def test_the_prompt_accepts_the_remembered_recipient(memory, directory_db) -> None:
    """ "my usual" at the prompt means the remembered person, not a name."""

    memory.learn_from_transfer(
        "demo",
        TransferRequest(
            amount=Decimal("40"), currency="SAR", recipient="Mona Ali", note=None
        ),
    )
    engine = ConversationEngine()
    _ask_who(engine, "sid-usual")

    result = engine.handle("my usual", "sid-usual", user_id="demo")

    assert result.state.slots.recipient == "Mona Ali"
    assert result.state.slots.account_number == "SA1122330000003333"


def test_repeating_the_last_transfer_reuses_it_whole(memory, directory_db) -> None:
    memory.learn_from_transfer(
        "demo",
        TransferRequest(
            amount=Decimal("100"), currency="SAR", recipient="Ahmed Khaled", note=None
        ),
    )

    result = ConversationEngine().handle(
        "repeat my last transfer", "sid-repeat", user_id="demo"
    )

    assert result.state.slots.recipient == "Ahmed Khaled"
    assert result.state.slots.amount == Decimal("100")
    assert result.state.slots.currency == "SAR"
    assert result.state.slots.account_number == "SA1122330000002211"
    assert result.state.status is ConversationStatus.CONFIRMING


def test_an_amount_the_customer_states_beats_the_remembered_one(
    memory, directory_db
) -> None:
    memory.learn_from_transfer(
        "demo",
        TransferRequest(
            amount=Decimal("100"), currency="SAR", recipient="Ahmed Khaled", note=None
        ),
    )

    result = ConversationEngine().handle(
        "transfer 50 again", "sid-again", user_id="demo"
    )

    assert result.state.slots.amount == Decimal("50")
    assert result.state.slots.recipient == "Ahmed Khaled"


def test_nothing_to_repeat_still_asks(memory, directory_db) -> None:
    result = ConversationEngine().handle(
        "repeat my last transfer", "sid-empty", user_id="fresh-user"
    )

    assert result.state.slots.recipient is None
    assert result.state.status is ConversationStatus.COLLECTING
