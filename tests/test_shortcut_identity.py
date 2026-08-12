"""A learned shortcut may complete a name, never swap in a different person.

Aliases are auto-created from repeated transfers and keyed on the recipient's *first
name* ("ahmed" -> "Ahmed Hassan"), so without a guard they answer for every message
mentioning that first name — including one that names somebody else entirely.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import app.db.directory as directory
from app.config import settings
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.memory import service, store
from app.memory.schemas import Shortcut
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


@pytest.fixture()
def engine(memory, directory_db) -> ConversationEngine:
    """Engine whose customer has the auto-created "ahmed" -> Ahmed Hassan alias."""

    memory.upsert_shortcut("demo", Shortcut(name="ahmed", recipient="Ahmed Hassan"))
    return ConversationEngine()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("send 100 sar to ahmed khaled", "SA1122330000002211"),
        ("send 100 sar to Ahmed Khaled", "SA1122330000002211"),
        ("send 100 sar to ahmed mahmoud", "SA1122330000008090"),
        ("حوّل ١٠٠ ريال لأحمد خالد", "SA1122330000002211"),
    ],
)
def test_a_shortcut_never_replaces_the_person_the_customer_named(
    engine: ConversationEngine, text: str, expected: str
) -> None:
    result = engine.handle(text, f"sid-{text}", user_id="demo")
    assert result.state.slots.account_number == expected
    assert result.state.status is ConversationStatus.CONFIRMING


def test_an_ambiguous_alias_asks_which_person(engine: ConversationEngine) -> None:
    """ "ahmed" fits three registered people, so memory must not pick one."""

    result = engine.handle("send 100 sar to ahmed", "sid-bare", user_id="demo")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert result.state.slots.account_number is None
    assert len(result.state.beneficiary_options) == 3


def test_an_alias_still_completes_an_unambiguous_name(memory, directory_db) -> None:
    """Nothing is lost where the alias is not ambiguous: "mona" -> Mona Ali."""

    memory.upsert_shortcut(
        "demo", Shortcut(name="mona", recipient="Mona Ali", amount=Decimal("75"))
    )
    result = ConversationEngine().handle("send to mona", "sid-mona", user_id="demo")
    assert result.state.slots.recipient == "Mona Ali"
    assert result.state.slots.account_number == "SA1122330000003333"
    assert result.state.slots.amount == Decimal("75")


def test_a_customer_saved_alias_is_untouched(memory, directory_db) -> None:
    """An alias the customer saved themselves is explicit intent, so it expands."""

    memory.upsert_shortcut(
        "demo",
        Shortcut(
            name="rent", recipient="Mona Ali", amount=Decimal("500"), currency="SAR"
        ),
    )
    result = ConversationEngine().handle("pay rent", "sid-rent", user_id="demo")
    assert result.state.slots.recipient == "Mona Ali"
    assert result.state.slots.amount == Decimal("500")
    assert result.state.status is ConversationStatus.CONFIRMING
