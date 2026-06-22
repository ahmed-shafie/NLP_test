"""Tests for the Memory Brain: store, service, engine integration, and endpoints."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.main import app
from app.memory import service, store
from app.memory.schemas import HabitsUpdate, Shortcut
from app.schemas import TransferRequest

client = TestClient(app)


@pytest.fixture()
def memory(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Point the Memory Brain at an isolated SQLite file and reset all caches."""

    db_url = f"sqlite:///{tmp_path}/mem.db"
    monkeypatch.setattr(settings, "memory_store_url", db_url)
    monkeypatch.setattr(settings, "memory_cache_backend", "memory")
    monkeypatch.setattr(settings, "memory_enabled", True)
    monkeypatch.setattr(settings, "memory_favorite_min_count", 2)

    store._get_engine.cache_clear()
    store._get_sessionmaker.cache_clear()
    store.get_memory_store.cache_clear()
    service._brain = None

    brain = service.get_memory_brain()
    yield brain

    store._get_engine.cache_clear()
    store._get_sessionmaker.cache_clear()
    store.get_memory_store.cache_clear()
    service._brain = None


# --------------------------------- store ----------------------------------- #


def test_empty_memory_for_unknown_user(memory):
    mem = memory.get_memory("nobody")
    assert mem.user_id == "nobody"
    assert mem.shortcuts == []
    assert mem.habits.total_transfers == 0


def test_update_and_read_habits(memory):
    memory.update_habits(
        "u1", HabitsUpdate(preferred_currency="EGP", preferred_source_account="ACC-1")
    )
    mem = memory.get_memory("u1")
    assert mem.habits.preferred_currency == "EGP"
    assert mem.habits.preferred_source_account == "ACC-1"


def test_shortcut_crud(memory):
    memory.upsert_shortcut(
        "u2",
        Shortcut(
            name="rent", amount=Decimal("5000"), currency="EGP", recipient="Landlord"
        ),
    )
    mem = memory.get_memory("u2")
    assert len(mem.shortcuts) == 1
    assert mem.shortcuts[0].recipient == "Landlord"
    # Upsert updates in place (no duplicate).
    memory.upsert_shortcut("u2", Shortcut(name="rent", amount=Decimal("5500")))
    assert len(memory.get_memory("u2").shortcuts) == 1
    assert memory.delete_shortcut("u2", "rent") is True
    assert memory.get_memory("u2").shortcuts == []


def test_resolve_shortcut_by_word_and_phrase(memory):
    memory.upsert_shortcut("u3", Shortcut(name="rent", recipient="Landlord"))
    memory.upsert_shortcut("u3", Shortcut(name="my mom", recipient="Mama"))
    assert memory.resolve_shortcut("u3", "pay rent please").name == "rent"
    assert memory.resolve_shortcut("u3", "send money to my mom").name == "my mom"
    assert memory.resolve_shortcut("u3", "send money to Ali") is None


def test_learn_from_transfer_builds_favorite(memory):
    t = TransferRequest(amount=Decimal("100"), currency="EGP", recipient="Ahmed")
    memory.learn_from_transfer("u4", t)
    assert memory.get_memory("u4").habits.favorite_recipient is None  # below threshold
    memory.learn_from_transfer("u4", t)
    habits = memory.get_memory("u4").habits
    assert habits.favorite_recipient == "Ahmed"
    assert habits.total_transfers == 2
    assert habits.preferred_currency == "EGP"
    assert Decimal("100") in habits.common_amounts


# --------------------------- engine integration ---------------------------- #


def test_engine_expands_shortcut(memory):
    memory.upsert_shortcut(
        "u5",
        Shortcut(
            name="rent", amount=Decimal("5000"), currency="EGP", recipient="Landlord"
        ),
    )
    engine = ConversationEngine()
    result = engine.handle("pay rent", "s5", user_id="u5")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.amount == Decimal("5000")
    assert result.state.slots.currency == "EGP"
    assert result.state.slots.recipient == "Landlord"


def test_engine_uses_default_currency_from_habits(memory):
    memory.update_habits("u6", HabitsUpdate(preferred_currency="EGP"))
    engine = ConversationEngine()
    # No currency stated; the habit fills it so we reach confirmation directly.
    result = engine.handle("send 50 to Ahmed", "s6", user_id="u6")
    assert result.state.slots.currency == "EGP"
    assert result.state.status is ConversationStatus.CONFIRMING


def test_engine_usual_recipient(memory):
    memory.update_habits("u7", HabitsUpdate(favorite_recipient="Sara"))
    engine = ConversationEngine()
    result = engine.handle("send 20 USD to my usual", "s7", user_id="u7")
    assert result.state.slots.recipient == "Sara"
    assert result.state.status is ConversationStatus.CONFIRMING


def test_engine_learns_on_completion(memory):
    engine = ConversationEngine()
    for sid in ("a", "b"):
        engine.handle("send 75 EGP to Mona", f"s8-{sid}", user_id="u8")
        engine.handle("yes", f"s8-{sid}", user_id="u8")
    habits = memory.get_memory("u8").habits
    assert habits.total_transfers == 2
    assert habits.favorite_recipient == "Mona"


def test_memory_ignored_without_user_id(memory):
    memory.update_habits("u9", HabitsUpdate(preferred_currency="EGP"))
    engine = ConversationEngine()
    result = engine.handle("send 50 to Ahmed", "s9")  # no user_id
    # Without a user, the habit currency (EGP) is ignored; the generic USD default
    # applies instead.
    assert result.state.slots.currency == "USD"


# ------------------------------- API endpoints ----------------------------- #


def test_memory_api_roundtrip(memory):
    put = client.put(
        "/memory/api-user/shortcuts",
        json={
            "name": "rent",
            "amount": "5000",
            "currency": "EGP",
            "recipient": "Landlord",
        },
    )
    assert put.status_code == 200
    assert len(put.json()["shortcuts"]) == 1

    got = client.get("/memory/api-user")
    assert got.status_code == 200
    assert got.json()["shortcuts"][0]["recipient"] == "Landlord"

    habits = client.put("/memory/api-user/habits", json={"preferred_currency": "USD"})
    assert habits.json()["habits"]["preferred_currency"] == "USD"

    deleted = client.delete("/memory/api-user/shortcuts/rent")
    assert deleted.status_code == 200
    assert deleted.json()["shortcuts"] == []

    missing = client.delete("/memory/api-user/shortcuts/nope")
    assert missing.status_code == 404


def test_memory_disabled_returns_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "memory_enabled", False)
    resp = client.get("/memory/anyone")
    assert resp.status_code == 503


# -------------------------------- monitoring ------------------------------- #


def test_overview_aggregates_across_users(memory):
    memory.upsert_shortcut("u1", Shortcut(name="rent", recipient="Landlord"))
    memory.learn_from_transfer(
        "u1", TransferRequest(amount=Decimal("75"), currency="EGP", recipient="Mona")
    )
    memory.learn_from_transfer(
        "u1", TransferRequest(amount=Decimal("75"), currency="EGP", recipient="Mona")
    )
    memory.learn_from_transfer(
        "u2", TransferRequest(amount=Decimal("10"), currency="USD", recipient="Mona")
    )

    overview = memory.overview()
    stats = overview.stats
    assert stats.total_users == 2
    assert stats.users_with_habits == 2
    assert stats.total_transfers == 3
    assert stats.total_shortcuts == 1
    assert stats.currency_distribution == {"EGP": 1, "USD": 1}
    # Mona is the top recipient with 3 transfers aggregated across users.
    assert stats.top_recipients[0].recipient == "Mona"
    assert stats.top_recipients[0].count == 3
    # Users are sorted by transfer volume (u1 has more).
    assert overview.users[0].user_id == "u1"
    assert overview.users[0].total_transfers == 2


def test_overview_endpoint(memory):
    client.put("/memory/ovu/shortcuts", json={"name": "rent", "recipient": "L"})
    resp = client.get("/memory")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stats"]["total_users"] == 1
    assert body["stats"]["total_shortcuts"] == 1
    assert body["users"][0]["user_id"] == "ovu"


def test_overview_disabled_returns_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "memory_enabled", False)
    resp = client.get("/memory")
    assert resp.status_code == 503
