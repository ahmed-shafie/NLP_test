"""Machine-readable turn reasons and per-slot provenance.

The reply text is for the customer; these two fields are the same decision in a
form a report can count. The assertions below therefore never look at wording —
they check the code and the recorded source of each slot.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

import app.banking_core_client as bcc
import app.db.directory as directory
from app.config import settings
from app.conversation.engine import ConversationEngine
from app.conversation.reasons import ReasonCode
from app.conversation.state import ConversationStatus, SlotSource
from app.main import app
from app.memory import service, store
from app.memory.schemas import HabitsUpdate

client = TestClient(app)


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


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
def directory_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two beneficiaries sharing a first name, plus one unique name."""

    url = f"sqlite:///{tmp_path}/dir.db"
    db = create_engine(url)
    with db.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE beneficiaries (id TEXT, owner_user TEXT, name TEXT, "
                "name_ar TEXT, account TEXT, bank TEXT, currency TEXT, "
                "status TEXT, is_favorite INTEGER)"
            )
        )
        rows = [
            ("B1", "demo", "Ahmed Hassan", "أحمد حسن", "SA1122330000007777"),
            ("B2", "demo", "Ahmed Khaled", "أحمد خالد", "SA1122330000002211"),
            ("B3", "demo", "Mona Ali", "منى علي", "SA1122330000003333"),
        ]
        for bid, owner, name, name_ar, account in rows:
            conn.execute(
                text(
                    "INSERT INTO beneficiaries VALUES "
                    "(:id,:o,:n,:na,:a,'SNB','SAR','active',0)"
                ),
                {"id": bid, "o": owner, "n": name, "na": name_ar, "a": account},
            )
    db.dispose()
    monkeypatch.setattr(settings, "beneficiary_lookup_enabled", True)
    monkeypatch.setattr(settings, "beneficiary_db_url", url)
    directory.get_beneficiary_directory.cache_clear()
    yield url
    directory.get_beneficiary_directory.cache_clear()


@pytest.fixture()
def poor_core(monkeypatch: pytest.MonkeyPatch):
    """A Banking Core that refuses for lack of funds."""

    monkeypatch.setattr(
        bcc,
        "get_balance",
        lambda owner_user, account=None, account_type=None: bcc.AccountInfo(
            account_id="ACC-1",
            account_type="current",
            number="SA0380009999888877",
            currency="SAR",
            balance=Decimal("100.00"),
            status="active",
        ),
    )
    monkeypatch.setattr(
        bcc,
        "preflight_transfer",
        lambda **k: bcc.PreflightResult(
            ok=False, warnings=[], blocking=["insufficient_funds:available=100.00"]
        ),
    )
    return bcc


# ------------------------------- reason codes ------------------------------ #


def test_missing_slot_carries_slot_required(engine: ConversationEngine):
    result = engine.handle("I want to transfer money", "r1")
    assert result.state.pending_slot == "amount"
    assert result.reason is ReasonCode.SLOT_REQUIRED


def test_a_turn_that_moves_forward_has_no_reason(engine: ConversationEngine):
    result = engine.handle("send 500 SAR to Ahmed", "r2")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.reason is None


def test_completion_has_no_reason(engine: ConversationEngine):
    engine.handle("send 500 SAR to Ahmed", "r3")
    done = engine.handle("yes", "r3")
    assert done.state.status is ConversationStatus.COMPLETED
    assert done.reason is None


def test_cancel_carries_cancelled_by_customer(engine: ConversationEngine):
    engine.handle("send 500 SAR to Ahmed", "r4")
    result = engine.handle("cancel", "r4")
    assert result.state.status is ConversationStatus.CANCELLED
    assert result.reason is ReasonCode.CANCELLED_BY_CUSTOMER


def test_unreadable_confirmation_carries_its_own_code(engine: ConversationEngine):
    engine.handle("send 500 SAR to Ahmed", "r5")
    result = engine.handle("maybe later", "r5")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.reason is ReasonCode.CONFIRMATION_NOT_RECOGNISED


def test_abuse_carries_inappropriate_input(engine: ConversationEngine):
    result = engine.handle("you are fucking useless", "r6")
    assert result.flagged_terms
    assert result.reason is ReasonCode.INAPPROPRIATE_INPUT


def test_ambiguous_beneficiary_carries_its_code(
    engine: ConversationEngine, directory_db
):
    result = engine.handle("send 100 SAR to Ahmed", "r7", user_id="demo")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert result.reason is ReasonCode.AMBIGUOUS_BENEFICIARY


def test_unreadable_choice_carries_choice_not_recognised(
    engine: ConversationEngine, directory_db
):
    engine.handle("send 100 SAR to Ahmed", "r8", user_id="demo")
    result = engine.handle("none of those", "r8", user_id="demo")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert result.reason is ReasonCode.CHOICE_NOT_RECOGNISED


def test_unknown_name_carries_beneficiary_not_found(
    engine: ConversationEngine, directory_db
):
    result = engine.handle("send 100 SAR to Zuhair", "r9", user_id="demo")
    assert result.reason is ReasonCode.BENEFICIARY_NOT_FOUND


def test_core_refusal_carries_insufficient_funds(
    engine: ConversationEngine, directory_db, poor_core
):
    result = engine.handle("send 5000 SAR to Mona", "r10", user_id="demo")
    assert result.reason is ReasonCode.INSUFFICIENT_FUNDS


def test_biller_outside_the_catalogue_carries_its_code(engine: ConversationEngine):
    result = engine.handle("pay my Netflix bill", "r11")
    assert result.reason is ReasonCode.BILLER_NOT_IN_CATALOGUE


# -------------------------------- provenance ------------------------------- #


def test_stated_values_are_attributed_to_the_customer(engine: ConversationEngine):
    result = engine.handle("send 500 USD to Ahmed", "p1")
    prov = result.state.slot_provenance
    assert prov["amount"] == SlotSource.USER_TEXT.value
    assert prov["currency"] == SlotSource.USER_TEXT.value
    assert prov["recipient"] == SlotSource.USER_TEXT.value


def test_an_unstated_currency_is_marked_as_a_default(engine: ConversationEngine):
    result = engine.handle("send 500 to Ahmed", "p2")
    assert result.state.slots.currency == "SAR"
    assert result.state.slot_provenance["currency"] == SlotSource.DEFAULT.value
    assert result.state.slot_provenance["amount"] == SlotSource.USER_TEXT.value


def test_a_remembered_currency_is_marked_as_memory(engine: ConversationEngine, memory):
    memory.update_habits("pm", HabitsUpdate(preferred_currency="EGP"))
    result = engine.handle("send 50 to Ahmed", "p3", user_id="pm")
    assert result.state.slots.currency == "EGP"
    assert result.state.slot_provenance["currency"] == SlotSource.MEMORY_SHORTCUT.value


def test_a_resolved_recipient_is_attributed_to_the_directory(
    engine: ConversationEngine, directory_db
):
    """The name the Core is given must come from the directory, not the text.

    A recipient still marked ``user_text`` at confirmation would mean identity
    was never resolved, so this is the assertion that matters most.
    """

    result = engine.handle("send 100 SAR to Mona", "p4", user_id="demo")
    assert result.state.status is ConversationStatus.CONFIRMING
    prov = result.state.slot_provenance
    assert result.state.slots.recipient == "Mona Ali"
    assert prov["recipient"] == SlotSource.DIRECTORY.value
    assert prov["account_number"] == SlotSource.DIRECTORY.value


def test_a_chosen_beneficiary_is_attributed_to_the_directory(
    engine: ConversationEngine, directory_db
):
    engine.handle("send 100 SAR to Ahmed", "p5", user_id="demo")
    result = engine.handle("1", "p5", user_id="demo")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slot_provenance["recipient"] == SlotSource.DIRECTORY.value


def test_a_biller_is_attributed_to_the_catalogue(engine: ConversationEngine):
    result = engine.handle("pay my mobily bill 100 SAR", "p6")
    prov = result.state.slot_provenance
    assert result.state.slots.biller_code
    assert prov["biller"] == SlotSource.BILLER_CATALOGUE.value
    assert prov["biller_code"] == SlotSource.BILLER_CATALOGUE.value
    assert prov["amount"] == SlotSource.USER_TEXT.value


def test_provenance_survives_the_turn_boundary(engine: ConversationEngine):
    engine.handle("I want to transfer money", "p7")
    engine.handle("500 SAR", "p7")
    result = engine.handle("Ahmed", "p7")
    prov = result.state.slot_provenance
    assert prov["amount"] == SlotSource.USER_TEXT.value
    assert prov["recipient"] == SlotSource.USER_TEXT.value


def test_a_new_request_starts_with_empty_provenance(engine: ConversationEngine):
    engine.handle("send 500 SAR to Ahmed", "p8")
    engine.handle("yes", "p8")
    result = engine.handle("hi", "p8")
    assert result.state.slots.amount is None
    assert result.state.slot_provenance == {}


# ----------------------------- API serialisation --------------------------- #


def test_the_response_exposes_both_fields():
    resp = client.post(
        "/conversation/text",
        json={"text": "I want to transfer money", "session_id": "api-reason"},
    )
    body = resp.json()
    assert body["reason_code"] == ReasonCode.SLOT_REQUIRED.value
    assert body["slot_provenance"] == {}

    resp = client.post(
        "/conversation/text",
        json={"text": "500 SAR", "session_id": "api-reason"},
    )
    body = resp.json()
    assert body["slot_provenance"]["amount"] == SlotSource.USER_TEXT.value
    assert body["reason_code"] == ReasonCode.SLOT_REQUIRED.value
