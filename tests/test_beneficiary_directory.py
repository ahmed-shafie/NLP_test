"""Direct-DB beneficiary lookup + disambiguation, balance inquiry, add-flow, FX/funds.

The transfer beneficiary check reads the database directly (not the API); a shared
first name triggers a "which one?" disambiguation. Balance/pre-flight/add-beneficiary
go through the Banking Core client, which is faked here to stay offline.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

import app.banking_core_client as bcc
import app.db.directory as directory
from app.config import settings
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus
from app.schemas import Intent

_BANKING_CORE = Path(__file__).resolve().parents[1] / "banking-core"
if str(_BANKING_CORE) not in sys.path:
    sys.path.insert(0, str(_BANKING_CORE))


def _seed_directory(db_path: Path) -> str:
    """Create a beneficiaries table with several shared first names."""

    url = f"sqlite:///{db_path}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE beneficiaries (id TEXT, owner_user TEXT, name TEXT, "
                "name_ar TEXT, account TEXT, bank TEXT, currency TEXT, "
                "status TEXT, is_favorite INTEGER)"
            )
        )
        rows = [
            (
                "B1",
                "demo",
                "Ahmed Hassan",
                "أحمد حسن",
                "SA1122330000007777",
                "Al Rajhi",
                "SAR",
                "active",
                1,
            ),
            (
                "B2",
                "demo",
                "Ahmed Khaled",
                "أحمد خالد",
                "SA1122330000002211",
                "SNB",
                "SAR",
                "active",
                0,
            ),
            (
                "B3",
                "demo",
                "Ahmed Mahmoud",
                "أحمد محمود",
                "SA1122330000008090",
                "Riyad Bank",
                "USD",
                "active",
                0,
            ),
            (
                "B6",
                "demo",
                "Mona Ali",
                "منى علي",
                "SA1122330000003333",
                "SNB",
                "SAR",
                "active",
                0,
            ),
        ]
        for r in rows:
            conn.execute(
                text("INSERT INTO beneficiaries VALUES (:id,:o,:n,:na,:a,:b,:c,:s,:f)"),
                dict(zip("id o n na a b c s f".split(), r, strict=True)),
            )
    engine.dispose()
    return url


@pytest.fixture()
def directory_db(tmp_path, monkeypatch):
    url = _seed_directory(tmp_path / "dir.db")
    monkeypatch.setattr(settings, "beneficiary_lookup_enabled", True)
    monkeypatch.setattr(settings, "beneficiary_db_url", url)
    directory.get_beneficiary_directory.cache_clear()
    yield url
    directory.get_beneficiary_directory.cache_clear()


@pytest.fixture()
def fake_core(monkeypatch):
    """Stub the Banking Core client so no HTTP server is needed."""

    def _balance(owner_user, account=None, account_type=None):
        return bcc.AccountInfo(
            account_id="ACC-002",
            account_type=account_type or "current",
            number="SA0380009999888877",
            currency="SAR",
            balance=Decimal("5000.00"),
            status="active",
        )

    monkeypatch.setattr(bcc, "get_balance", _balance)
    monkeypatch.setattr(
        bcc,
        "preflight_transfer",
        lambda **k: bcc.PreflightResult(ok=True, warnings=[]),
    )
    monkeypatch.setattr(
        bcc, "add_beneficiary", lambda **k: {"ok": True, "beneficiary": {"id": "z1"}}
    )
    return bcc


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


# ----------------------------- directory search ---------------------------- #


def test_search_many_matches(directory_db):
    d = directory.get_beneficiary_directory()
    hits = d.search("Ahmed", "demo")
    assert hits is not None and len(hits) == 3


def test_search_single_match(directory_db):
    d = directory.get_beneficiary_directory()
    hits = d.search("Mona", "demo")
    assert hits is not None and len(hits) == 1
    assert hits[0].name == "Mona Ali"


def test_search_no_match_returns_empty(directory_db):
    d = directory.get_beneficiary_directory()
    assert d.search("Zzz", "demo") == []


def test_search_arabic(directory_db):
    d = directory.get_beneficiary_directory()
    hits = d.search("أحمد", "demo")
    assert hits is not None and len(hits) == 3


# ------------------------- engine disambiguation --------------------------- #


def test_shared_first_name_triggers_disambiguation(engine, directory_db, fake_core):
    result = engine.handle("send 500 SAR to Ahmed", "b-dis-1")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert result.state.disambiguation_kind == "beneficiary"
    assert len(result.state.beneficiary_options) == 3
    assert "which one" in result.reply.lower()


def test_disambiguation_by_number(engine, directory_db, fake_core):
    engine.handle("send 500 SAR to Ahmed", "b-dis-2")
    result = engine.handle("2", "b-dis-2")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.recipient == "Ahmed Khaled"


def test_disambiguation_by_full_name(engine, directory_db, fake_core):
    engine.handle("send 500 SAR to Ahmed", "b-dis-3")
    result = engine.handle("Ahmed Mahmoud", "b-dis-3")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.recipient == "Ahmed Mahmoud"


def test_disambiguation_by_last_four(engine, directory_db, fake_core):
    engine.handle("send 500 SAR to Ahmed", "b-dis-4")
    result = engine.handle("2211", "b-dis-4")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.recipient == "Ahmed Khaled"


def test_single_match_locks_and_confirms(engine, directory_db, fake_core):
    result = engine.handle("send 500 SAR to Mona", "b-one-1")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.recipient == "Mona Ali"


# ----------------------------- add beneficiary ----------------------------- #


def test_not_found_offers_add_then_adds(engine, directory_db, fake_core):
    result = engine.handle("send 500 SAR to Zeyad", "b-add-1")
    assert result.state.status is ConversationStatus.COLLECTING
    assert result.state.pending_add_name == "Zeyad"
    assert "add" in result.reply.lower()

    added = engine.handle("SA555444333", "b-add-1")
    assert added.state.status is ConversationStatus.CONFIRMING
    assert added.state.slots.recipient == "Zeyad"
    assert added.state.slots.account_number == "SA555444333"


def test_not_found_decline_add(engine, directory_db, fake_core):
    engine.handle("send 500 SAR to Zeyad", "b-add-2")
    result = engine.handle("no", "b-add-2")
    assert result.state.pending_add_name is None
    assert result.state.pending_slot == "recipient"


# ------------------------------ balance / FX ------------------------------- #


def test_balance_inquiry_uses_api(engine, fake_core):
    result = engine.handle("what is my savings balance", "b-bal-1")
    assert result.state.intent is Intent.BALANCE_INQUIRY
    assert "5000" in result.reply
    assert result.state.status is ConversationStatus.COMPLETED


def test_balance_inquiry_arabic(engine, fake_core):
    result = engine.handle("كم رصيدي", "b-bal-2")
    assert result.state.intent is Intent.BALANCE_INQUIRY
    assert "5000" in result.reply


def test_low_funds_warns_without_blocking(engine, directory_db, monkeypatch):
    monkeypatch.setattr(
        bcc,
        "preflight_transfer",
        lambda **k: bcc.PreflightResult(
            ok=True, warnings=["low_funds: short 4000.00 SAR"]
        ),
    )
    engine.handle("send 9000 SAR to Mona", "b-fx-1")
    result = engine.handle("send 9000 SAR to Mona", "b-fx-1")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert any("low_funds" in w for w in result.state.preflight_warnings)
    # Confirmation still proceeds (warning never blocks).
    done = engine.handle("yes", "b-fx-1")
    assert done.state.status is ConversationStatus.COMPLETED


def test_fx_note_without_blocking(engine, directory_db, monkeypatch):
    monkeypatch.setattr(
        bcc,
        "preflight_transfer",
        lambda **k: bcc.PreflightResult(ok=True, warnings=["fx: SAR->USD"]),
    )
    result = engine.handle("send 100 USD to Mona", "b-fx-2")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert any("fx" in w for w in result.state.preflight_warnings)
