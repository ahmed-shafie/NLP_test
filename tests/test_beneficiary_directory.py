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
from app.schemas import Intent, Language

# A structurally valid Saudi IBAN (24 chars, mod-97 checks out).
VALID_IBAN = "SA0380000000608010167519"

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


def test_search_arabic_plain_alef_matches_hamza(directory_db):
    """ "احمد" (plain alef) must match the stored "أحمد" (alef-with-hamza)."""

    d = directory.get_beneficiary_directory()
    hits = d.search("احمد", "demo")
    assert hits is not None and len(hits) == 3


def test_malicious_identifier_rejected(monkeypatch, directory_db):
    """A non-identifier table/column name must be rejected, not interpolated."""

    monkeypatch.setattr(settings, "beneficiary_table", "beneficiaries; DROP TABLE x")
    directory.get_beneficiary_directory.cache_clear()
    # The factory swallows the ValueError and returns None (lookup disabled).
    assert directory.get_beneficiary_directory() is None
    with pytest.raises(ValueError):
        directory._safe_identifier("owner_user; --", "owner column")


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


def test_recipient_answer_strips_arabic_verb(engine, directory_db, fake_core):
    """A colloquial recipient answer ("ابغي احمد") resolves, verb stripped."""

    engine.handle("transfer 500 SAR", "b-verb-1")  # leaves recipient pending
    result = engine.handle("ابغي احمد", "b-verb-1")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert result.state.slots.recipient == "أحمد"
    assert len(result.state.beneficiary_options) == 3


def test_recipient_answer_english_to_phrase(engine, directory_db, fake_core):
    """ "send to Ahmed" as a slot answer keeps only the name."""

    engine.handle("transfer 500 SAR", "b-verb-2")
    result = engine.handle("send to Ahmed", "b-verb-2")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert len(result.state.beneficiary_options) == 3


def test_arabic_chat_shows_arabic_names(engine, directory_db, fake_core):
    """An Arabic conversation lists and confirms with the Arabic name."""

    result = engine.handle("حوّل ٥٠٠ ريال إلى أحمد", "b-ar-names")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert "أحمد خالد" in result.reply  # AR name, not the English "Ahmed Khaled"
    assert "Ahmed" not in result.reply
    confirm = engine.handle("2", "b-ar-names")
    assert confirm.state.status is ConversationStatus.CONFIRMING
    assert confirm.state.slots.recipient == "أحمد خالد"


def test_language_sticks_on_numeric_reply(engine, directory_db, fake_core):
    """A bare "2" keeps the conversation's Arabic language (not English)."""

    engine.handle("حوّل ٥٠٠ ريال إلى أحمد", "b-ar-stick")
    confirm = engine.handle("2", "b-ar-stick")
    assert confirm.state.language is Language.AR
    assert "تأكيد" in confirm.reply  # Arabic confirmation prompt


def test_disambiguation_by_arabic_full_name(engine, directory_db, fake_core):
    """Selecting by the Arabic full name (plain alef) resolves the beneficiary."""

    engine.handle("حوّل ٥٠٠ ريال إلى أحمد", "b-ar-full")
    result = engine.handle("احمد خالد", "b-ar-full")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.recipient == "أحمد خالد"


# ----------------------------- add beneficiary ----------------------------- #


def test_not_found_offers_add_then_adds(engine, directory_db, fake_core):
    result = engine.handle("send 500 SAR to Zeyad", "b-add-1")
    assert result.state.status is ConversationStatus.COLLECTING
    assert result.state.pending_add_name == "Zeyad"
    assert "add" in result.reply.lower()

    # The account is held pending: one confirmation covers the add + transfer.
    quoted = engine.handle(VALID_IBAN, "b-add-1")
    assert quoted.state.status is ConversationStatus.CONFIRMING
    assert quoted.state.pending_add_account == VALID_IBAN
    assert quoted.state.slots.account_number is None  # not saved yet

    added = engine.handle("yes", "b-add-1")
    assert added.state.slots.recipient == "Zeyad"
    assert added.state.slots.account_number == VALID_IBAN
    assert added.transfer is not None


def test_not_found_decline_add(engine, directory_db, fake_core):
    engine.handle("send 500 SAR to Zeyad", "b-add-2")
    result = engine.handle("no", "b-add-2")
    assert result.state.pending_add_name is None
    assert result.state.pending_slot == "recipient"


def test_add_invalid_account_message(engine, directory_db, fake_core):
    """A non-account reply during the add flow explains the expected format."""

    engine.handle("send 500 SAR to Zeyad", "b-add-3")
    result = engine.handle("abcd", "b-add-3")
    assert result.state.pending_add_name == "Zeyad"  # still awaiting the account
    assert "valid iban" in result.reply.lower()


def test_add_failed_surfaces_api_reason(engine, directory_db, monkeypatch, fake_core):
    """A specific failure message from the banking service is shown to the user."""

    monkeypatch.setattr(
        bcc,
        "add_beneficiary",
        lambda **k: {
            "ok": False,
            "message": "A beneficiary with that account already exists.",
        },
    )
    engine.handle("send 500 SAR to Zeyad", "b-add-4")
    engine.handle(VALID_IBAN, "b-add-4")
    result = engine.handle("yes", "b-add-4")
    assert "already exists" in result.reply
    assert result.state.pending_add_name is None


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


def test_balance_aside_during_disambiguation(engine, directory_db, fake_core):
    """A balance question mid-disambiguation is answered, then the flow resumes."""

    engine.handle("send 9000 SAR to Ahmed", "b-aside-1")
    result = engine.handle("what is my savings balance", "b-aside-1")
    # Balance answered inline, but the transaction is untouched.
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert result.state.disambiguation_kind == "beneficiary"
    assert len(result.state.beneficiary_options) == 3
    assert "5000" in result.reply
    assert "which one" in result.reply.lower()
    # The flow still resolves normally afterwards.
    picked = engine.handle("2", "b-aside-1")
    assert picked.state.status is ConversationStatus.CONFIRMING
    assert picked.state.slots.recipient == "Ahmed Khaled"


def test_balance_aside_during_confirmation(engine, directory_db, fake_core):
    """A balance question at the confirm step keeps the transfer confirmable."""

    engine.handle("send 500 SAR to Mona", "b-aside-2")
    result = engine.handle("what is my current balance", "b-aside-2")
    assert result.state.status is ConversationStatus.CONFIRMING
    assert result.state.slots.recipient == "Mona Ali"
    assert "5000" in result.reply
    done = engine.handle("yes", "b-aside-2")
    assert done.state.status is ConversationStatus.COMPLETED


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


# ----------------------- list beneficiaries (read-only) -------------------- #


def test_list_all_returns_favorites_first(directory_db):
    d = directory.get_beneficiary_directory()
    hits = d.list_all("demo")
    assert hits is not None and len(hits) == 4
    # Ahmed Hassan is the only favorite (is_favorite=1) -> sorted first.
    assert hits[0].name == "Ahmed Hassan"


def test_list_all_unknown_owner_is_empty(directory_db):
    d = directory.get_beneficiary_directory()
    assert d.list_all("nobody") == []


def test_list_beneficiaries_arabic(engine, directory_db):
    result = engine.handle("من المستفيدين عندي", "b-list-ar")
    assert result.state.intent is Intent.LIST_BENEFICIARIES
    assert result.state.status is ConversationStatus.COMPLETED
    # Arabic names shown, and it never asks for a transfer amount.
    assert "أحمد حسن" in result.reply
    assert "كم المبلغ" not in result.reply


def test_list_beneficiaries_arabic_misspelled(engine, directory_db):
    """The common misspelling المستف[ي]دين still lists, never starts a transfer."""

    result = engine.handle("من المستفدين عندي", "b-list-ar-typo")
    assert result.state.intent is Intent.LIST_BENEFICIARIES
    assert result.state.status is ConversationStatus.COMPLETED
    assert "أحمد" in result.reply


@pytest.mark.parametrize(
    "phrase",
    [
        "list my beneficiary",  # singular
        "list my benificary",  # misspelled
        "beneficiaries",  # the bare noun
        "who are my beneficiaries",
        "show me the list of my payees",
        "check my beneficiaries",
        "المستفدين",  # bare, misspelled
        "قائمة المستفيدين",
        "وش المستفيدين اللي عندي",
        "ابغى اشوف المستفيدين",  # colloquial "let me see"
        "شوف المستفيدين",
    ],
)
def test_list_phrasings_all_list(engine, directory_db, phrase):
    """Every way of asking to see beneficiaries lists them, read-only."""

    result = engine.handle(phrase, f"b-list-{abs(hash(phrase))}")
    assert result.state.intent is Intent.LIST_BENEFICIARIES
    assert result.state.status is ConversationStatus.COMPLETED
    assert result.state.pending_add_name is None
    assert result.transfer is None


def test_list_beneficiaries_english(engine, directory_db):
    result = engine.handle("show my beneficiaries", "b-list-en")
    assert result.state.intent is Intent.LIST_BENEFICIARIES
    assert "Ahmed Hassan" in result.reply
    assert "Mona Ali" in result.reply


def test_list_beneficiaries_does_not_start_transfer(engine, directory_db):
    result = engine.handle("عرض المستفيدين لدي", "b-list-notransfer")
    assert result.state.status is not ConversationStatus.COLLECTING
    assert result.state.pending_slot is None
    assert result.state.slots.amount is None
    assert result.state.slots.recipient is None


def test_list_beneficiaries_masks_account(engine, directory_db):
    result = engine.handle("list my beneficiaries", "b-list-mask")
    assert "SA••7777" in result.reply
    assert "SA1122330000007777" not in result.reply


def test_list_beneficiaries_empty(engine, directory_db):
    result = engine.handle("show my beneficiaries", "b-list-empty", user_id="nobody")
    assert result.state.intent is Intent.LIST_BENEFICIARIES
    assert "don't have any saved beneficiaries" in result.reply.lower()


def test_list_beneficiaries_unavailable(engine, directory_db, monkeypatch):
    monkeypatch.setattr(settings, "beneficiary_lookup_enabled", False)
    directory.get_beneficiary_directory.cache_clear()
    result = engine.handle("show my beneficiaries", "b-list-unavail")
    assert result.state.intent is Intent.LIST_BENEFICIARIES
    assert "couldn't fetch your beneficiaries" in result.reply.lower()
