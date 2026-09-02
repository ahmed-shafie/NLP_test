"""Choosing a beneficiary from the "which one?" list names that same person.

Somebody saved under a bare first name ("عمر") sits inside another candidate's
name ("ليلى عمر"), and the choice matcher used to accept either — which ended a
transfer at a confirmation naming a different person. It must now take the exact
name, keep a word that only fits somebody else's surname ambiguous, and find an
Arabic-only record when the customer types the name in English.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text

import app.banking_core_client as bcc
import app.db.directory as directory
from app.config import settings
from app.conversation.engine import ConversationEngine
from app.conversation.state import ConversationStatus

# (id, name, name_ar, account): Laila carries "عمر" as her surname, and عمر is a
# separate person saved under that name alone — exactly the reported directory.
ROWS = [
    ("C1", "Laila Omar", "ليلى عمر", "SA1122330000006464"),
    ("C2", "عمر", "عمر", "SA0380000000608010167519"),
    ("C3", "Mohammed Nour", "محمد نور", "SA1122330000001200"),
    ("C4", "Mohammed Saad", "محمد سعد", "SA1122330000001201"),
]


def _seed(db_path) -> str:
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
        for bid, name, name_ar, account in ROWS:
            conn.execute(
                text("INSERT INTO beneficiaries VALUES (:id,:o,:n,:na,:a,:b,:c,:s,:f)"),
                {
                    "id": bid,
                    "o": "demo",
                    "n": name,
                    "na": name_ar,
                    "a": account,
                    "b": "Al Rajhi",
                    "c": "SAR",
                    "s": "active",
                    "f": 0,
                },
            )
    engine.dispose()
    return url


@pytest.fixture()
def directory_db(tmp_path, monkeypatch):
    url = _seed(tmp_path / "choice.db")
    monkeypatch.setattr(settings, "beneficiary_lookup_enabled", True)
    monkeypatch.setattr(settings, "beneficiary_db_url", url)
    directory.get_beneficiary_directory.cache_clear()
    yield url
    directory.get_beneficiary_directory.cache_clear()


@pytest.fixture()
def fake_core(monkeypatch):
    monkeypatch.setattr(
        bcc,
        "get_balance",
        lambda owner_user, account=None, account_type=None: bcc.AccountInfo(
            account_id="ACC-002",
            account_type=account_type or "current",
            number="SA0380009999888877",
            currency="SAR",
            balance=Decimal("5000.00"),
            status="active",
        ),
    )
    monkeypatch.setattr(
        bcc, "preflight_transfer", lambda **k: bcc.PreflightResult(ok=True, warnings=[])
    )
    return bcc


@pytest.fixture()
def engine() -> ConversationEngine:
    return ConversationEngine()


def test_the_exact_name_wins_over_the_name_containing_it(
    engine, directory_db, fake_core
):
    engine.handle("حول 500 لعمر", "c-exact")
    result = engine.handle("عمر", "c-exact")
    assert result.state.slots.recipient == "عمر"
    assert result.state.slots.account_number == "SA0380000000608010167519"
    assert "ليلى" not in result.reply


def test_a_surname_typed_in_english_reaches_the_arabic_record(
    engine, directory_db, fake_core
):
    """ "Omar" is عمر: the directory must cross scripts before it offers names."""

    result = engine.handle("send 8 to Omar", "c-en")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    accounts = {o.account for o in result.state.beneficiary_options}
    assert "SA0380000000608010167519" in accounts
    picked = engine.handle("Omar", "c-en")
    assert picked.state.slots.account_number == "SA0380000000608010167519"


def test_a_word_that_fits_two_candidates_still_asks(engine, directory_db, fake_core):
    engine.handle("حول 500 لمحمد", "c-two")
    result = engine.handle("محمد", "c-two")
    assert result.state.status is ConversationStatus.DISAMBIGUATING
    assert result.state.slots.account_number is None


def test_the_full_name_of_the_other_person_is_still_selectable(
    engine, directory_db, fake_core
):
    engine.handle("حول 500 لعمر", "c-full")
    result = engine.handle("ليلى عمر", "c-full")
    assert result.state.slots.account_number == "SA1122330000006464"


def test_the_last_four_digits_still_select(engine, directory_db, fake_core):
    engine.handle("حول 500 لعمر", "c-four")
    result = engine.handle("7519", "c-four")
    assert result.state.slots.account_number == "SA0380000000608010167519"
