"""Tests for the standalone Banking Core service (balance / pre-flight / add)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_BANKING_CORE = Path(__file__).resolve().parents[1] / "banking-core"
if str(_BANKING_CORE) not in sys.path:
    sys.path.insert(0, str(_BANKING_CORE))

from banking_core import db as bc_db  # noqa: E402
from banking_core import seed as bc_seed  # noqa: E402
from banking_core.config import settings as bc_settings  # noqa: E402
from banking_core.main import app as bc_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'bc.db'}"
    monkeypatch.setattr(bc_settings, "db_url", url)
    bc_db.get_engine.cache_clear()
    bc_db.get_sessionmaker.cache_clear()
    bc_seed.seed()
    with TestClient(bc_app) as test_client:
        yield test_client
    bc_db.get_engine.cache_clear()
    bc_db.get_sessionmaker.cache_clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_balance_by_type(client):
    resp = client.post(
        "/accounts/balance", json={"owner_user": "demo", "account_type": "savings"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["account_type"] == "savings"
    assert body["balance"] == "5000.00"
    assert body["currency"] == "SAR"


def test_balance_missing_account(client):
    resp = client.post(
        "/accounts/balance", json={"owner_user": "nobody", "account_type": "savings"}
    )
    assert resp.status_code == 404


def test_preflight_insufficient_funds_blocks_and_reports_the_balance(client):
    resp = client.post(
        "/preflight/transfer",
        json={
            "owner_user": "demo",
            "amount": "9000",
            "currency": "SAR",
            "source_account_type": "savings",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    # The spendable balance travels with the refusal so the assistant can offer
    # it instead of inviting a confirmation the account cannot fund.
    assert body["blocking"] == ["insufficient_funds: available 5000.00 SAR"]


def test_preflight_fx_note_not_blocks(client):
    resp = client.post(
        "/preflight/transfer",
        json={
            "owner_user": "demo",
            "amount": "100",
            "currency": "USD",
            "source_account_type": "savings",
        },
    )
    body = resp.json()
    assert body["ok"] is True
    assert any(w.startswith("fx") for w in body["warnings"])


def test_preflight_unknown_source_blocks(client):
    resp = client.post(
        "/preflight/transfer",
        json={
            "owner_user": "demo",
            "amount": "10",
            "currency": "SAR",
            "source_account": "SA-does-not-exist",
        },
    )
    body = resp.json()
    assert body["ok"] is False
    assert "source_account_not_found" in body["blocking"]


def test_add_beneficiary_and_duplicate(client):
    payload = {
        "owner_user": "demo",
        "name": "New Person",
        "account": "SA9998887",
        "bank": "Test Bank",
        "currency": "SAR",
    }
    first = client.post("/beneficiary/add", json=payload)
    assert first.status_code == 200
    assert first.json()["ok"] is True
    assert first.json()["beneficiary"]["name"] == "New Person"

    dup = client.post("/beneficiary/add", json=payload)
    assert dup.json()["ok"] is False


def test_accounts_list_returns_the_customers_active_accounts(client):
    resp = client.post("/accounts/list", json={"owner_user": "demo"})
    assert resp.status_code == 200
    accounts = resp.json()["accounts"]
    assert [a["account_id"] for a in accounts] == [
        "ACC-001",
        "ACC-002",
        "ACC-003",
        "ACC-004",
    ]
    # The current account leads the list, because the assistant numbers the rows
    # in this order and a saved "1" must mean the same account next turn.
    assert accounts[0]["account_type"] == "current"
    assert accounts[0]["balance"] == "12300.00"


def test_accounts_list_of_an_unknown_customer_is_empty(client):
    resp = client.post("/accounts/list", json={"owner_user": "nobody"})
    assert resp.status_code == 200
    assert resp.json()["accounts"] == []


def test_accounts_list_requires_the_api_key(client, monkeypatch):
    monkeypatch.setattr(bc_settings, "api_key", "s3cret")

    refused = client.post("/accounts/list", json={"owner_user": "demo"})
    assert refused.status_code == 401

    allowed = client.post(
        "/accounts/list",
        json={"owner_user": "demo"},
        headers={"x-api-key": "s3cret"},
    )
    assert allowed.status_code == 200
