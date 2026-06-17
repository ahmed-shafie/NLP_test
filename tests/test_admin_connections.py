"""Tests for the external-resource connection store and its activation flow."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

import app.admin.store as store
import app.db.beneficiary as beneficiary
from app.admin import connections
from app.admin.schemas import ConnectionCreate, ConnectionUpdate
from app.config import settings


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    """Point the admin store at a throwaway SQLite file for each test."""

    monkeypatch.setattr(settings, "admin_store_url", f"sqlite:///{tmp_path/'cfg.db'}")
    store.get_engine.cache_clear()
    store.get_sessionmaker.cache_clear()
    beneficiary.get_beneficiary_repository.cache_clear()
    yield
    store.get_engine.cache_clear()
    store.get_sessionmaker.cache_clear()
    beneficiary.get_beneficiary_repository.cache_clear()


def _seed_bank(tmp_path) -> str:
    url = f"sqlite:///{tmp_path/'bank.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE beneficiaries (id TEXT, name TEXT, account TEXT, bank TEXT)")
        )
        conn.execute(
            text("INSERT INTO beneficiaries VALUES ('b1','Sara Adel','EG1003','CIB')")
        )
    engine.dispose()
    return url


def test_create_list_and_get(isolated_store, tmp_path):
    created = connections.create_connection(
        ConnectionCreate(name="pg", provider="postgresql", url="sqlite:///x.db")
    )
    assert created.id is not None
    assert connections.get_connection(created.id).name == "pg"
    assert len(connections.list_connections()) == 1


def test_update_and_delete(isolated_store):
    created = connections.create_connection(
        ConnectionCreate(name="c", provider="sqlite", url="sqlite:///x.db")
    )
    updated = connections.update_connection(
        created.id, ConnectionUpdate(name="renamed", column_map={"name": "full_name"})
    )
    assert updated is not None
    assert updated.name == "renamed"
    assert updated.column_map == {"name": "full_name"}
    assert connections.delete_connection(created.id) is True
    assert connections.get_connection(created.id) is None


def test_activate_is_exclusive(isolated_store):
    a = connections.create_connection(
        ConnectionCreate(name="a", provider="sqlite", url="sqlite:///a.db")
    )
    b = connections.create_connection(
        ConnectionCreate(name="b", provider="sqlite", url="sqlite:///b.db")
    )
    connections.activate_connection(a.id)
    assert connections.get_active_connection().id == a.id
    connections.activate_connection(b.id)
    active = connections.get_active_connection()
    assert active.id == b.id
    # Only one connection may be active at a time.
    assert sum(1 for c in connections.list_connections() if c.is_active) == 1


def test_test_connection_success_and_failure(isolated_store, tmp_path):
    url = _seed_bank(tmp_path)
    ok = connections.test_connection(
        url,
        "SELECT id, name, account, bank FROM beneficiaries WHERE account = :account_number",
        "account_number",
        sample_account="EG1003",
    )
    assert ok.ok is True
    assert "name" in ok.sample_columns

    bad = connections.test_connection(
        url, "SELECT * FROM missing WHERE x = :p", "p", sample_account="1"
    )
    assert bad.ok is False


def test_active_connection_drives_beneficiary_repository(isolated_store, tmp_path):
    url = _seed_bank(tmp_path)
    conn = connections.create_connection(
        ConnectionCreate(
            name="bank",
            provider="sqlite",
            url=url,
            query=(
                "SELECT id, name, account, bank FROM beneficiaries "
                "WHERE account = :account_number"
            ),
        )
    )
    connections.activate_connection(conn.id)
    repo = beneficiary.get_beneficiary_repository()
    assert repo is not None
    found = repo.lookup("EG1003")
    assert found is not None
    assert found.name == "Sara Adel"
    assert repo.lookup("EG9999") is None
