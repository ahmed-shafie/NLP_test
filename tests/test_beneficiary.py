"""Tests for the configurable, SQLAlchemy-backed beneficiary repository.

These use SQLite (a SQLAlchemy provider like any other) to prove the lookup, the
column mapping, and graceful degradation work without code changes per provider.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text

from app.db.beneficiary import BeneficiaryRepository


def _seed(db_path: Path, ddl: str, rows: list[dict]) -> str:
    url = f"sqlite:///{db_path}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(ddl))
        for row in rows:
            cols = ", ".join(row)
            params = ", ".join(f":{c}" for c in row)
            conn.execute(
                text(f"INSERT INTO beneficiaries ({cols}) VALUES ({params})"), row
            )
    engine.dispose()
    return url


def test_lookup_returns_beneficiary(tmp_path):
    url = _seed(
        tmp_path / "bank.db",
        "CREATE TABLE beneficiaries (id TEXT, name TEXT, account TEXT, bank TEXT)",
        [{"id": "b1", "name": "Sara Adel", "account": "EG1003", "bank": "CIB"}],
    )
    repo = BeneficiaryRepository(
        url=url,
        query=(
            "SELECT id, name, account, bank FROM beneficiaries "
            "WHERE account = :account_number"
        ),
        account_param="account_number",
    )

    ben = repo.lookup("EG1003")

    assert ben is not None
    assert ben.name == "Sara Adel"
    assert ben.account == "EG1003"
    assert ben.bank == "CIB"


def test_lookup_missing_returns_none(tmp_path):
    url = _seed(
        tmp_path / "bank.db",
        "CREATE TABLE beneficiaries (id TEXT, name TEXT, account TEXT, bank TEXT)",
        [{"id": "b1", "name": "Sara Adel", "account": "EG1003", "bank": "CIB"}],
    )
    repo = BeneficiaryRepository(
        url=url,
        query=(
            "SELECT id, name, account, bank FROM beneficiaries "
            "WHERE account = :account_number"
        ),
        account_param="account_number",
    )

    assert repo.lookup("EG9999") is None


def test_column_map_translates_columns(tmp_path):
    # A schema whose column names differ from Beneficiary fields.
    url = _seed(
        tmp_path / "bank.db",
        "CREATE TABLE beneficiaries "
        "(ben_id TEXT, full_name TEXT, acct_no TEXT, bank_name TEXT)",
        [
            {
                "ben_id": "x9",
                "full_name": "محمد علي",
                "acct_no": "EG1002",
                "bank_name": "NBE",
            }
        ],
    )
    repo = BeneficiaryRepository(
        url=url,
        query=(
            "SELECT ben_id, full_name, acct_no, bank_name FROM beneficiaries "
            "WHERE acct_no = :account_number"
        ),
        account_param="account_number",
        column_map={
            "id": "ben_id",
            "name": "full_name",
            "account": "acct_no",
            "bank": "bank_name",
        },
    )

    ben = repo.lookup("EG1002")

    assert ben is not None
    assert ben.id == "x9"
    assert ben.name == "محمد علي"
    assert ben.account == "EG1002"
    assert ben.bank == "NBE"


def test_bad_query_degrades_to_none(tmp_path):
    url = _seed(
        tmp_path / "bank.db",
        "CREATE TABLE beneficiaries (id TEXT, name TEXT, account TEXT, bank TEXT)",
        [{"id": "b1", "name": "Sara Adel", "account": "EG1003", "bank": "CIB"}],
    )
    repo = BeneficiaryRepository(
        url=url,
        query="SELECT * FROM nonexistent_table WHERE account = :account_number",
        account_param="account_number",
    )

    # A failing query must not raise; it degrades to None so the LLM can take over.
    assert repo.lookup("EG1003") is None
