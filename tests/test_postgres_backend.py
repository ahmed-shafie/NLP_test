"""Banking Core + beneficiary directory against a real Postgres database.

The ORM is provider-agnostic, but raw SQL and column types are not: this suite
caught ``COALESCE(is_favorite, 0)`` failing on Postgres with "types boolean and
integer cannot be matched" while passing happily on SQLite.

Skipped unless a Postgres URL is provided, so the default test run stays offline::

    export BANKING_CORE_TEST_DB_URL=\
        postgresql+psycopg://banking:banking@localhost:5432/banking_core
    pytest tests/test_postgres_backend.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

import app.db.directory as directory

_BANKING_CORE = Path(__file__).resolve().parents[1] / "banking-core"
if str(_BANKING_CORE) not in sys.path:
    sys.path.insert(0, str(_BANKING_CORE))

PG_URL = os.environ.get("BANKING_CORE_TEST_DB_URL", "")

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="Set BANKING_CORE_TEST_DB_URL to run the Postgres suite."
)


@pytest.fixture()
def pg_schema():
    """A throwaway schema so the suite never touches real rows."""

    schema = f"t_{uuid.uuid4().hex[:8]}"
    engine = create_engine(PG_URL)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
        conn.execute(
            text(
                f"CREATE TABLE {schema}.beneficiaries ("
                "id VARCHAR PRIMARY KEY, owner_user VARCHAR, name VARCHAR, "
                "name_ar VARCHAR, account VARCHAR, bank VARCHAR, "
                "currency VARCHAR, status VARCHAR, is_favorite BOOLEAN)"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {schema}.beneficiaries VALUES "
                "('b1','demo','Ahmed Hassan','أحمد حسن','SA11','Al Rajhi',"
                "'SAR','active',false),"
                "('b2','demo','Ahmed Khaled','أحمد خالد','SA22','SNB',"
                "'SAR','active',true),"
                "('b3','demo','Mona Ali',NULL,'SA33',NULL,'SAR','active',NULL)"
            )
        )
    yield schema
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    engine.dispose()


def test_list_all_reads_postgres(pg_schema):
    """A boolean is_favorite sorts favorites first without a type mismatch."""

    d = directory.BeneficiaryDirectory(
        PG_URL, f"{pg_schema}.beneficiaries", "owner_user"
    )
    hits = d.list_all("demo")
    assert hits is not None
    assert [h.name for h in hits] == ["Ahmed Khaled", "Ahmed Hassan", "Mona Ali"]
    assert hits[0].is_favorite is True
    # A NULL is_favorite must read as False, not blow up.
    assert hits[2].is_favorite is False


def test_search_matches_arabic_on_postgres(pg_schema):
    """Normalized Arabic matching works the same over Postgres rows."""

    d = directory.BeneficiaryDirectory(
        PG_URL, f"{pg_schema}.beneficiaries", "owner_user"
    )
    assert {h.name for h in d.search("احمد", "demo") or []} == {
        "Ahmed Hassan",
        "Ahmed Khaled",
    }
    assert d.search("Mona", "demo") == d.search("mona", "demo")


def test_banking_core_round_trip_on_postgres(monkeypatch):
    """Schema creation, seeding and a beneficiary write all run on Postgres."""

    schema = f"t_{uuid.uuid4().hex[:8]}"
    engine = create_engine(PG_URL)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    url = f"{PG_URL}?options=-csearch_path%3D{schema}"
    monkeypatch.setenv("BANKING_CORE_DB_URL", url)

    from banking_core import config as bc_config

    monkeypatch.setattr(bc_config.settings, "db_url", url)
    from banking_core import db as bc_db
    from banking_core import seed as bc_seed
    from banking_core import service as bc_service
    from banking_core.schemas import AddBeneficiaryRequest

    bc_db.get_engine.cache_clear()
    bc_db.get_sessionmaker.cache_clear()
    try:
        assert bc_seed.is_empty() is True
        assert bc_seed.seed(reset=False) is True
        # Seeding an already-populated database is a no-op, never a wipe.
        assert bc_seed.seed(reset=False) is False

        result = bc_service.add_beneficiary(
            AddBeneficiaryRequest(
                owner_user="demo",
                name="Noura Saad",
                account="SA0244200000012345678912",
                currency="SAR",
            )
        )
        assert result.ok is True
        with engine.begin() as conn:
            count = conn.execute(
                text(f"SELECT count(*) FROM {schema}.beneficiaries")
            ).scalar()
        assert count == 10
    finally:
        bc_db.get_engine.cache_clear()
        bc_db.get_sessionmaker.cache_clear()
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        engine.dispose()
