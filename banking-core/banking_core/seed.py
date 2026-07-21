"""Seed the Banking Core database with demo accounts, beneficiaries, and billers.

Idempotent: running it re-creates the tables and repopulates the demo rows. The
beneficiary set deliberately contains several people who share a first name
("Ahmed", "Mohammed") so the assistant must disambiguate.
"""

from __future__ import annotations

from decimal import Decimal

from banking_core.db import (
    Account,
    Base,
    Beneficiary,
    Biller,
    get_engine,
    init_db,
    session_scope,
)

DEMO_USER = "demo"

ACCOUNTS: list[dict[str, object]] = [
    dict(
        account_id="ACC-001",
        owner_user=DEMO_USER,
        account_type="current",
        number="SA0380001234567890",
        currency="SAR",
        balance=Decimal("12300.00"),
        status="active",
    ),
    dict(
        account_id="ACC-002",
        owner_user=DEMO_USER,
        account_type="savings",
        number="SA0380009999888877",
        currency="SAR",
        balance=Decimal("5000.00"),
        status="active",
    ),
    dict(
        account_id="ACC-003",
        owner_user=DEMO_USER,
        account_type="credit",
        number="SA0380004444555566",
        currency="USD",
        balance=Decimal("800.00"),
        status="active",
    ),
    dict(
        account_id="ACC-004",
        owner_user=DEMO_USER,
        account_type="salary",
        number="SA0380007777000011",
        currency="SAR",
        balance=Decimal("240.00"),
        status="active",
    ),
]

BENEFICIARIES: list[dict[str, object]] = [
    # Three people named "Ahmed" -> forces disambiguation.
    dict(
        id="B1",
        owner_user=DEMO_USER,
        name="Ahmed Hassan",
        name_ar="أحمد حسن",
        account="SA1122330000007777",
        bank="Al Rajhi",
        currency="SAR",
        is_favorite=True,
    ),
    dict(
        id="B2",
        owner_user=DEMO_USER,
        name="Ahmed Khaled",
        name_ar="أحمد خالد",
        account="SA1122330000002211",
        bank="SNB",
        currency="SAR",
    ),
    dict(
        id="B3",
        owner_user=DEMO_USER,
        name="Ahmed Mahmoud",
        name_ar="أحمد محمود",
        account="SA1122330000008090",
        bank="Riyad Bank",
        currency="USD",
    ),
    # Two people named "Mohammed".
    dict(
        id="B4",
        owner_user=DEMO_USER,
        name="Mohammed Nour",
        name_ar="محمد نور",
        account="SA1122330000001200",
        bank="Al Rajhi",
        currency="SAR",
    ),
    dict(
        id="B5",
        owner_user=DEMO_USER,
        name="Mohammed Saad",
        name_ar="محمد سعد",
        account="SA1122330000001201",
        bank="Alinma",
        currency="SAR",
    ),
    # Unique first names.
    dict(
        id="B6",
        owner_user=DEMO_USER,
        name="Mona Ali",
        name_ar="منى علي",
        account="SA1122330000003333",
        bank="SNB",
        currency="SAR",
    ),
    dict(
        id="B7",
        owner_user=DEMO_USER,
        name="Sara Adel",
        name_ar="سارة عادل",
        account="SA1122330000005555",
        bank="SNB",
        currency="SAR",
    ),
    dict(
        id="B8",
        owner_user=DEMO_USER,
        name="Laila Omar",
        name_ar="ليلى عمر",
        account="SA1122330000006464",
        bank="Al Rajhi",
        currency="SAR",
    ),
    dict(
        id="B9",
        owner_user=DEMO_USER,
        name="Khalid Fahad",
        name_ar="خالد فهد",
        account="SA1122330000007878",
        bank="Riyad Bank",
        currency="SAR",
    ),
]

BILLERS: list[dict[str, object]] = [
    dict(biller_code="001", name="STC", category="telecom"),
    dict(biller_code="002", name="Mobily", category="telecom"),
    dict(biller_code="003", name="Zain", category="telecom"),
    dict(biller_code="004", name="Saudi Electricity Company", category="electricity"),
    dict(biller_code="005", name="National Water Company", category="water"),
    dict(biller_code="006", name="Ejar", category="rent"),
]


def seed() -> None:
    """(Re)create the schema and load the demo rows (fresh ORM instances)."""

    Base.metadata.drop_all(get_engine())
    init_db()
    with session_scope() as session:
        session.add_all(Account(**row) for row in ACCOUNTS)
        session.add_all(Beneficiary(**row) for row in BENEFICIARIES)
        session.add_all(Biller(**row) for row in BILLERS)


if __name__ == "__main__":
    seed()
    print("Seeded banking_core database.")
