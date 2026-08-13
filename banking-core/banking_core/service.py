"""Business logic: account resolution, balance, pre-flight checks, add beneficiary."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from banking_core.db import Account, Beneficiary, session_scope
from banking_core.schemas import (
    AccountOut,
    AddBeneficiaryRequest,
    AddBeneficiaryResult,
    BeneficiaryOut,
    PreflightBillRequest,
    PreflightResult,
    PreflightTransferRequest,
)


def _to_account_out(row: Account) -> AccountOut:
    return AccountOut(
        account_id=row.account_id,
        account_type=row.account_type,
        number=row.number,
        currency=row.currency,
        balance=row.balance,
        status=row.status,
    )


def _resolve_account(
    session: Session,
    owner_user: str,
    account: str | None,
    account_type: str | None,
) -> Account | None:
    """Find one account by explicit id/number, else by type, else the default one."""

    stmt = select(Account).where(Account.owner_user == owner_user)
    if account:
        row = session.scalars(
            stmt.where(
                (Account.account_id == account)
                | (Account.number == account)
                | (Account.number.like(f"%{account}"))
            )
        ).first()
        if row is not None:
            return row
    if account_type:
        row = session.scalars(
            stmt.where(func.lower(Account.account_type) == account_type.lower())
        ).first()
        if row is not None:
            return row
    if account or account_type:
        # An explicit hint was given but nothing matched.
        return None
    # No hint: fall back to the first active account (current preferred).
    rows = session.scalars(stmt.where(Account.status == "active")).all()
    if not rows:
        return None
    current = next((r for r in rows if r.account_type == "current"), None)
    return current or rows[0]


def get_balance(
    owner_user: str, account: str | None, account_type: str | None
) -> AccountOut | None:
    with session_scope() as session:
        row = _resolve_account(session, owner_user, account, account_type)
        return _to_account_out(row) if row is not None else None


def _funds_and_fx(
    account: Account, amount: Decimal, currency: str
) -> tuple[list[str], list[str]]:
    """Return (warnings, blocking) for a debit of ``amount`` in ``currency``.

    A debit larger than the balance is refused, not merely flagged: the reply
    carries the spendable balance so the assistant can offer it instead of
    inviting the customer to confirm a transfer the account cannot fund. A
    currency mismatch stays advisory (an FX conversion is noted).
    """

    warnings: list[str] = []
    blocking: list[str] = []
    if account.status != "active":
        blocking.append(f"source_account_inactive: {account.account_id}")
    if account.currency.upper() != currency.upper():
        warnings.append(f"fx: {account.currency}->{currency} conversion applies")
    elif account.balance < amount:
        available = account.balance.quantize(Decimal("0.01"))
        blocking.append(f"insufficient_funds: available {available} {currency}")
    return warnings, blocking


def preflight_transfer(req: PreflightTransferRequest) -> PreflightResult:
    with session_scope() as session:
        account = _resolve_account(
            session, req.owner_user, req.source_account, req.source_account_type
        )
        if account is None:
            return PreflightResult(ok=False, blocking=["source_account_not_found"])
        warnings, blocking = _funds_and_fx(account, req.amount, req.currency)
        return PreflightResult(
            ok=not blocking,
            source_account=_to_account_out(account),
            warnings=warnings,
            blocking=blocking,
        )


def preflight_bill(req: PreflightBillRequest) -> PreflightResult:
    with session_scope() as session:
        account = _resolve_account(
            session, req.owner_user, req.source_account, req.source_account_type
        )
        if account is None:
            return PreflightResult(ok=False, blocking=["source_account_not_found"])
        warnings, blocking = _funds_and_fx(account, req.amount, req.currency)
        return PreflightResult(
            ok=not blocking,
            source_account=_to_account_out(account),
            warnings=warnings,
            blocking=blocking,
        )


def add_beneficiary(req: AddBeneficiaryRequest) -> AddBeneficiaryResult:
    with session_scope() as session:
        existing = session.scalars(
            select(Beneficiary).where(
                Beneficiary.owner_user == req.owner_user,
                Beneficiary.account == req.account,
            )
        ).first()
        if existing is not None:
            return AddBeneficiaryResult(
                ok=False, message="A beneficiary with that account already exists."
            )
        row = Beneficiary(
            id=uuid.uuid4().hex[:8],
            owner_user=req.owner_user,
            name=req.name,
            name_ar=req.name_ar,
            account=req.account,
            bank=req.bank,
            currency=req.currency,
            status="active",
            is_favorite=False,
        )
        session.add(row)
        session.flush()
        out = BeneficiaryOut(
            id=row.id,
            name=row.name,
            name_ar=row.name_ar,
            account=row.account,
            bank=row.bank,
            currency=row.currency,
            status=row.status,
            is_favorite=row.is_favorite,
        )
    return AddBeneficiaryResult(ok=True, beneficiary=out)
