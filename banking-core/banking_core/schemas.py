"""Pydantic request/response models for the Banking Core API."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class BalanceRequest(BaseModel):
    owner_user: str = Field(..., min_length=1)
    account: str | None = None
    account_type: str | None = None


class AccountsRequest(BaseModel):
    owner_user: str = Field(..., min_length=1)


class AccountOut(BaseModel):
    account_id: str
    account_type: str
    number: str
    currency: str
    balance: Decimal
    status: str


class AccountsOut(BaseModel):
    accounts: list[AccountOut] = Field(default_factory=list)


class BeneficiaryOut(BaseModel):
    id: str
    name: str
    name_ar: str | None = None
    account: str
    bank: str | None = None
    currency: str
    status: str
    is_favorite: bool = False


class PreflightTransferRequest(BaseModel):
    owner_user: str = Field(..., min_length=1)
    amount: Decimal = Field(..., gt=0)
    currency: str
    recipient_account: str | None = None
    source_account: str | None = None
    source_account_type: str | None = None


class PreflightBillRequest(BaseModel):
    owner_user: str = Field(..., min_length=1)
    biller_code: str | None = None
    reference_number: str
    amount: Decimal = Field(..., gt=0)
    currency: str
    source_account: str | None = None
    source_account_type: str | None = None


class PreflightResult(BaseModel):
    ok: bool
    source_account: AccountOut | None = None
    # Advisory notes that DO NOT block confirmation (low funds, FX conversion).
    warnings: list[str] = Field(default_factory=list)
    # Hard stops that SHOULD block (account not found / inactive).
    blocking: list[str] = Field(default_factory=list)


class AddBeneficiaryRequest(BaseModel):
    owner_user: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    name_ar: str | None = None
    account: str = Field(..., min_length=1)
    bank: str | None = None
    currency: str = "SAR"


class AddBeneficiaryResult(BaseModel):
    ok: bool
    beneficiary: BeneficiaryOut | None = None
    message: str | None = None
