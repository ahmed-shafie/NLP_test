"""HTTP client for the standalone Banking Core service.

Used for the API-backed cases: account balance, pre-flight funds/FX checks, and
adding a beneficiary. Every call degrades gracefully — on any network/HTTP error
it returns ``None`` (or an unavailable result) so a turn is never broken by the
external service being down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountInfo:
    account_id: str
    account_type: str
    number: str
    currency: str
    balance: Decimal
    status: str


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    source_account: AccountInfo | None = None
    warnings: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)


def _headers() -> dict[str, str]:
    if settings.banking_core_api_key:
        return {"x-api-key": settings.banking_core_api_key}
    return {}


def _post(path: str, payload: dict) -> dict | None:
    if not settings.banking_core_enabled:
        return None
    url = f"{settings.banking_core_url.rstrip('/')}{path}"
    try:
        resp = httpx.post(
            url, json=payload, headers=_headers(), timeout=settings.banking_core_timeout
        )
    except Exception as exc:  # noqa: BLE001 - service optional; never break a turn
        logger.warning("Banking Core call to %s failed: %s", path, exc)
        return None
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        logger.warning("Banking Core %s -> HTTP %s", path, resp.status_code)
        return None
    return resp.json()


def health() -> bool:
    """Return ``True`` if the Banking Core /health endpoint responds ok."""

    url = f"{settings.banking_core_url.rstrip('/')}/health"
    try:
        resp = httpx.get(url, timeout=settings.banking_core_timeout)
    except Exception as exc:  # noqa: BLE001 - report unreachable, never raise
        logger.warning("Banking Core health check failed: %s", exc)
        return False
    return resp.status_code == 200


def _to_account(data: dict | None) -> AccountInfo | None:
    if not data:
        return None
    return AccountInfo(
        account_id=str(data["account_id"]),
        account_type=str(data["account_type"]),
        number=str(data["number"]),
        currency=str(data["currency"]),
        balance=Decimal(str(data["balance"])),
        status=str(data["status"]),
    )


def get_balance(
    owner_user: str, account: str | None = None, account_type: str | None = None
) -> AccountInfo | None:
    data = _post(
        "/accounts/balance",
        {"owner_user": owner_user, "account": account, "account_type": account_type},
    )
    return _to_account(data)


def _to_preflight(data: dict | None) -> PreflightResult | None:
    if data is None:
        return None
    return PreflightResult(
        ok=bool(data.get("ok")),
        source_account=_to_account(data.get("source_account")),
        warnings=list(data.get("warnings") or []),
        blocking=list(data.get("blocking") or []),
    )


def preflight_transfer(
    owner_user: str,
    amount: Decimal,
    currency: str,
    recipient_account: str | None = None,
    source_account: str | None = None,
    source_account_type: str | None = None,
) -> PreflightResult | None:
    data = _post(
        "/preflight/transfer",
        {
            "owner_user": owner_user,
            "amount": str(amount),
            "currency": currency,
            "recipient_account": recipient_account,
            "source_account": source_account,
            "source_account_type": source_account_type,
        },
    )
    return _to_preflight(data)


def preflight_bill(
    owner_user: str,
    amount: Decimal,
    currency: str,
    biller_code: str | None,
    reference_number: str,
    source_account: str | None = None,
    source_account_type: str | None = None,
) -> PreflightResult | None:
    data = _post(
        "/preflight/bill",
        {
            "owner_user": owner_user,
            "amount": str(amount),
            "currency": currency,
            "biller_code": biller_code,
            "reference_number": reference_number,
            "source_account": source_account,
            "source_account_type": source_account_type,
        },
    )
    return _to_preflight(data)


def add_beneficiary(
    owner_user: str,
    name: str,
    account: str,
    bank: str | None = None,
    currency: str = "SAR",
) -> dict | None:
    return _post(
        "/beneficiary/add",
        {
            "owner_user": owner_user,
            "name": name,
            "account": account,
            "bank": bank,
            "currency": currency,
        },
    )
