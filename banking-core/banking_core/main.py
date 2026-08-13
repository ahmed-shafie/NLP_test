"""FastAPI app for the standalone Banking Core service."""

from __future__ import annotations

import hmac
import logging

from fastapi import Depends, FastAPI, Header, HTTPException

from banking_core import __version__, service
from banking_core.config import settings
from banking_core.db import init_db
from banking_core.schemas import (
    AccountOut,
    AddBeneficiaryRequest,
    AddBeneficiaryResult,
    BalanceRequest,
    PreflightBillRequest,
    PreflightResult,
    PreflightTransferRequest,
)
from banking_core.seed import seed

logger = logging.getLogger(__name__)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce the optional shared API key when one is configured."""

    if settings.api_key and not (
        x_api_key is not None and hmac.compare_digest(x_api_key, settings.api_key)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


app = FastAPI(title=settings.app_name, version=__version__)


@app.on_event("startup")
def _startup() -> None:
    """Provision the schema, and demo rows on a brand-new (empty) database."""

    if settings.auto_create_tables:
        init_db()
    if settings.seed_on_startup and seed(reset=False):
        logger.info("Seeded demo data into the empty database.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/accounts/balance", response_model=AccountOut)
def accounts_balance(
    req: BalanceRequest, _: None = Depends(require_api_key)
) -> AccountOut:
    account = service.get_balance(req.owner_user, req.account, req.account_type)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found.")
    return account


@app.post("/preflight/transfer", response_model=PreflightResult)
def preflight_transfer(
    req: PreflightTransferRequest, _: None = Depends(require_api_key)
) -> PreflightResult:
    return service.preflight_transfer(req)


@app.post("/preflight/bill", response_model=PreflightResult)
def preflight_bill(
    req: PreflightBillRequest, _: None = Depends(require_api_key)
) -> PreflightResult:
    return service.preflight_bill(req)


@app.post("/beneficiary/add", response_model=AddBeneficiaryResult)
def beneficiary_add(
    req: AddBeneficiaryRequest, _: None = Depends(require_api_key)
) -> AddBeneficiaryResult:
    return service.add_beneficiary(req)
