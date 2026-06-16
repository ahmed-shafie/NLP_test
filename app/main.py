"""FastAPI application exposing the banking NLU brain."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app import __version__
from app.config import settings
from app.nlu import arabic, english, pipeline
from app.schemas import NLUResponse, ParseRequest, ValidateRequest, ValidationResult

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Optionally warm the NLP models so the first request is fast."""

    if settings.preload_models:
        english._load_model()
        arabic._load_model()
    yield


app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""

    return {"status": "ok", "version": __version__}


@app.post("/nlu/parse", response_model=NLUResponse)
def nlu_parse(request: ParseRequest) -> NLUResponse:
    """Parse raw text into an intent and transfer slots."""

    return pipeline.parse(request.text, request.language)


@app.post("/transfer/validate", response_model=ValidationResult)
def transfer_validate(request: ValidateRequest) -> ValidationResult:
    """Validate gathered transfer slots, returning errors or a ready transfer."""

    return pipeline.validate_transfer(
        amount=request.amount,
        currency=request.currency,
        recipient=request.recipient,
        source_account=request.source_account,
        note=request.note,
    )
