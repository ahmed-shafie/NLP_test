"""FastAPI application exposing the banking NLU brain."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import settings
from app.embeddings import get_embedder
from app.llm import get_llm_handler
from app.nlu import arabic, english, pipeline
from app.nlu.semantic_intents import get_semantic_classifier
from app.orchestration import get_nlu_pipeline
from app.schemas import (
    NLUResponse,
    ParseRequest,
    ResolveContactRequest,
    ResolveContactResponse,
    SimilarExampleSchema,
    ValidateRequest,
    ValidationResult,
)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Optionally warm the NLP models so the first request is fast."""

    if settings.preload_models:
        english._load_model()
        arabic._load_model()
        get_embedder()
        get_semantic_classifier()
        get_nlu_pipeline()
        get_llm_handler()
    yield


app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """Serve the browser-based simulation & testing page."""

    return FileResponse(STATIC_DIR / "index.html")


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


@app.get("/nlu/similar", response_model=list[SimilarExampleSchema])
def nlu_similar(
    text: str = Query(
        ..., min_length=1, description="Utterance to find neighbours for."
    ),
    k: int = Query(default=5, ge=1, le=20),
) -> list[SimilarExampleSchema]:
    """Return the nearest labeled example utterances (semantic debug/eval)."""

    classifier = get_semantic_classifier()
    if classifier is None:
        return []
    return [
        SimilarExampleSchema(text=ex.text, intent=ex.intent, score=ex.score)
        for ex in classifier.similar(text, k)
    ]


@app.post("/contacts/resolve", response_model=ResolveContactResponse)
def contacts_resolve(request: ResolveContactRequest) -> ResolveContactResponse:
    """Resolve a recipient name to an address-book contact (cross-lingual)."""

    matched, candidates = pipeline.resolve_contact(
        request.name, request.contacts, request.top_k
    )
    return ResolveContactResponse(matched=matched, candidates=candidates)
