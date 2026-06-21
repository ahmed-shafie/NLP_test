"""FastAPI application exposing the banking NLU brain."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.admin import audit
from app.admin.router import router as admin_router
from app.admin.store import get_engine
from app.config import settings
from app.db.beneficiary import get_beneficiary_repository
from app.embeddings import get_embedder
from app.errors import register_exception_handlers
from app.llm import get_llm_handler
from app.middleware import BodySizeLimitMiddleware, SecurityHeadersMiddleware
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
from app.security import require_api_key

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Optionally warm the NLP models so the first request is fast."""

    # Ensure the admin store (connections + audit log) tables exist.
    get_engine()
    if settings.preload_models:
        english._load_model()
        arabic._load_model()
        get_embedder()
        get_semantic_classifier()
        get_nlu_pipeline()
        get_llm_handler()
        get_beneficiary_repository()
    yield


app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)

register_exception_handlers(app)

# Middleware is applied outermost-first (last added runs first). Audit stays innermost
# so it records the final status code; the body-size guard runs first to reject early.
if settings.audit_enabled:
    app.add_middleware(audit.AuditMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
if settings.cors_origins_list():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.add_middleware(BodySizeLimitMiddleware)

app.include_router(admin_router)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """Serve the browser-based simulation & testing page."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin", include_in_schema=False)
def admin_connections_page() -> FileResponse:
    """Serve the external-resource connections configuration page."""

    return FileResponse(STATIC_DIR / "connections.html")


@app.get("/admin/audit", include_in_schema=False)
def admin_audit_page() -> FileResponse:
    """Serve the audit log monitor / observability dashboard."""

    return FileResponse(STATIC_DIR / "audit.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""

    return {"status": "ok", "version": __version__}


@app.post("/nlu/parse", response_model=NLUResponse)
def nlu_parse(
    request: ParseRequest,
    http_request: Request,
    _: str = Depends(require_api_key),
) -> NLUResponse:
    """Parse raw text into an intent and transfer slots.

    When ``account_number`` is supplied, the destination beneficiary is resolved by an
    account lookup against the configured database provider; if the account is not
    found, the LiteLLM handler processes the query and generates the response.
    """

    result = pipeline.parse(request.text, request.language, request.account_number)
    audit.record(
        "nlu.parse",
        category="nlu",
        actor=http_request.headers.get("x-actor") or "anonymous",
        detail={
            "language": result.language.value,
            "intent": result.intent.value,
            "intent_source": result.intent_source,
            "account_number": request.account_number,
            "beneficiary_source": result.beneficiary_source,
            "llm_assisted": result.llm_assisted,
        },
    )
    return result


@app.post("/transfer/validate", response_model=ValidationResult)
def transfer_validate(
    request: ValidateRequest, _: str = Depends(require_api_key)
) -> ValidationResult:
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
    _: str = Depends(require_api_key),
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
def contacts_resolve(
    request: ResolveContactRequest, _: str = Depends(require_api_key)
) -> ResolveContactResponse:
    """Resolve a recipient name to an address-book contact (cross-lingual)."""

    matched, candidates = pipeline.resolve_contact(
        request.name, request.contacts, request.top_k
    )
    return ResolveContactResponse(matched=matched, candidates=candidates)
