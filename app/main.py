"""FastAPI application exposing the banking NLU brain."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app import __version__
from app.admin import audit
from app.admin.router import router as admin_router
from app.admin.store import get_engine
from app.config import settings
from app.conversation.router import router as conversation_router
from app.db.beneficiary import get_beneficiary_repository
from app.embeddings import get_embedder
from app.errors import register_exception_handlers
from app.llm import get_llm_handler
from app.logging_config import configure_logging
from app.metrics import MetricsMiddleware, metrics_response
from app.middleware import BodySizeLimitMiddleware, SecurityHeadersMiddleware
from app.nlu import arabic, english, pipeline
from app.nlu.semantic_intents import get_semantic_classifier
from app.orchestration import get_nlu_pipeline
from app.ratelimit import RateLimitMiddleware
from app.request_context import RequestContextMiddleware
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

configure_logging()


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

# Middleware is applied outermost-first (last added runs first). The intended order,
# outermost -> innermost: request-context (assigns the request id used by logs, errors
# and audit) -> CORS -> metrics -> security headers -> rate limit -> body-size guard ->
# audit. Outer placement of CORS/security-headers ensures even early rejections (429,
# 413) carry those headers; audit stays innermost so it records the final status code.
if settings.audit_enabled:
    app.add_middleware(audit.AuditMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
if settings.metrics_enabled:
    app.add_middleware(MetricsMiddleware)
if settings.cors_origins_list():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.add_middleware(RequestContextMiddleware)

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
    """Liveness probe: the process is up and serving."""

    return {"status": "ok", "version": __version__}


@app.get("/health/ready")
def readiness() -> JSONResponse:
    """Readiness probe: report dependency health and gate on the audit/config store."""

    checks: dict[str, str] = {}
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["store"] = "ok"
    except Exception:  # noqa: BLE001 - readiness must report, not raise
        checks["store"] = "error"
    # Informational only: the service degrades gracefully without the embedder.
    checks["embedder"] = "ok" if get_embedder() is not None else "unavailable"

    ready = checks["store"] == "ok"
    body = {"status": "ready" if ready else "not_ready", "checks": checks}
    return JSONResponse(body, status_code=200 if ready else 503)


if settings.metrics_enabled:

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus metrics exposition."""

        return metrics_response()


nlu_router = APIRouter(tags=["nlu"])


@nlu_router.post("/nlu/parse", response_model=NLUResponse)
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


@nlu_router.post("/transfer/validate", response_model=ValidationResult)
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


@nlu_router.get("/nlu/similar", response_model=list[SimilarExampleSchema])
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


@nlu_router.post("/contacts/resolve", response_model=ResolveContactResponse)
def contacts_resolve(
    request: ResolveContactRequest, _: str = Depends(require_api_key)
) -> ResolveContactResponse:
    """Resolve a recipient name to an address-book contact (cross-lingual)."""

    matched, candidates = pipeline.resolve_contact(
        request.name, request.contacts, request.top_k
    )
    return ResolveContactResponse(matched=matched, candidates=candidates)


# Public endpoints are served at both the unversioned paths (kept for
# back-compatibility) and under the canonical "/v1" prefix.
app.include_router(nlu_router)
app.include_router(nlu_router, prefix="/v1")
app.include_router(conversation_router)
app.include_router(conversation_router, prefix="/v1")
