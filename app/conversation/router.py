"""FastAPI routes for the multi-turn conversation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.config import settings
from app.conversation import templates
from app.conversation.engine import ConversationResult, get_engine
from app.conversation.schemas import (
    ConversationRequest,
    ConversationResponse,
    OpeningResponse,
)
from app.conversation.state import ConversationStatus
from app.schemas import Language

router = APIRouter(tags=["conversation"])


def _to_response(result: ConversationResult) -> ConversationResponse:
    state = result.state
    return ConversationResponse(
        session_id=state.session_id,
        reply=result.reply,
        status=state.status,
        language=state.language,
        intent=state.intent,
        pending_slot=state.pending_slot,
        complete=state.status is ConversationStatus.COMPLETED,
        slots=state.slots,
        transfer=result.transfer,
        bill=result.bill,
        flagged_terms=result.flagged_terms,
        warnings=state.preflight_warnings,
        block_trace=result.block_trace,
    )


def _require_conversation() -> None:
    if not settings.conversation_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The conversation engine is disabled.",
        )


@router.get("/conversation/opening", response_model=OpeningResponse)
def conversation_opening(
    language: Language = Query(
        default=Language.EN, description="Language to greet the customer in."
    ),
) -> OpeningResponse:
    """Return the line to show when a conversation opens, before any message."""

    _require_conversation()
    return OpeningResponse(reply=templates.opening(language), language=language)


@router.post("/conversation/text", response_model=ConversationResponse)
def conversation_text(request: ConversationRequest) -> ConversationResponse:
    """Advance a multi-turn transfer dialogue with a single text message."""

    _require_conversation()
    result = get_engine().handle(
        request.text,
        request.session_id,
        request.language,
        request.user_id,
    )
    return _to_response(result)
