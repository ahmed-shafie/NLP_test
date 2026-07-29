"""Headless HTTP API — a single conversational endpoint. No GUI.

Mirrors ``app/conversation/router.py``. Run it with::

    uvicorn service_template.api:app --reload --port 8200

Then drive it with plain HTTP (curl / Postman / another service)::

    curl -s localhost:8200/conversation/text \\
        -H 'content-type: application/json' \\
        -d '{"text": "send 500 SAR to Ahmed", "session_id": "s1"}'

The response echoes the FSM status, the slots gathered so far, and — once the
dialogue completes — the validated ``action`` object ready for a downstream
system. Keep calling with the SAME ``session_id`` to continue a dialogue.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from service_template import __version__
from service_template.config import settings
from service_template.core_client import health as core_health
from service_template.engine import get_engine
from service_template.schemas import (
    ActionSlots,
    Intent,
    Language,
    TransferAction,
)
from service_template.state import ConversationStatus


class ConversationRequest(BaseModel):
    """Input for ``POST /conversation/text``."""

    text: str = Field(..., min_length=1, description="The user's message this turn.")
    session_id: str | None = Field(
        default=None, description="Opaque id; a new one is created when omitted."
    )
    language: Language | None = Field(
        default=None, description="Optional language hint; auto-detected otherwise."
    )
    user_id: str | None = Field(
        default=None, description="Optional user id (scopes accounts downstream)."
    )


class ConversationResponse(BaseModel):
    """Output for ``POST /conversation/text``."""

    session_id: str
    reply: str
    status: ConversationStatus
    language: Language
    intent: Intent | None = None
    pending_slot: str | None = None
    complete: bool = False
    slots: ActionSlots
    warnings: list[str] = Field(default_factory=list)
    # The validated action object — present only when ``complete`` is true.
    action: TransferAction | None = None


app = FastAPI(title=settings.app_name, version=__version__)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness + whether the optional external core service is reachable."""

    return {
        "status": "ok",
        "version": __version__,
        "core_enabled": settings.core_enabled,
        "core_reachable": core_health(),
    }


@app.post("/conversation/text", response_model=ConversationResponse)
def conversation_text(request: ConversationRequest) -> ConversationResponse:
    """Advance the multi-turn dialogue by one user message."""

    result = get_engine().handle(
        request.text, request.session_id, request.language, request.user_id
    )
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
        warnings=state.warnings,
        action=result.action,
    )
