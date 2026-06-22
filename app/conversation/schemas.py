"""Request/response schemas for the conversation endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.conversation.state import ConversationSlots, ConversationStatus
from app.schemas import Intent, Language, TransferRequest


class ConversationRequest(BaseModel):
    """Input for ``POST /conversation/text``."""

    text: str = Field(..., min_length=1, description="The user's message this turn.")
    session_id: str | None = Field(
        default=None,
        description="Opaque session id; a new one is created when omitted.",
    )
    language: Language | None = Field(
        default=None, description="Optional language hint; auto-detected otherwise."
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "Optional user identifier; scopes the Memory Brain (habits + shortcuts) "
            "so known preferences pre-fill slots and completed transfers are learned."
        ),
    )


class ConversationResponse(BaseModel):
    """Output for the conversation endpoints."""

    session_id: str
    reply: str
    status: ConversationStatus
    language: Language
    intent: Intent | None = None
    pending_slot: str | None = None
    complete: bool = False
    slots: ConversationSlots
    transfer: TransferRequest | None = None
