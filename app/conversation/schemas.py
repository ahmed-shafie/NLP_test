"""Request/response schemas for the conversation endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.conversation.state import ConversationSlots, ConversationStatus
from app.schemas import BillPaymentRequest, Intent, Language, TransferRequest
from app.trace import BlockTrace


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
    bill: BillPaymentRequest | None = None
    flagged_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Abusive terms detected in the user's message this turn (for UI "
            "highlighting); empty when the message is clean."
        ),
    )
    block_trace: list[BlockTrace] = Field(
        default_factory=list,
        description="Per-block execution trace (timing + status) for this turn.",
    )
