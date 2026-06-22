"""Serializable conversation state for the slot-filling state machine."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas import Intent, Language

# Slots that must be filled before a transfer can be confirmed.
REQUIRED_SLOTS: tuple[str, ...] = ("amount", "currency", "recipient")


class ConversationStatus(str, Enum):
    """Where the dialogue currently stands."""

    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ConversationSlots(BaseModel):
    """Transfer slots gathered across turns."""

    amount: Decimal | None = None
    currency: str | None = None
    recipient: str | None = None
    source_account: str | None = None
    account_number: str | None = None
    note: str | None = None

    def first_missing_required(self) -> str | None:
        candidates: tuple[tuple[str, object | None], ...] = (
            ("amount", self.amount),
            ("currency", self.currency),
            ("recipient", self.recipient),
        )
        for slot, value in candidates:
            if value is None or (isinstance(value, str) and not value.strip()):
                return slot
        return None


class ConversationState(BaseModel):
    """Full per-session state, persisted to the session store between turns."""

    session_id: str
    user_id: str | None = None
    language: Language = Language.EN
    intent: Intent | None = None
    status: ConversationStatus = ConversationStatus.COLLECTING
    slots: ConversationSlots = Field(default_factory=ConversationSlots)
    pending_slot: str | None = None
    turns: int = 0

    def reset(self) -> None:
        """Clear transfer progress to begin a fresh dialogue (keeps language)."""

        self.intent = None
        self.status = ConversationStatus.COLLECTING
        self.slots = ConversationSlots()
        self.pending_slot = None
