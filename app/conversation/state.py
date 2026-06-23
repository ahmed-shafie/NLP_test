"""Serializable conversation state for the slot-filling state machine."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas import Intent, Language

# Slots that must be filled before each action can be confirmed, per intent.
REQUIRED_SLOTS: tuple[str, ...] = ("amount", "currency", "recipient")
BILL_REQUIRED_SLOTS: tuple[str, ...] = (
    "biller",
    "reference_number",
    "amount",
    "currency",
)


class ConversationStatus(str, Enum):
    """Where the dialogue currently stands."""

    SELECTING = "selecting"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ConversationSlots(BaseModel):
    """Slots gathered across turns (covers both transfers and bill payments)."""

    amount: Decimal | None = None
    currency: str | None = None
    recipient: str | None = None
    source_account: str | None = None
    account_number: str | None = None
    biller: str | None = None
    biller_category: str | None = None
    reference_number: str | None = None
    note: str | None = None

    def first_missing_required(
        self, required: tuple[str, ...] = REQUIRED_SLOTS
    ) -> str | None:
        for slot in required:
            value = getattr(self, slot)
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
