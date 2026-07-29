"""Per-session conversation state and the FSM status enum.

Mirrors ``app/conversation/state.py``. The whole dialogue is a finite-state
machine; ``ConversationState`` is the serializable snapshot the session store
persists between turns.

Status transitions for the example "transfer" case::

    COLLECTING ─▶ (missing slots asked one by one)
       │  all slots present
       ▼
    DISAMBIGUATING ─▶ (only when several beneficiaries share a name)
       │  one candidate chosen
       ▼
    CONFIRMING ─▶ yes ─▶ COMPLETED   (emit the action object)
       │           no  ─▶ CANCELLED
       ▼
    (a fresh message after COMPLETED/CANCELLED resets to COLLECTING)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from service_template.schemas import ActionSlots, Intent, Language

# Slots that must be present before the "transfer" action can be confirmed.
# >>> EDIT PER CASE: define the required slots for your case, e.g.
#     BILL_REQUIRED_SLOTS = ("biller", "reference_number", "amount", "currency")
TRANSFER_REQUIRED_SLOTS: tuple[str, ...] = ("amount", "currency", "recipient")


class ConversationStatus(str, Enum):
    """Where the dialogue currently stands."""

    COLLECTING = "collecting"  # still gathering slots
    DISAMBIGUATING = "disambiguating"  # asking the user to pick among candidates
    CONFIRMING = "confirming"  # slots complete, awaiting yes/no
    COMPLETED = "completed"  # action emitted
    CANCELLED = "cancelled"  # user cancelled / said no


class Candidate(BaseModel):
    """One option shown during DISAMBIGUATING (e.g. one of several "Ahmed"s)."""

    id: str
    name: str
    detail: str | None = None  # free-text (bank, masked account, currency, ...)


class ConversationState(BaseModel):
    """Full per-session state, persisted by the session store between turns."""

    session_id: str
    user_id: str | None = None
    language: Language = Language.EN
    intent: Intent | None = None
    status: ConversationStatus = ConversationStatus.COLLECTING
    slots: ActionSlots = Field(default_factory=ActionSlots)

    # The slot we last asked about (so an unrecognised reply re-asks the same
    # question rather than losing context).
    pending_slot: str | None = None

    # Disambiguation candidates + the resolved recipient flag.
    candidates: list[Candidate] = Field(default_factory=list)
    recipient_resolved: bool = False

    # Advisory pre-flight notes (low funds / FX). These are shown at
    # confirmation but NEVER block — matching the product rule in the main app.
    warnings: list[str] = Field(default_factory=list)

    turns: int = 0

    def reset(self) -> None:
        """Clear progress to begin a fresh dialogue (keeps language)."""

        self.intent = None
        self.status = ConversationStatus.COLLECTING
        self.slots = ActionSlots()
        self.pending_slot = None
        self.candidates = []
        self.recipient_resolved = False
        self.warnings = []
