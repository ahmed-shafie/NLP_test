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
    DISAMBIGUATING = "disambiguating"
    CONFIRMING = "confirming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class BillerOption(BaseModel):
    """A candidate biller offered when a generic term is ambiguous."""

    code: str
    name: str
    category: str | None = None


class BeneficiaryOption(BaseModel):
    """A candidate beneficiary offered when a first name matches several people."""

    id: str
    name: str
    account: str
    bank: str | None = None
    currency: str = "SAR"
    is_favorite: bool = False
    name_ar: str | None = None


class ConversationSlots(BaseModel):
    """Slots gathered across turns (covers both transfers and bill payments)."""

    amount: Decimal | None = None
    currency: str | None = None
    recipient: str | None = None
    source_account: str | None = None
    account_number: str | None = None
    biller: str | None = None
    biller_category: str | None = None
    biller_code: str | None = None
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
    biller_options: list[BillerOption] = Field(default_factory=list)
    # Beneficiary disambiguation (transfer): candidates + which flow is disambiguating.
    beneficiary_options: list[BeneficiaryOption] = Field(default_factory=list)
    disambiguation_kind: str | None = None  # "biller" | "beneficiary"
    # Whether the transfer recipient has been resolved to a directory beneficiary.
    beneficiary_resolved: bool = False
    # In-progress "add beneficiary" flow: the name we are collecting an account
    # for, the validated account awaiting confirmation, and whether adding was
    # triggered mid-transfer (so the transfer resumes once they're saved).
    pending_add_name: str | None = None
    pending_add_account: str | None = None
    add_resumes_transfer: bool = False
    # An IBAN that failed only its mod-97 checksum, held for one turn in case the
    # customer insists it is right; ``account_checksum_overridden`` records that
    # they did, so the write is traceable to their explicit override.
    pending_unchecked_account: str | None = None
    account_checksum_overridden: bool = False
    # Advisory pre-flight notes (FX) shown at confirmation; these never block.
    preflight_warnings: list[str] = Field(default_factory=list)
    # Pre-flight refusals from the Banking Core (insufficient funds, inactive
    # account): confirmation is not offered while any of these stand.
    preflight_blocking: list[str] = Field(default_factory=list)
    # The spendable balance offered after an insufficient-funds refusal, so a
    # plain "yes" means "send that amount instead".
    offered_amount: Decimal | None = None
    turns: int = 0
    # Count of flagged (abusive) turns in this session; drives the repeat-offense
    # cutoff. Kept across ``reset`` so it spans the whole session.
    flagged_count: int = 0
    # Last picked index per varied-reply group, so a reply isn't repeated
    # back-to-back. Keyed e.g. "inappropriate:mild:en".
    last_variant: dict[str, int] = Field(default_factory=dict)

    def reset(self) -> None:
        """Clear transfer progress to begin a fresh dialogue (keeps language)."""

        self.intent = None
        self.status = ConversationStatus.COLLECTING
        self.slots = ConversationSlots()
        self.pending_slot = None
        self.biller_options = []
        self.beneficiary_options = []
        self.disambiguation_kind = None
        self.beneficiary_resolved = False
        self.pending_add_name = None
        self.pending_add_account = None
        self.add_resumes_transfer = False
        self.pending_unchecked_account = None
        self.account_checksum_overridden = False
        self.preflight_warnings = []
        self.preflight_blocking = []
        self.offered_amount = None
