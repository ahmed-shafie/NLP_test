"""Serializable conversation state for the slot-filling state machine."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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
    # The dialogue ran to its end but the bank refused the action (a rejected
    # write, an unreachable account). Terminal like COMPLETED, and never a
    # claim that anything was done.
    FAILED = "failed"


class SlotSource(str, Enum):
    """Where a slot's value came from.

    Recorded per slot so a confirmed transfer can be read back afterwards: an
    amount is only ever ``USER_TEXT``, ``MEMORY_SHORTCUT`` or ``BANKING_CORE``,
    and a recipient that reached the Core must say ``DIRECTORY`` — a name still
    marked ``USER_TEXT`` at confirmation means identity was never resolved.
    """

    USER_TEXT = "user_text"
    MEMORY_SHORTCUT = "memory_shortcut"
    DIRECTORY = "directory"
    BILLER_CATALOGUE = "biller_catalogue"
    BANKING_CORE = "banking_core"
    DEFAULT = "default"


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


class PendingFlowSwitch(BaseModel):
    """A fresh instruction held back while we ask which flow to serve.

    ``text`` is the customer's own message, replayed unchanged if they choose
    the new request, so the amount and recipient are read from what they wrote
    rather than from anything we inferred while the other flow was open.
    """

    intent: Intent
    text: str


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
    # Which source filled each slot (slot name -> ``SlotSource`` value), kept in
    # step with ``slots`` by ``slots_from``.
    slot_provenance: dict[str, str] = Field(default_factory=dict)
    # A new request that arrived while this flow was waiting for a slot, held
    # until the customer says which of the two to serve.
    pending_switch: PendingFlowSwitch | None = None

    @contextmanager
    def slots_from(self, source: SlotSource) -> Iterator[ConversationSlots]:
        """Attribute every slot the block fills to ``source``.

        Compares the slots before and after, so the caller states the source
        once instead of at each assignment and cannot forget one.
        """

        before = self.slots.model_dump()
        try:
            yield self.slots
        finally:
            for slot, value in self.slots.model_dump().items():
                if value is None or value == "":
                    self.slot_provenance.pop(slot, None)
                elif before.get(slot) != value:
                    self.slot_provenance[slot] = source.value

    def reset(self) -> None:
        """Clear transfer progress to begin a fresh dialogue (keeps language)."""

        self.intent = None
        self.status = ConversationStatus.COLLECTING
        self.slots = ConversationSlots()
        self.slot_provenance = {}
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
        self.pending_switch = None
