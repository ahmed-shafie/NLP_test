"""The conversation engine: a small, predictable finite-state machine.

This is the heart of the template and the file you will extend most. It mirrors
``app/conversation/engine.py`` but is stripped to the essentials so the control
flow is obvious.

Every turn goes through ``handle()``:

    load state ─▶ (cancel? / mid-flow aside?) ─▶ dispatch on status
        COLLECTING     ─▶ extract + merge slots, ask next question,
                          or (recipient ambiguous) → DISAMBIGUATING,
                          or (slots complete) → pre-flight → CONFIRMING
        DISAMBIGUATING ─▶ interpret pick (number / name) → CONFIRMING
        CONFIRMING     ─▶ yes → COMPLETED (+emit action) | no → CANCELLED
    save state ─▶ return reply (+ action object when complete)

Design rules worth preserving when you extend it:
* **Never overwrite a filled slot** with a later, emptier extraction.
* **Ask one question at a time** (``first_missing``).
* **Pre-flight is advisory**: warnings never block ``yes``; only ``blocking``
  hard-stops do.
* **The engine only emits a validated action object** — it never executes it.
"""

from __future__ import annotations

import re
import uuid
from functools import lru_cache

from service_template import extractor, prompts
from service_template.config import settings
from service_template.core_client import preflight_transfer
from service_template.schemas import ActionSlots, Intent, Language, TransferAction
from service_template.state import (
    TRANSFER_REQUIRED_SLOTS,
    Candidate,
    ConversationState,
    ConversationStatus,
)
from service_template.store import get_session_store

# Replies that mean "yes" / "no" / "cancel", in both languages.
_AFFIRM = {"yes", "y", "yeah", "yep", "confirm", "ok", "okay", "نعم", "اكيد", "أكيد"}
_NEGATIVE = {"no", "n", "nope", "لا", "كلا"}
_CANCEL = {"cancel", "stop", "الغاء", "إلغاء", "توقف"}


# --------------------------------------------------------------------------- #
# A tiny in-memory "beneficiary directory" so the disambiguation path is real.
# In the main app this is a direct DB query (``app/db/directory.py``). Replace
# ``_lookup_recipients`` with your own data source.
# --------------------------------------------------------------------------- #
_DIRECTORY: list[dict[str, str]] = [
    {"id": "b1", "name": "Ahmed Hassan", "detail": "Al Rajhi · SA••7777 · SAR"},
    {"id": "b2", "name": "Ahmed Khaled", "detail": "SNB · SA••2211 · SAR"},
    {"id": "b3", "name": "Ahmed Mahmoud", "detail": "Riyad Bank · SA••8090 · USD"},
    {"id": "b4", "name": "Mona Ali", "detail": "SNB · SA••3333 · SAR"},
    {"id": "b5", "name": "Sara Adel", "detail": "SNB · SA••5555 · SAR"},
]


def _lookup_recipients(name: str) -> list[Candidate]:
    """Return directory rows whose name contains the requested (first) name."""

    needle = name.strip().lower()
    return [
        Candidate(id=row["id"], name=row["name"], detail=row["detail"])
        for row in _DIRECTORY
        if needle and needle in row["name"].lower()
    ]


class ConversationResult:
    """What a single turn produces: the new state, a reply, and (on success)
    the emitted action object."""

    def __init__(
        self,
        state: ConversationState,
        reply: str,
        action: TransferAction | None = None,
    ) -> None:
        self.state = state
        self.reply = reply
        self.action = action


class ConversationEngine:
    def __init__(self) -> None:
        self._store = get_session_store()

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def handle(
        self,
        text: str,
        session_id: str | None = None,
        language: Language | None = None,
        user_id: str | None = None,
    ) -> ConversationResult:
        sid = session_id or uuid.uuid4().hex
        state = self._store.load(sid) or ConversationState(session_id=sid)
        state.turns += 1
        if user_id:
            state.user_id = user_id
        lang = language or extractor.detect_language(text)
        state.language = lang

        # A fresh message after a finished dialogue starts over.
        if state.status in (ConversationStatus.COMPLETED, ConversationStatus.CANCELLED):
            state.reset()
            state.language = lang

        # Cancel takes priority in every state.
        if _matches(text, _CANCEL):
            state.reset()
            state.status = ConversationStatus.CANCELLED
            return self._finish(state, prompts.cancelled(lang))

        # Dispatch on the current FSM status.
        if state.status is ConversationStatus.CONFIRMING:
            result = self._handle_confirmation(state, text, lang)
        elif state.status is ConversationStatus.DISAMBIGUATING:
            result = self._handle_disambiguation(state, text, lang)
        else:
            result = self._collect(state, text, lang)
        return result

    # ------------------------------------------------------------------ #
    # COLLECTING: figure out intent, gather slots, advance the flow
    # ------------------------------------------------------------------ #
    def _collect(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        # Detect intent only on the first turn of a new action.
        if state.intent is None:
            state.intent = extractor.detect_intent(text)

        # Non-actionable intents: answer and wait (no flow started).
        if state.intent is Intent.SMALL_TALK:
            state.intent = None
            return self._finish(state, prompts.small_talk(lang))
        if state.intent is Intent.FALLBACK:
            state.intent = None
            return self._finish(state, prompts.fallback(lang))

        # >>> EDIT PER CASE: dispatch to the right collector for the intent.
        #     For now we only have TRANSFER_MONEY.
        return self._collect_transfer(state, text, lang)

    def _collect_transfer(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        # 1) Extract this turn's slots and merge without clobbering filled ones.
        extracted = extractor.extract_slots(text)
        self._merge_slots(state.slots, extracted)
        extractor.apply_defaults(state.slots)

        # 2) Ask for the first still-missing required slot, one at a time.
        missing = state.slots.first_missing(TRANSFER_REQUIRED_SLOTS)
        if missing is not None:
            state.status = ConversationStatus.COLLECTING
            state.pending_slot = missing
            return self._finish(state, prompts.slot_prompt(missing, lang))

        # 3) All slots present. Resolve the recipient against the directory.
        #    Several matches → disambiguate; exactly one → lock it; none → keep
        #    the typed name as-is (an "add beneficiary" flow would live here).
        if not state.recipient_resolved and state.slots.recipient:
            candidates = _lookup_recipients(state.slots.recipient)
            if len(candidates) > 1:
                state.candidates = candidates
                state.status = ConversationStatus.DISAMBIGUATING
                return self._finish(state, prompts.choose_candidate(candidates, lang))
            if len(candidates) == 1:
                state.slots.recipient = candidates[0].name
            state.recipient_resolved = True

        # 4) Slots complete + recipient resolved → pre-flight, then confirm.
        return self._enter_confirmation(state, lang)

    # ------------------------------------------------------------------ #
    # DISAMBIGUATING: interpret the user's pick
    # ------------------------------------------------------------------ #
    def _handle_disambiguation(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        picked = self._resolve_pick(state.candidates, text)
        if picked is None:
            # Unrecognised: re-show the same list.
            return self._finish(state, prompts.choose_candidate(state.candidates, lang))
        state.slots.recipient = picked.name
        state.recipient_resolved = True
        state.candidates = []
        return self._enter_confirmation(state, lang)

    @staticmethod
    def _resolve_pick(candidates: list[Candidate], text: str) -> Candidate | None:
        """Resolve a pick by list position (1..N) or by (partial) full name."""

        stripped = text.strip()
        match = re.fullmatch(r"\d+", stripped)
        if match:
            index = int(match.group()) - 1
            if 0 <= index < len(candidates):
                return candidates[index]
            return None
        lowered = stripped.lower()
        for candidate in candidates:
            if lowered and lowered in candidate.name.lower():
                return candidate
        return None

    # ------------------------------------------------------------------ #
    # CONFIRMING
    # ------------------------------------------------------------------ #
    def _enter_confirmation(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult:
        """Run advisory pre-flight, then ask for yes/no confirmation."""

        assert state.slots.amount is not None
        assert state.slots.currency is not None
        result = preflight_transfer(
            owner_user=state.user_id or "demo",
            amount=state.slots.amount,
            currency=state.slots.currency,
            source_account=state.slots.source_account,
        )
        state.warnings = list(result.warnings) if result else []
        state.status = ConversationStatus.CONFIRMING
        state.pending_slot = None
        return self._finish(state, self._confirm_text(state, lang))

    def _handle_confirmation(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        if _matches(text, _NEGATIVE):
            state.reset()
            state.status = ConversationStatus.CANCELLED
            return self._finish(state, prompts.cancelled(lang))
        if _matches(text, _AFFIRM):
            return self._complete(state, lang)
        # Anything else: re-ask the confirmation, keep the transaction intact.
        return self._finish(state, self._confirm_text(state, lang))

    def _confirm_text(self, state: ConversationState, lang: Language) -> str:
        assert state.slots.amount is not None and state.slots.currency is not None
        assert state.slots.recipient is not None
        text = prompts.confirm_transfer(
            state.slots.amount, state.slots.currency, state.slots.recipient, lang
        )
        note = prompts.warnings_note(state.warnings, lang)
        return f"{text} {note}" if note else text

    # ------------------------------------------------------------------ #
    # COMPLETED: build and emit the validated action object
    # ------------------------------------------------------------------ #
    def _complete(self, state: ConversationState, lang: Language) -> ConversationResult:
        # Building the strict model validates every slot one final time. If it
        # somehow fails, fall back to collecting the offending slot.
        try:
            action = TransferAction(
                amount=state.slots.amount,  # type: ignore[arg-type]
                currency=state.slots.currency,  # type: ignore[arg-type]
                recipient=state.slots.recipient,  # type: ignore[arg-type]
                source_account=state.slots.source_account,
                note=state.slots.note,
            )
        except ValueError:
            state.status = ConversationStatus.COLLECTING
            return self._finish(state, prompts.slot_prompt("amount", lang))
        recipient = action.recipient
        state.status = ConversationStatus.COMPLETED
        # NOTE: we return the action object for a downstream system to execute;
        # we never move money here.
        return self._finish(state, prompts.completed(recipient, lang), action=action)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_slots(current: ActionSlots, new: ActionSlots) -> None:
        """Copy filled values from ``new`` into ``current`` without clobbering."""

        for field_name in current.model_fields:
            incoming = getattr(new, field_name)
            if incoming is None:
                continue
            if isinstance(incoming, str) and not incoming.strip():
                continue
            if getattr(current, field_name) is None:
                setattr(current, field_name, incoming)

    def _finish(
        self,
        state: ConversationState,
        reply: str,
        action: TransferAction | None = None,
    ) -> ConversationResult:
        """Persist state and package the turn's result.

        Also enforces the ``max_turns`` safety valve so a stuck dialogue resets.
        """

        if (
            settings.max_turns
            and state.turns >= settings.max_turns
            and state.status
            not in (ConversationStatus.COMPLETED, ConversationStatus.CANCELLED)
        ):
            state.reset()
            state.status = ConversationStatus.CANCELLED
        self._store.save(state)
        return ConversationResult(state=state, reply=reply, action=action)


def _matches(text: str, vocabulary: set[str]) -> bool:
    return bool(set(re.findall(r"\w+", text.lower())) & vocabulary)


@lru_cache(maxsize=1)
def get_engine() -> ConversationEngine:
    """Return the process-wide singleton engine."""

    return ConversationEngine()
