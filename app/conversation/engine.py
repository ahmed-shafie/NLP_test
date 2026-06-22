"""Slot-filling dialogue engine: drive a money transfer to confirmation over turns."""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal

from app.config import DEFAULT_CURRENCY, settings
from app.conversation import templates
from app.conversation.state import ConversationState, ConversationStatus
from app.conversation.store import get_session_store
from app.memory.schemas import Shortcut
from app.memory.service import get_memory_brain
from app.nlu import entities, pipeline
from app.nlu.lang import detect_language
from app.schemas import Intent, Language, TransferRequest

logger = logging.getLogger(__name__)

_AFFIRM = {
    "yes",
    "y",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "confirm",
    "correct",
    "go",
    "proceed",
    "نعم",
    "ايوه",
    "أيوه",
    "تمام",
    "اكد",
    "أكد",
    "موافق",
    "صح",
    "اوك",
}
_NEGATIVE = {
    "no",
    "n",
    "nope",
    "cancel",
    "wrong",
    "stop",
    "لا",
    "كلا",
    "خطأ",
    "غلط",
}
_CANCEL = {
    "cancel",
    "stop",
    "quit",
    "exit",
    "nevermind",
    "الغاء",
    "إلغاء",
    "توقف",
    "خروج",
}


def _tokens(text: str) -> set[str]:
    return {t.strip(".,!؟،").lower() for t in text.split()}


def _matches(text: str, vocabulary: set[str]) -> bool:
    return bool(_tokens(text) & vocabulary)


class ConversationResult:
    """Outcome of a single turn (kept plain so the API layer maps it to a schema)."""

    def __init__(
        self,
        state: ConversationState,
        reply: str,
        transfer: TransferRequest | None = None,
    ) -> None:
        self.state = state
        self.reply = reply
        self.transfer = transfer


class ConversationEngine:
    def __init__(self) -> None:
        self._store = get_session_store()

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
        lang = language or detect_language(text)
        state.language = lang

        # A fresh utterance after a finished dialogue starts a new transfer.
        if state.status in (ConversationStatus.COMPLETED, ConversationStatus.CANCELLED):
            state.reset()
            state.language = lang

        if _matches(text, _CANCEL):
            state.reset()
            state.status = ConversationStatus.CANCELLED
            return self._finish(state, templates.cancelled(lang))

        if state.status is ConversationStatus.CONFIRMING:
            return self._handle_confirmation(state, text, lang)

        return self._collect(state, text, lang)

    # ------------------------------------------------------------------ #

    def _handle_confirmation(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        if _matches(text, _NEGATIVE):
            state.reset()
            state.status = ConversationStatus.CANCELLED
            return self._finish(state, templates.cancelled(lang))
        if _matches(text, _AFFIRM):
            return self._complete(state, lang)
        # Unrecognised reply: re-ask for confirmation.
        return self._finish(state, self._confirm_text(state, lang))

    def _collect(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        parsed = pipeline.parse(text, lang)
        slots = state.slots

        # Memory Brain: a saved shortcut ("pay rent") pre-fills the transfer template.
        self._apply_shortcut(state, text)

        if parsed.intent is Intent.TRANSFER_MONEY:
            state.intent = Intent.TRANSFER_MONEY

        if state.intent is not Intent.TRANSFER_MONEY:
            return self._finish(state, templates.fallback(lang))

        # Merge any newly extracted slots (never overwrite an already-filled slot).
        ent = parsed.entities
        if slots.amount is None and ent.amount is not None:
            slots.amount = ent.amount
        # Only adopt a currency the user *explicitly* stated this turn; the generic
        # USD default is deferred below so a learned habit currency can win first.
        explicit_currency = entities.extract_currency(text)
        if not slots.currency and explicit_currency:
            slots.currency = explicit_currency
        if not slots.recipient and ent.recipient:
            slots.recipient = ent.recipient
        if not slots.source_account and ent.source_account:
            slots.source_account = ent.source_account

        # If we just asked for a specific slot and parsing didn't fill it, interpret
        # the bare answer (e.g. "Ahmed" for recipient, "five" is out of scope).
        if state.pending_slot:
            self._fill_pending_from_raw(state, text, lang)

        # Memory Brain: fall back to learned habits for any still-empty slot
        # ("send my usual" -> favourite recipient; default currency / source account).
        self._apply_memory_defaults(state, text)

        # Generic currency default (mirrors the parser) once habits have had a say.
        if slots.currency is None and slots.amount is not None:
            slots.currency = DEFAULT_CURRENCY

        missing = slots.first_missing_required()
        if missing is not None:
            state.pending_slot = missing
            state.status = ConversationStatus.COLLECTING
            return self._finish(state, templates.slot_prompt(missing, lang))

        state.pending_slot = None
        state.status = ConversationStatus.CONFIRMING
        return self._finish(state, self._confirm_text(state, lang))

    def _fill_pending_from_raw(
        self, state: ConversationState, text: str, lang: Language
    ) -> None:
        slot = state.pending_slot
        slots = state.slots
        if slot == "amount" and slots.amount is None:
            amount = entities.extract_amount(text)
            if amount is not None:
                slots.amount = amount
        elif slot == "currency" and not slots.currency:
            currency = entities.extract_currency(text)
            if currency is not None:
                slots.currency = currency
        elif slot == "recipient" and not slots.recipient:
            candidate = entities.extract_recipient(text, lang) or text.strip(" .,،؟?")
            if candidate:
                slots.recipient = candidate

    def _memory(self, state: ConversationState):
        """Return the Memory Brain when it is enabled and scoped to a user."""

        if not settings.memory_enabled or not state.user_id:
            return None
        return get_memory_brain()

    def _apply_shortcut(self, state: ConversationState, text: str) -> None:
        brain = self._memory(state)
        if brain is None:
            return
        # Only expand a shortcut at the start of a transfer, not mid slot-filling.
        if state.pending_slot is not None:
            return
        shortcut = brain.resolve_shortcut(state.user_id, text)
        if shortcut is None:
            return
        state.intent = Intent.TRANSFER_MONEY
        self._fill_from_shortcut(state, shortcut)

    @staticmethod
    def _fill_from_shortcut(state: ConversationState, shortcut: Shortcut) -> None:
        slots = state.slots
        if slots.amount is None and shortcut.amount is not None:
            slots.amount = shortcut.amount
        if not slots.currency and shortcut.currency:
            slots.currency = shortcut.currency
        if not slots.recipient and shortcut.recipient:
            slots.recipient = shortcut.recipient
        if not slots.source_account and shortcut.source_account:
            slots.source_account = shortcut.source_account
        if not slots.note and shortcut.note:
            slots.note = shortcut.note

    def _apply_memory_defaults(self, state: ConversationState, text: str) -> None:
        brain = self._memory(state)
        if brain is None:
            return
        uid = state.user_id
        slots = state.slots
        if not slots.recipient and brain.wants_usual_recipient(text):
            favorite = brain.favorite_recipient(uid)
            if favorite:
                slots.recipient = favorite
        if not slots.currency:
            currency = brain.default_currency(uid)
            if currency:
                slots.currency = currency
        if not slots.source_account:
            source = brain.default_source_account(uid)
            if source:
                slots.source_account = source

    def _complete(self, state: ConversationState, lang: Language) -> ConversationResult:
        slots = state.slots
        result = pipeline.validate_transfer(
            amount=slots.amount,
            currency=slots.currency,
            recipient=slots.recipient,
            source_account=slots.source_account,
            note=slots.note,
        )
        if not result.valid or result.transfer is None:
            # Validation failed (e.g. unsupported currency): re-collect that slot.
            state.status = ConversationStatus.COLLECTING
            field = result.errors[0].field if result.errors else "currency"
            state.pending_slot = field
            slots.currency = None if field == "currency" else slots.currency
            return self._finish(state, templates.slot_prompt(field, lang))

        state.status = ConversationStatus.COMPLETED
        state.pending_slot = None
        transfer = result.transfer
        self._learn(state, transfer)
        reply = templates.completed(
            self._fmt_amount(transfer.amount),
            transfer.currency,
            transfer.recipient,
            lang,
        )
        return self._finish(state, reply, transfer)

    def _learn(self, state: ConversationState, transfer: TransferRequest) -> None:
        brain = self._memory(state)
        if brain is None:
            return
        try:
            brain.learn_from_transfer(state.user_id, transfer)
        except Exception:  # noqa: BLE001 - learning must never break a transfer
            logger.warning("Memory Brain failed to learn from transfer", exc_info=True)

    def _confirm_text(self, state: ConversationState, lang: Language) -> str:
        slots = state.slots
        return templates.confirm_prompt(
            self._fmt_amount(slots.amount),
            slots.currency or "",
            slots.recipient or "",
            lang,
        )

    @staticmethod
    def _fmt_amount(amount: Decimal | None) -> str:
        if amount is None:
            return ""
        normalized = amount.normalize()
        return f"{normalized:f}"

    def _finish(
        self,
        state: ConversationState,
        reply: str,
        transfer: TransferRequest | None = None,
    ) -> ConversationResult:
        self._store.save(state)
        return ConversationResult(state, reply, transfer)


_engine: ConversationEngine | None = None


def get_engine() -> ConversationEngine:
    global _engine
    if _engine is None:
        _engine = ConversationEngine()
    return _engine
