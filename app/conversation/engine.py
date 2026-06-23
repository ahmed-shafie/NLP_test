"""Slot-filling dialogue engine: drive a money transfer to confirmation over turns."""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal

from app.config import DEFAULT_CURRENCY, settings
from app.conversation import templates
from app.conversation.state import (
    BILL_REQUIRED_SLOTS,
    ConversationState,
    ConversationStatus,
)
from app.conversation.store import get_session_store
from app.memory.schemas import Shortcut
from app.memory.service import get_memory_brain
from app.nlu import entities, pipeline
from app.nlu.lang import detect_language
from app.schemas import (
    BillPaymentRequest,
    Intent,
    Language,
    NLUResponse,
    TransferRequest,
)
from app.trace import BlockTracer

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
# Leading words that mean "delete the named shortcut" (e.g. "forget laila").
_FORGET = {"forget", "remove", "delete", "احذف", "امسح"}

# Tokens that pick a flow when the assistant asks "Transfer or Pay a bill?".
_CHOOSE_TRANSFER = {"1", "١", "transfer", "send", "تحويل", "حوالة", "حول", "حوّل"}
_CHOOSE_BILL = {"2", "٢", "bill", "bills", "فاتورة", "فواتير", "دفع"}


def _tokens(text: str) -> set[str]:
    return {t.strip(".,!؟،").lower() for t in text.split()}


def _matches(text: str, vocabulary: set[str]) -> bool:
    return bool(_tokens(text) & vocabulary)


def _forget_target(text: str) -> str | None:
    """If the message is 'forget <name>', return <name>; otherwise ``None``."""

    words = text.split()
    if len(words) < 2:
        return None
    if words[0].strip(".,!؟،").lower() not in _FORGET:
        return None
    return " ".join(words[1:]).strip(".,!؟، ")


def _parse_choice(text: str) -> Intent | None:
    """Interpret a reply to the Transfer/Pay-bill menu, if it is one."""

    tokens = _tokens(text)
    if tokens & _CHOOSE_BILL:
        return Intent.PAY_BILL
    if tokens & _CHOOSE_TRANSFER:
        return Intent.TRANSFER_MONEY
    return None


class ConversationResult:
    """Outcome of a single turn (kept plain so the API layer maps it to a schema)."""

    def __init__(
        self,
        state: ConversationState,
        reply: str,
        transfer: TransferRequest | None = None,
        bill: BillPaymentRequest | None = None,
        block_trace: list | None = None,
    ) -> None:
        self.state = state
        self.reply = reply
        self.transfer = transfer
        self.bill = bill
        self.block_trace = block_trace or []


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
        tracer = BlockTracer()
        sid = session_id or uuid.uuid4().hex
        # Memory Brain restores the conversation state (slots, FSM status) across turns.
        with tracer.block("memory_restore") as span:
            loaded = self._store.load(sid)
            if loaded is None:
                span.annotate("new session")
            state = loaded or ConversationState(session_id=sid)
        state.turns += 1
        if user_id:
            state.user_id = user_id
        lang = language or detect_language(text)
        state.language = lang

        # A fresh utterance after a finished dialogue starts a new transfer.
        if state.status in (ConversationStatus.COMPLETED, ConversationStatus.CANCELLED):
            state.reset()
            state.language = lang

        forget_name = _forget_target(text)
        if forget_name:
            with tracer.block("orchestrator"):
                result = self._forget_shortcut(state, forget_name, lang)
        elif _matches(text, _CANCEL):
            with tracer.block("orchestrator"):
                state.reset()
                state.status = ConversationStatus.CANCELLED
                result = self._finish(state, templates.cancelled(lang))
        elif state.status is ConversationStatus.CONFIRMING:
            result = self._handle_confirmation(state, text, lang, tracer)
        elif state.status is ConversationStatus.SELECTING:
            result = self._handle_selection(state, text, lang, tracer)
        else:
            result = self._collect(state, text, lang, tracer)

        # All blocks have closed; attach the completed trace to the turn's result.
        result.block_trace = list(tracer.entries)
        return result

    # ------------------------------------------------------------------ #

    def _handle_confirmation(
        self,
        state: ConversationState,
        text: str,
        lang: Language,
        tracer: BlockTracer,
    ) -> ConversationResult:
        with tracer.block("orchestrator"):
            if _matches(text, _NEGATIVE):
                state.reset()
                state.status = ConversationStatus.CANCELLED
                return self._finish(state, templates.cancelled(lang))
            if _matches(text, _AFFIRM):
                return self._complete(state, lang, tracer)
            # Unrecognised reply: re-ask for confirmation.
            return self._finish(state, self._confirm_text(state, lang))

    def _handle_selection(
        self,
        state: ConversationState,
        text: str,
        lang: Language,
        tracer: BlockTracer,
    ) -> ConversationResult:
        """Resolve a reply to the Transfer/Pay-bill menu, then ask the first slot."""

        with tracer.block("orchestrator"):
            choice = _parse_choice(text)
            if choice is None:
                # Couldn't tell which one — ask again.
                return self._finish(state, templates.choose_action(lang))
            state.intent = choice
            state.status = ConversationStatus.COLLECTING
            # The menu reply itself carries no slots; ask for the first one.
            required = self._required_slots(choice)
            if choice is Intent.PAY_BILL:
                self._apply_bill_defaults(state)
            else:
                self._apply_memory_defaults(state, text)
            missing = state.slots.first_missing_required(required)
            if missing is None:
                state.pending_slot = None
                state.status = ConversationStatus.CONFIRMING
                return self._finish(state, self._confirm_text(state, lang))
            state.pending_slot = missing
            return self._finish(
                state, templates.slot_prompt(missing, lang, choice)
            )

    @staticmethod
    def _required_slots(intent: Intent | None) -> tuple[str, ...]:
        if intent is Intent.PAY_BILL:
            return BILL_REQUIRED_SLOTS
        return ("amount", "currency", "recipient")

    def _decide_action(
        self,
        state: ConversationState,
        text: str,
        lang: Language,
        parsed: NLUResponse,
    ) -> Intent | None:
        """Pick the flow for a new dialogue, or ``None`` when it's unclear.

        Bill signals (biller keyword / "bill" / "فاتورة") win; otherwise a transfer
        signal (transfer intent, a matched shortcut, or a recipient) selects the
        transfer flow. When neither is present we ask the customer to choose.
        """

        bills = entities.extract_bill_entities(text, lang)
        if (
            parsed.intent is Intent.PAY_BILL
            or bills.biller is not None
            or entities.has_bill_word(text)
        ):
            return Intent.PAY_BILL
        if parsed.intent is Intent.TRANSFER_MONEY:
            return Intent.TRANSFER_MONEY
        if state.intent is Intent.TRANSFER_MONEY:  # set by a matched shortcut
            return Intent.TRANSFER_MONEY
        if entities.extract_recipient(text, lang):
            return Intent.TRANSFER_MONEY
        return None

    def _collect(
        self,
        state: ConversationState,
        text: str,
        lang: Language,
        tracer: BlockTracer,
    ) -> ConversationResult:
        parsed = pipeline.parse(text, lang)
        # Fold the NLU pipeline's own per-block trace into this turn's trace.
        tracer.extend(parsed.block_trace)

        with tracer.block("orchestrator"):
            # Memory Brain: a saved shortcut ("pay rent") pre-fills the template.
            self._apply_shortcut(state, text)

            # Smart chooser: determine transfer vs bill, or ask the customer.
            if state.intent not in (Intent.TRANSFER_MONEY, Intent.PAY_BILL):
                action = self._decide_action(state, text, lang, parsed)
                if action is None:
                    state.status = ConversationStatus.SELECTING
                    return self._finish(state, templates.choose_action(lang))
                state.intent = action

            if state.intent is Intent.PAY_BILL:
                return self._collect_bill(state, text, lang)
            return self._collect_transfer(state, text, lang, parsed)

    def _collect_transfer(
        self,
        state: ConversationState,
        text: str,
        lang: Language,
        parsed: NLUResponse,
    ) -> ConversationResult:
        slots = state.slots

        # Merge newly extracted slots (never overwrite an already-filled slot).
        ent = parsed.entities
        if slots.amount is None and ent.amount is not None:
            slots.amount = ent.amount
        # Only adopt a currency the user *explicitly* stated this turn; the
        # generic USD default is deferred below so a learned habit currency
        # can win first.
        explicit_currency = entities.extract_currency(text)
        if not slots.currency and explicit_currency:
            slots.currency = explicit_currency
        if not slots.recipient and ent.recipient:
            slots.recipient = ent.recipient
        if not slots.source_account and ent.source_account:
            slots.source_account = ent.source_account

        # If we just asked for a specific slot and parsing didn't fill it,
        # interpret the bare answer (e.g. "Ahmed" for recipient).
        if state.pending_slot:
            self._fill_pending_from_raw(state, text, lang)

        # Memory Brain: fall back to learned habits for any still-empty slot
        # ("send my usual" -> favourite recipient; default currency / source).
        self._apply_memory_defaults(state, text)

        # Generic currency default (mirrors the parser) after habits had a say.
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

    def _collect_bill(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        slots = state.slots

        # First interpret a bare answer to the slot we just asked for.
        if state.pending_slot:
            self._fill_pending_bill(state, text, lang)

        # Then merge anything else stated this turn (without overwriting).
        bills = entities.extract_bill_entities(text, lang)
        if not slots.biller and bills.biller:
            slots.biller = bills.biller
            slots.biller_category = bills.biller_category
        if not slots.reference_number and bills.reference_number:
            slots.reference_number = bills.reference_number
        if slots.amount is None and bills.amount is not None:
            slots.amount = bills.amount
        if not slots.currency and bills.currency:
            slots.currency = bills.currency

        # Learned default currency, then the generic fallback once amount is known.
        self._apply_bill_defaults(state)
        if slots.currency is None and slots.amount is not None:
            slots.currency = DEFAULT_CURRENCY

        missing = slots.first_missing_required(BILL_REQUIRED_SLOTS)
        if missing is not None:
            state.pending_slot = missing
            state.status = ConversationStatus.COLLECTING
            return self._finish(
                state, templates.slot_prompt(missing, lang, Intent.PAY_BILL)
            )

        state.pending_slot = None
        state.status = ConversationStatus.CONFIRMING
        return self._finish(state, self._confirm_text(state, lang))

    def _fill_pending_bill(
        self, state: ConversationState, text: str, lang: Language
    ) -> None:
        slot = state.pending_slot
        slots = state.slots
        if slot == "biller" and not slots.biller:
            biller, category = entities.extract_biller(text, lang)
            slots.biller = biller or text.strip(" .,،؟?")
            slots.biller_category = category
        elif slot == "reference_number" and not slots.reference_number:
            ref = entities.extract_reference_number(text)
            if ref:
                slots.reference_number = ref
        elif slot == "amount" and slots.amount is None:
            amount = entities.extract_amount(text)
            if amount is not None:
                slots.amount = amount
        elif slot == "currency" and not slots.currency:
            currency = entities.extract_currency(text)
            if currency is not None:
                slots.currency = currency

    def _apply_bill_defaults(self, state: ConversationState) -> None:
        brain = self._memory(state)
        if brain is None:
            return
        slots = state.slots
        if not slots.currency:
            currency = brain.default_currency(state.user_id)
            if currency:
                slots.currency = currency

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
        # A recipient-only alias ("send to mona") reuses the remembered amount
        # from that recipient's template alias when the customer gave none.
        if slots.amount is None and slots.recipient:
            template = brain.template_for_recipient(uid, slots.recipient)
            if template is not None:
                slots.amount = template.amount
                if not slots.currency and template.currency:
                    slots.currency = template.currency
        if not slots.currency:
            currency = brain.default_currency(uid)
            if currency:
                slots.currency = currency
        if not slots.source_account:
            source = brain.default_source_account(uid)
            if source:
                slots.source_account = source

    def _complete(
        self, state: ConversationState, lang: Language, tracer: BlockTracer
    ) -> ConversationResult:
        if state.intent is Intent.PAY_BILL:
            return self._complete_bill(state, lang)
        return self._complete_transfer(state, lang)

    def _complete_transfer(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult:
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
        created = self._learn(state, transfer)
        reply = templates.completed(
            self._fmt_amount(transfer.amount),
            transfer.currency,
            transfer.recipient,
            lang,
        )
        for shortcut in created:
            reply = f"{reply} {templates.alias_created(shortcut.name, lang)}"
        return self._finish(state, reply, transfer=transfer)

    def _complete_bill(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult:
        slots = state.slots
        payment, _missing, errors = pipeline.validate_bill_payment(
            biller=slots.biller,
            reference_number=slots.reference_number,
            amount=slots.amount,
            currency=slots.currency,
            biller_category=slots.biller_category,
            note=slots.note,
        )
        if payment is None:
            # Validation failed (e.g. unsupported currency): re-collect that slot.
            state.status = ConversationStatus.COLLECTING
            field = errors[0].field if errors else "currency"
            state.pending_slot = field
            slots.currency = None if field == "currency" else slots.currency
            return self._finish(
                state, templates.slot_prompt(field, lang, Intent.PAY_BILL)
            )

        state.status = ConversationStatus.COMPLETED
        state.pending_slot = None
        reply = templates.bill_completed(
            self._fmt_amount(payment.amount),
            payment.currency,
            payment.biller,
            payment.reference_number,
            lang,
        )
        return self._finish(state, reply, bill=payment)

    def _learn(
        self, state: ConversationState, transfer: TransferRequest
    ) -> list[Shortcut]:
        brain = self._memory(state)
        if brain is None:
            return []
        try:
            return brain.learn_from_transfer(state.user_id, transfer)
        except Exception:  # noqa: BLE001 - learning must never break a transfer
            logger.warning("Memory Brain failed to learn from transfer", exc_info=True)
            return []

    def _forget_shortcut(
        self, state: ConversationState, name: str, lang: Language
    ) -> ConversationResult:
        brain = self._memory(state)
        removed = False
        if brain is not None:
            try:
                removed = brain.delete_shortcut(state.user_id, name)
            except Exception:  # noqa: BLE001 - never break the turn on a delete
                logger.warning("Memory Brain failed to delete shortcut", exc_info=True)
        reply = (
            templates.alias_forgotten(name, lang)
            if removed
            else templates.alias_not_found(name, lang)
        )
        return self._finish(state, reply)

    def _confirm_text(self, state: ConversationState, lang: Language) -> str:
        slots = state.slots
        if state.intent is Intent.PAY_BILL:
            return templates.bill_confirm_prompt(
                self._fmt_amount(slots.amount),
                slots.currency or "",
                slots.biller or "",
                slots.reference_number or "",
                lang,
            )
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
        bill: BillPaymentRequest | None = None,
    ) -> ConversationResult:
        self._store.save(state)
        return ConversationResult(state, reply, transfer=transfer, bill=bill)


_engine: ConversationEngine | None = None


def get_engine() -> ConversationEngine:
    global _engine
    if _engine is None:
        _engine = ConversationEngine()
    return _engine
