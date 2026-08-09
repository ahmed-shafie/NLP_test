"""Slot-filling dialogue engine: drive a money transfer to confirmation over turns."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from decimal import Decimal

from app import banking_core_client
from app.config import DEFAULT_CURRENCY, settings
from app.conversation import moderation, templates
from app.conversation.moderation import ModerationResult
from app.conversation.state import (
    BILL_REQUIRED_SLOTS,
    BeneficiaryOption,
    BillerOption,
    ConversationSlots,
    ConversationState,
    ConversationStatus,
)
from app.conversation.store import get_session_store
from app.data_loader import (
    BillerRecord,
    biller_categories,
    canonicalize_recipient,
    resolve_biller_by_code,
    resolve_biller_candidates,
)
from app.db.directory import BeneficiaryHit, get_beneficiary_directory
from app.memory.schemas import Shortcut
from app.memory.service import get_memory_brain
from app.nlu import accounts, entities, pipeline
from app.nlu.lang import detect_language
from app.nlu.normalize import normalize, normalize_digits
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
# Filler tokens allowed alongside a bare menu pick ("option 2", "let's transfer").
_CHOICE_FILLERS = {
    "please",
    "the",
    "a",
    "option",
    "number",
    "i",
    "want",
    "to",
    "choose",
    "pick",
    "let",
    "me",
    "it",
    "s",
}


# Cues that a message is a balance inquiry ("what's my balance", "كم رصيدي").
_BALANCE_CUES = {"balance", "رصيد", "رصيدي", "الرصيد"}
_BALANCE_PHRASES = ("how much do i have", "كم عندي", "كم لدي")

# Map account-type words (EN/AR) to a canonical type used by the Banking Core API.
_ACCOUNT_TYPES: dict[str, str] = {
    "current": "current",
    "checking": "current",
    "جاري": "current",
    "الجاري": "current",
    "savings": "savings",
    "saving": "savings",
    "توفير": "savings",
    "التوفير": "savings",
    "credit": "credit",
    "ائتمان": "credit",
    "salary": "salary",
    "راتب": "salary",
    "الراتب": "salary",
}


def _account_type(text: str) -> str | None:
    """Return a canonical source-account type mentioned in ``text``, if any."""

    tokens = _tokens(text)
    for word, canonical in _ACCOUNT_TYPES.items():
        if word in tokens:
            return canonical
    return None


def _is_balance_inquiry(text: str) -> bool:
    tokens = _tokens(text)
    if tokens & _BALANCE_CUES:
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in _BALANCE_PHRASES)


# Cues that a message asks to *list* saved beneficiaries ("show my beneficiaries",
# "من المستفيدين عندي"). Normalized so Arabic letter-form variants collapse. A
# beneficiary noun plus a list/possessive marker distinguishes "show my
# beneficiaries" from "send money to a beneficiary".
_LIST_BENE_NOUNS = {
    normalize(w)
    for w in (
        "beneficiaries",
        "beneficiary",
        "payees",
        "payee",
        "المستفيدين",
        "مستفيدين",
        "المستفيدون",
        "المستفيد",
        "مستفيد",
        "مستفيدي",
        # common misspelling that drops the yaa (المستف[ي]دين)
        "المستفدين",
        "مستفدين",
        "المستفدون",
    )
}
_LIST_BENE_MARKERS = {
    normalize(w)
    for w in (
        "list",
        "show",
        "view",
        "see",
        "display",
        "my",
        "all",
        "who",
        "which",
        "saved",
        "registered",
        "عرض",
        "اعرض",
        "اظهر",
        "وريني",
        "ورني",
        "قائمة",
        "من",
        "مين",
        "كل",
        "عندي",
        "لدي",
        "المسجلين",
        "مسجلين",
        "اللي",
    )
}


# Verbs that turn a beneficiary noun into an *add* request rather than a list one.
_ADD_BENE_VERBS = {
    normalize(w)
    for w in (
        "add",
        "adding",
        "save",
        "store",
        "register",
        "create",
        "new",
        "اضف",
        "أضف",
        "اضافة",
        "إضافة",
        "اضيف",
        "أضيف",
        "ضيف",
        "سجل",
        "احفظ",
        "جديد",
        "جديدا",
        "جديدة",
    )
}


# Dropped when reading a name out of "add Sara Ali as a beneficiary".
_ADD_REQUEST_STOPWORDS = (
    _ADD_BENE_VERBS
    | _LIST_BENE_NOUNS
    | {
        normalize(w)
        for w in (
            "a",
            "an",
            "the",
            "as",
            "my",
            "please",
            "i",
            "want",
            "need",
            "would",
            "like",
            "to",
            "for",
            "recipient",
            "contact",
            "ابغى",
            "ابغي",
            "اريد",
            "أريد",
            "عايز",
            "لو",
            "سمحت",
            "رجاء",
            "من",
            "فضلك",
            "عندي",
            "لدي",
            "باسم",
            "كمستفيد",
        )
    }
)


def _is_add_beneficiary(text: str) -> bool:
    """True for "add a beneficiary" style requests (checked before listing)."""

    tokens = {normalize(t) for t in _tokens(text)}
    if not tokens & _LIST_BENE_NOUNS:
        return False
    return bool(tokens & _ADD_BENE_VERBS)


def _is_list_beneficiaries(text: str) -> bool:
    tokens = {normalize(t) for t in _tokens(text)}
    if not tokens & _LIST_BENE_NOUNS:
        return False
    if tokens & _ADD_BENE_VERBS:  # "add a beneficiary" is not a listing request
        return False
    return bool(tokens & _LIST_BENE_MARKERS)


# An account-shaped message: digits/separators behind at most a country code
# ("SA03 8000 …"). Its letters say nothing about the customer's language.
_ACCOUNT_SHAPED = re.compile(r"^[A-Za-z]{0,2}[\d\s\u00a0-]+$")


def _carries_language_signal(text: str) -> bool:
    """False for messages detection can't judge: bare numbers, IBANs, "2"."""

    stripped = text.strip(" .,،؟?")
    if not re.search(r"[^\W\d_]", stripped, re.UNICODE):
        return False
    return not _ACCOUNT_SHAPED.match(stripped)


def _mask_account(account: str) -> str:
    """Show only the last four characters of an account/IBAN."""

    tail = account[-4:] if len(account) >= 4 else account
    return f"SA••{tail}"


def _tokens(text: str) -> set[str]:
    return {t.strip(".,!؟،").lower() for t in text.split()}


def _matches(text: str, vocabulary: set[str]) -> bool:
    return bool(_tokens(text) & vocabulary)


# Request verbs / fillers (normalized) that leak into a bare recipient answer
# such as "ابغي احمد" ("I want Ahmed") or "send to Ahmed". Stripped before the
# remainder is canonicalized to a name.
_RECIPIENT_FILLERS = {
    # Arabic colloquial + MSA verbs and prepositions
    "ابغي",
    "ابي",
    "اريد",
    "عايز",
    "عاوز",
    "احب",
    "حول",
    "احول",
    "ارسل",
    "ابعت",
    "ادفع",
    "الى",
    "ل",
    "مبلغ",
    "بمبلغ",
    # English
    "i",
    "want",
    "to",
    "send",
    "transfer",
    "pay",
    "wire",
    "remit",
    "please",
    "the",
    "a",
    "an",
    "money",
    "cash",
    "some",
    "for",
}


def _clean_recipient_answer(text: str) -> str | None:
    """Turn a free-text recipient reply into a name.

    Drops leading/embedded request verbs and prepositions (so "ابغي احمد" →
    "احمد", "send to Ahmed" → "Ahmed"), then canonicalizes the remaining tokens
    against the name gazetteer. Returns ``None`` when nothing name-like is left.
    """

    stripped = text.strip(" .,،؟?")
    if not stripped:
        return None
    kept = [tok for tok in stripped.split() if normalize(tok) not in _RECIPIENT_FILLERS]
    candidate = " ".join(kept).strip() or stripped
    return canonicalize_recipient(candidate) or None


def _recipient_from_answer(text: str, lang: Language) -> str | None:
    """Best-effort recipient from a slot answer: surface pattern, else cleanup."""

    return entities.extract_recipient(text, lang) or _clean_recipient_answer(text)


def _display_name(name: str, name_ar: str | None, lang: Language) -> str:
    """Prefer the Arabic beneficiary name in Arabic conversations (else English)."""

    if lang is Language.AR and name_ar:
        return name_ar
    return name


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
        flagged_terms: list[str] | None = None,
    ) -> None:
        self.state = state
        self.reply = reply
        self.transfer = transfer
        self.bill = bill
        self.block_trace = block_trace or []
        self.flagged_terms = flagged_terms or []


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
        lang = self._effective_language(text, language, loaded)
        state.language = lang

        # A fresh utterance after a finished dialogue starts a new transfer.
        if state.status in (ConversationStatus.COMPLETED, ConversationStatus.CANCELLED):
            state.reset()
            state.language = lang

        # Content moderation: abusive ("ribald") input is refused with a calm,
        # professional redirect and never processed as a slot. Runs in every
        # state so it also catches abuse mid-flow (confirming/disambiguating).
        flagged = moderation.detect(text)
        if flagged.flagged:
            with tracer.block("moderation") as span:
                span.annotate(f"{flagged.severity}:{len(flagged.terms)}")
                result = self._handle_inappropriate(state, flagged, lang)
            result.block_trace = list(tracer.entries)
            return result

        forget_name = _forget_target(text)
        if forget_name:
            with tracer.block("orchestrator"):
                result = self._forget_shortcut(state, forget_name, lang)
        elif _matches(text, _CANCEL):
            with tracer.block("orchestrator"):
                state.reset()
                state.status = ConversationStatus.CANCELLED
                result = self._finish(state, templates.cancelled(lang))
        elif self._is_mid_transaction(state) and _is_balance_inquiry(text):
            # Allowed "aside": answer a balance question in the middle of a
            # transfer/bill, then re-emit the current prompt so the flow
            # resumes untouched (status and slots are preserved).
            with tracer.block("orchestrator") as span:
                span.annotate("balance_aside")
                result = self._answer_balance_aside(state, text, lang)
        elif state.status is ConversationStatus.CONFIRMING:
            result = self._handle_confirmation(state, text, lang, tracer)
        elif state.status is ConversationStatus.DISAMBIGUATING:
            result = self._handle_disambiguation(state, text, lang, tracer)
        elif state.status is ConversationStatus.SELECTING:
            result = self._handle_selection(state, text, lang, tracer)
        else:
            result = self._collect(state, text, lang, tracer)

        # All blocks have closed; attach the completed trace to the turn's result.
        result.block_trace = list(tracer.entries)
        return result

    # ------------------------------------------------------------------ #

    @staticmethod
    def _effective_language(
        text: str, override: Language | None, prior: ConversationState | None
    ) -> Language:
        """Pick the reply language, keeping the conversation's language sticky.

        An explicit override wins. Otherwise detect from the text, but when the
        message carries no language signal — a bare "2", an account number, or
        an IBAN whose only letters are its country code — detection is
        meaningless, so inherit the ongoing conversation's language instead of
        defaulting to English.
        """

        if override is not None:
            return override
        if prior is not None and not _carries_language_signal(text):
            return prior.language
        return detect_language(text)

    def _handle_confirmation(
        self,
        state: ConversationState,
        text: str,
        lang: Language,
        tracer: BlockTracer,
    ) -> ConversationResult:
        with tracer.block("orchestrator"):
            if _matches(text, _NEGATIVE):
                if state.pending_add_account:
                    return self._abandon_add_beneficiary(state, lang)
                state.reset()
                state.status = ConversationStatus.CANCELLED
                return self._finish(state, templates.cancelled(lang))
            if _matches(text, _AFFIRM):
                if state.pending_add_account:
                    return self._commit_add_beneficiary(state, lang, tracer)
                return self._complete(state, lang, tracer)
            # Unrecognised reply: re-ask for confirmation.
            return self._finish(state, self._confirm_text(state, lang))

    def _handle_inappropriate(
        self,
        state: ConversationState,
        flagged: ModerationResult,
        lang: Language,
    ) -> ConversationResult:
        """Refuse an abusive turn with a varied, professional redirect.

        The in-progress flow (status/slots) is preserved so a follow-up clean
        message continues where it left off. After ``moderation_max_strikes``
        flagged turns the session is ended.
        """

        state.flagged_count += 1
        terms = list(flagged.terms)
        if state.flagged_count >= settings.moderation_max_strikes:
            state.reset()
            state.flagged_count = 0
            state.status = ConversationStatus.CANCELLED
            return self._finish(
                state, templates.repeat_offense(lang), flagged_terms=terms
            )

        severity = flagged.severity or "severe"
        group = f"inappropriate:{severity}:{lang.value}"
        reply, index = templates.inappropriate(
            lang, severity, flagged.terms, state.last_variant.get(group)
        )
        state.last_variant[group] = index
        return self._finish(state, reply, flagged_terms=terms)

    def _handle_selection(
        self,
        state: ConversationState,
        text: str,
        lang: Language,
        tracer: BlockTracer,
    ) -> ConversationResult:
        """Resolve a reply to the Transfer/Pay-bill menu, then ask the first slot."""

        choice = _parse_choice(text)
        bare = choice is not None and _tokens(text) <= (
            _CHOOSE_TRANSFER | _CHOOSE_BILL | _CHOICE_FILLERS
        )
        if not bare:
            # Not a plain "1"/"2" pick — re-run the smart collector so a full
            # request ("pay my electricity bill 778899") keeps its slots, and an
            # unrecognised reply ("maybe later") re-asks the menu.
            state.status = ConversationStatus.COLLECTING
            state.intent = None
            return self._collect(state, text, lang, tracer)

        with tracer.block("orchestrator"):
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
                if choice is Intent.TRANSFER_MONEY:
                    pending = self._resolve_beneficiary(state, lang)
                    if pending is not None:
                        return pending
                return self._enter_confirmation(state, lang)
            state.pending_slot = missing
            return self._finish(state, templates.slot_prompt(missing, lang, choice))

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
        # Mid "add beneficiary" flow: this turn answers the name or the account.
        if state.intent is Intent.ADD_BENEFICIARY or state.pending_add_name:
            with tracer.block("orchestrator"):
                return self._continue_add_beneficiary(state, text, lang)

        # "Add a beneficiary" as a request of its own (deterministic cue first,
        # so the classifier can't read it as a listing request).
        if (
            state.intent is None
            and state.pending_slot is None
            and _is_add_beneficiary(text)
        ):
            with tracer.block("orchestrator") as span:
                span.annotate("add_beneficiary")
                return self._start_add_beneficiary(state, text, lang)

        # Balance inquiry (its own intent) when starting fresh — answered by the
        # external Banking Core API, not the slot-filling flow.
        if (
            state.intent is None
            and state.pending_slot is None
            and _is_balance_inquiry(text)
        ):
            with tracer.block("orchestrator") as span:
                span.annotate("balance_inquiry")
                return self._handle_balance_inquiry(state, text, lang)

        # "Show my beneficiaries" (read-only) when starting fresh — deterministic
        # cue check beats the semantic classifier, which otherwise reads the word
        # "beneficiary" as a transfer and wrongly asks for an amount.
        if (
            state.intent is None
            and state.pending_slot is None
            and _is_list_beneficiaries(text)
        ):
            with tracer.block("orchestrator") as span:
                span.annotate("list_beneficiaries")
                return self._handle_list_beneficiaries(state, lang)

        parsed = pipeline.parse(text, lang)
        # Fold the NLU pipeline's own per-block trace into this turn's trace.
        tracer.extend(parsed.block_trace)

        with tracer.block("orchestrator"):
            # Memory Brain: a saved shortcut ("pay rent") pre-fills the template.
            self._apply_shortcut(state, text)

            # Semantic safety net: the classifier may flag abuse the blocklist
            # missed (novel insults). Redirect without echoing any text.
            if settings.moderation_enabled and parsed.intent is Intent.INAPPROPRIATE:
                return self._handle_inappropriate(
                    state, ModerationResult(True, "severe"), lang
                )

            # Smart chooser: determine transfer vs bill, or ask the customer.
            if state.intent not in (Intent.TRANSFER_MONEY, Intent.PAY_BILL):
                action = self._decide_action(state, text, lang, parsed)
                # Semantic fallback for a "list my beneficiaries" phrasing the
                # deterministic cue check above didn't catch (read-only). Only
                # when nothing points at a concrete bill/transfer, so the
                # classifier can't hijack e.g. "ادفع فاتورة ...".
                if state.intent is None and action is None:
                    if parsed.intent is Intent.ADD_BENEFICIARY:
                        return self._start_add_beneficiary(state, text, lang)
                    if parsed.intent is Intent.LIST_BENEFICIARIES:
                        return self._handle_list_beneficiaries(state, lang)
                if action is None:
                    # Warm chit-chat reply for pure greetings/thanks; then wait
                    # in SELECTING so a follow-up choice/request is understood.
                    if templates.is_small_talk(text):
                        state.status = ConversationStatus.SELECTING
                        return self._finish(state, templates.small_talk(text, lang))
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
        if not slots.recipient:
            if state.pending_slot == "recipient":
                # We explicitly asked for the recipient, so the whole message is
                # the answer: trust the deterministic bare-answer parse over the
                # pipeline's free extraction (which can mangle colloquial input
                # like "ابغي احمد" into a garbled name).
                slots.recipient = _recipient_from_answer(text, lang) or ent.recipient
            elif ent.recipient:
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

        # Beneficiary check goes DIRECT to the database (not the API): 0 -> offer to
        # add, 1 -> lock and continue, many (shared first name) -> disambiguate.
        pending = self._resolve_beneficiary(state, lang)
        if pending is not None:
            return pending

        return self._enter_confirmation(state, lang)

    def _collect_bill(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        slots = state.slots

        # First interpret a bare answer to a non-biller slot we just asked for.
        if state.pending_slot and state.pending_slot != "biller":
            self._fill_pending_bill(state, text, lang)

        # Merge non-biller slots stated this turn (without overwriting).
        bills = entities.extract_bill_entities(text, lang, allow_semantic=True)
        if not slots.reference_number and bills.reference_number:
            slots.reference_number = bills.reference_number
        if slots.amount is None and bills.amount is not None:
            slots.amount = bills.amount
        if not slots.currency and bills.currency:
            slots.currency = bills.currency

        # Apply currency defaults now so they hold even if we pause to ask the
        # customer which biller they meant.
        self._apply_bill_defaults(state)
        if slots.currency is None and slots.amount is not None:
            slots.currency = DEFAULT_CURRENCY

        # Resolve the biller, asking the customer to choose if it's ambiguous.
        if not slots.biller:
            asked = self._resolve_bill_biller(state, text, lang)
            if asked is not None:
                return asked

        return self._advance_bill(state, lang)

    def _resolve_bill_biller(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult | None:
        """Fill the biller slot, or return a prompt asking which biller is meant.

        Returns a :class:`ConversationResult` (the disambiguation question) when
        the term matches several SADAD billers; otherwise sets the biller slot
        (from a single catalogue hit or free text) and returns ``None``.
        """

        slots = state.slots
        candidates = resolve_biller_candidates(text, allow_semantic=True)
        if len(candidates) > 1:
            return self._ask_biller_choice(state, candidates, lang)
        if len(candidates) == 1:
            self._set_biller(slots, candidates[0], lang)
            return None
        # A short number sent as the biller -> its SADAD code (returns the name).
        record = self._biller_from_code(state, text)
        if record is not None:
            self._set_biller(slots, record, lang)
            return None
        # No catalogue match. Only billers in the SADAD catalogue can be paid, so
        # rather than accepting free text we tell the customer the biller isn't
        # on the list and ask again. Stay silent when they named nothing yet
        # (e.g. they only answered the reference-number question).
        # Quote the customer's own wording back: when they were answering "which
        # bill?" the whole reply is the name, otherwise use the extracted span.
        named: str | None
        if state.pending_slot == "biller":
            named = text.strip(" .,،؟?")
        else:
            named, _, _ = entities.extract_biller(text, lang, allow_semantic=True)
        if not named:
            return None
        state.pending_slot = "biller"
        state.status = ConversationStatus.COLLECTING
        return self._finish(
            state,
            templates.biller_not_found(named, list(biller_categories()), lang),
        )

    @staticmethod
    def _set_biller(
        slots: ConversationSlots, record: BillerRecord, lang: Language
    ) -> None:
        name = record.name_ar if lang is Language.AR else record.name_en
        slots.biller = name or record.name_en
        slots.biller_category = record.category
        slots.biller_code = record.biller_code

    @staticmethod
    def _biller_from_code(state: ConversationState, text: str) -> BillerRecord | None:
        """Resolve a short numeric token in ``text`` to a biller via its code.

        Digit runs already consumed as the amount or reference number are skipped
        so a bill amount of 200 is never misread as biller code 200.
        """

        slots = state.slots
        used: set[str] = set()
        if slots.reference_number:
            used.add("".join(ch for ch in slots.reference_number if ch.isdigit()))
        if slots.amount is not None:
            used.add("".join(ch for ch in str(slots.amount) if ch.isdigit()))
        for run in re.findall(r"\d+", normalize_digits(text)):
            if len(run) > 3 or run in used:
                continue
            record = resolve_biller_by_code(run)
            if record is not None:
                return record
        return None

    def _ask_biller_choice(
        self,
        state: ConversationState,
        candidates: list[BillerRecord],
        lang: Language,
    ) -> ConversationResult:
        options = [
            BillerOption(
                code=rec.biller_code,
                name=(rec.name_ar if lang is Language.AR else rec.name_en)
                or rec.name_en,
                category=rec.category,
            )
            for rec in candidates
        ]
        state.biller_options = options
        state.pending_slot = "biller"
        state.status = ConversationStatus.DISAMBIGUATING
        return self._finish(
            state, templates.choose_biller([opt.name for opt in options], lang)
        )

    def _handle_disambiguation(
        self,
        state: ConversationState,
        text: str,
        lang: Language,
        tracer: BlockTracer,
    ) -> ConversationResult:
        """Resolve a reply to a "which one?" question (biller or beneficiary)."""

        if state.disambiguation_kind == "beneficiary":
            with tracer.block("orchestrator"):
                return self._handle_beneficiary_choice(state, text, lang)

        with tracer.block("orchestrator"):
            choice = self._match_biller_option(state.biller_options, text)
            if choice is None:
                # Unrecognised pick — re-ask with the same options.
                return self._finish(
                    state,
                    templates.choose_biller(
                        [opt.name for opt in state.biller_options], lang
                    ),
                )
            slots = state.slots
            slots.biller = choice.name
            slots.biller_category = choice.category
            slots.biller_code = choice.code
            state.biller_options = []
            state.pending_slot = None
            state.status = ConversationStatus.COLLECTING
            return self._advance_bill(state, lang)

    @staticmethod
    def _match_biller_option(
        options: list[BillerOption], text: str
    ) -> BillerOption | None:
        if not options:
            return None
        normalized = normalize(text)
        digits = "".join(ch for ch in normalize_digits(normalized) if ch.isdigit())
        if digits:
            # A zero-padded or 3-digit number is a SADAD code (e.g. "005"); a
            # plain small number is the list position (e.g. "2").
            looks_like_code = len(digits) == 3 or digits.startswith("0")
            by_code = next(
                (o for o in options if o.code in (digits, digits.zfill(3))), None
            )
            index = int(digits)
            in_range = options[index - 1] if 1 <= index <= len(options) else None
            if looks_like_code and by_code is not None:
                return by_code
            if in_range is not None:
                return in_range
            if by_code is not None:
                return by_code
        for option in options:
            name = normalize(option.name)
            if name and (name in normalized or normalized in name):
                return option
        return None

    def _advance_bill(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult:
        """Apply defaults, then prompt for the next missing slot or confirm."""

        slots = state.slots
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

        return self._enter_confirmation(state, lang)

    def _fill_pending_bill(
        self, state: ConversationState, text: str, lang: Language
    ) -> None:
        slot = state.pending_slot
        slots = state.slots
        if slot == "reference_number" and not slots.reference_number:
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
            candidate = _recipient_from_answer(text, lang)
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
            biller_code=slots.biller_code,
            biller_name=slots.biller if slots.biller_code else None,
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

    # ---- Beneficiary directory (direct DB), balance & pre-flight (API) ------- #

    @staticmethod
    def _owner(state: ConversationState) -> str:
        """Owner scope for the Banking Core DB/API (demo user when unauthenticated)."""

        return state.user_id or "demo"

    def _resolve_beneficiary(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult | None:
        """Look the recipient up DIRECTLY in the beneficiaries DB.

        Returns a :class:`ConversationResult` when the turn must pause (to ask
        "which one?" or to offer adding a new beneficiary), or ``None`` to
        continue straight to confirmation.
        """

        if state.beneficiary_resolved:
            return None
        directory = get_beneficiary_directory()
        if directory is None:
            state.beneficiary_resolved = True
            return None
        name = (state.slots.recipient or "").strip()
        hits = directory.search(name, self._owner(state))
        if hits is None:
            # Directory unavailable — keep the free-text recipient.
            state.beneficiary_resolved = True
            return None
        if not hits:
            # Nobody matched: offer to add them (write goes through the API).
            state.pending_add_name = name
            state.pending_add_account = None
            state.add_resumes_transfer = True
            state.pending_slot = "beneficiary_account"
            state.status = ConversationStatus.COLLECTING
            return self._finish(state, templates.beneficiary_not_found(name, lang))
        if len(hits) == 1:
            self._lock_beneficiary(
                state,
                _display_name(hits[0].name, hits[0].name_ar, lang),
                hits[0].account,
            )
            return None
        state.beneficiary_options = [
            BeneficiaryOption(
                id=h.id,
                name=h.name,
                account=h.account,
                bank=h.bank,
                currency=h.currency,
                is_favorite=h.is_favorite,
                name_ar=h.name_ar,
            )
            for h in hits
        ]
        state.disambiguation_kind = "beneficiary"
        state.pending_slot = "recipient"
        state.status = ConversationStatus.DISAMBIGUATING
        return self._finish(
            state, templates.choose_beneficiary(self._option_rows(hits, lang), lang)
        )

    @staticmethod
    def _option_rows(
        items: Sequence[BeneficiaryHit | BeneficiaryOption], lang: Language
    ) -> list[tuple[str, str, str, str]]:
        """Build the (name, bank, masked-account, currency) rows for the prompt."""

        return [
            (
                _display_name(it.name, it.name_ar, lang),
                it.bank or "",
                _mask_account(it.account),
                it.currency,
            )
            for it in items
        ]

    @staticmethod
    def _lock_beneficiary(state: ConversationState, name: str, account: str) -> None:
        slots = state.slots
        slots.recipient = name
        slots.account_number = account
        state.beneficiary_resolved = True
        state.beneficiary_options = []
        state.disambiguation_kind = None
        state.pending_slot = None

    def _handle_beneficiary_choice(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        choice = self._match_beneficiary_option(state.beneficiary_options, text)
        if choice is None:
            rows = self._option_rows(state.beneficiary_options, lang)
            return self._finish(state, templates.choose_beneficiary(rows, lang))
        self._lock_beneficiary(
            state, _display_name(choice.name, choice.name_ar, lang), choice.account
        )
        return self._enter_confirmation(state, lang)

    @staticmethod
    def _match_beneficiary_option(
        options: list[BeneficiaryOption], text: str
    ) -> BeneficiaryOption | None:
        if not options:
            return None
        normalized = normalize(text)
        digits = "".join(ch for ch in normalize_digits(normalized) if ch.isdigit())
        # Last-4 digits of an account take priority over a list index.
        if len(digits) >= 4:
            for option in options:
                acct_digits = "".join(ch for ch in option.account if ch.isdigit())
                if acct_digits.endswith(digits[-4:]):
                    return option
        if digits:
            index = int(digits)
            if 1 <= index <= len(options):
                return options[index - 1]
        for option in options:
            for candidate in (option.name, option.name_ar):
                if not candidate:
                    continue
                name = normalize(candidate)
                if name and (name in normalized or normalized in name):
                    return option
        return None

    def _start_add_beneficiary(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        """Begin the standalone "add a beneficiary" flow (name, then account)."""

        state.intent = Intent.ADD_BENEFICIARY
        state.status = ConversationStatus.COLLECTING
        state.add_resumes_transfer = False
        name, account = self._parse_add_request(text)
        state.pending_add_name = name
        state.pending_add_account = account
        return self._advance_add_beneficiary(state, lang)

    def _continue_add_beneficiary(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        """Fill the name or the account slot from this turn, then move forward."""

        if _matches(text, _NEGATIVE):
            return self._abandon_add_beneficiary(state, lang)

        if state.pending_slot == "beneficiary_name":
            name, account = self._parse_add_request(text)
            state.pending_add_name = name or text.strip(" .,،؟?") or None
            if account:
                state.pending_add_account = account
            return self._advance_add_beneficiary(state, lang)

        # Otherwise this turn is the account/IBAN.
        raw = text.strip(" .,،؟?")
        if not raw:
            return self._advance_add_beneficiary(state, lang)
        account, reason = accounts.validate_account(raw)
        if account is None:
            name = state.pending_add_name or ""
            return self._finish(
                state,
                templates.beneficiary_invalid_account(name, reason or "", lang),
            )
        state.pending_add_account = account
        return self._advance_add_beneficiary(state, lang)

    def _advance_add_beneficiary(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult:
        """Ask for whichever piece is missing, else move to confirmation."""

        if not state.pending_add_name:
            state.pending_slot = "beneficiary_name"
            return self._finish(state, templates.ask_beneficiary_name(lang))
        if not state.pending_add_account:
            state.pending_slot = "beneficiary_account"
            return self._finish(
                state,
                templates.ask_beneficiary_account(state.pending_add_name, lang),
            )

        state.pending_slot = None
        state.status = ConversationStatus.CONFIRMING
        return self._finish(state, self._confirm_text(state, lang))

    def _abandon_add_beneficiary(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult:
        """The customer declined: drop back to the transfer, or cancel outright."""

        resumes = state.add_resumes_transfer
        state.pending_add_name = None
        state.pending_add_account = None
        state.add_resumes_transfer = False
        if not resumes:
            state.reset()
            state.status = ConversationStatus.CANCELLED
            return self._finish(state, templates.cancelled(lang))
        state.intent = Intent.TRANSFER_MONEY
        state.slots.recipient = None
        state.pending_slot = "recipient"
        state.status = ConversationStatus.COLLECTING
        return self._finish(state, templates.slot_prompt("recipient", lang))

    def _commit_add_beneficiary(
        self, state: ConversationState, lang: Language, tracer: BlockTracer
    ) -> ConversationResult:
        """Confirmed: write through the Banking Core API, then close or transfer."""

        name = state.pending_add_name or ""
        account = state.pending_add_account or ""
        resumes = state.add_resumes_transfer
        created = banking_core_client.add_beneficiary(
            owner_user=self._owner(state),
            name=name,
            account=account,
            currency=state.slots.currency or DEFAULT_CURRENCY,
        )
        state.pending_add_name = None
        state.pending_add_account = None
        state.add_resumes_transfer = False
        if not created or not created.get("ok"):
            reason = created.get("message") if created else None
            reply = templates.beneficiary_add_failed(name, lang, reason)
            if not resumes:
                state.reset()
                state.status = ConversationStatus.COMPLETED
                return self._finish(state, reply)
            state.intent = Intent.TRANSFER_MONEY
            state.status = ConversationStatus.COLLECTING
            state.pending_slot = "recipient"
            state.slots.recipient = None
            return self._finish(state, reply)

        state.slots.recipient = name
        state.slots.account_number = account
        state.beneficiary_resolved = True
        if not resumes:
            state.status = ConversationStatus.COMPLETED
            state.pending_slot = None
            return self._finish(
                state,
                templates.beneficiary_add_completed(name, _mask_account(account), lang),
            )
        # Mid-transfer: the single "yes" covered both, so send it straight through.
        state.intent = Intent.TRANSFER_MONEY
        result = self._complete(state, lang, tracer)
        result.reply = f"{templates.beneficiary_added(name, lang)} {result.reply}"
        return result

    @staticmethod
    def _parse_add_request(text: str) -> tuple[str | None, str | None]:
        """Split "add Sara Ali SA03…" into ``(name, validated_account)``."""

        name_parts: list[str] = []
        account: str | None = None
        for token in text.replace(",", " ").split():
            cleaned = token.strip(" .,،؟?\"'“”«»")
            if not cleaned:
                continue
            if account is None:
                valid, _ = accounts.validate_account(cleaned)
                if valid is not None:
                    account = valid
                    continue
            if normalize(cleaned) in _ADD_REQUEST_STOPWORDS:
                continue
            name_parts.append(cleaned)
        return (" ".join(name_parts) or None), account

    @staticmethod
    def _extract_account(text: str) -> str | None:
        """Pull a *valid* account/IBAN token from a free-text reply."""

        for token in text.replace(",", " ").split():
            cleaned = token.strip(" .,،؟?")
            account, _ = accounts.validate_account(cleaned)
            if account is not None:
                return account
        return None

    @staticmethod
    def _is_mid_transaction(state: ConversationState) -> bool:
        """True when a transfer/bill is in progress and awaiting the user."""

        if state.status in (
            ConversationStatus.CONFIRMING,
            ConversationStatus.DISAMBIGUATING,
        ):
            return True
        return state.status is ConversationStatus.COLLECTING and (
            state.pending_slot is not None
            or state.pending_add_name is not None
            or state.intent in (Intent.TRANSFER_MONEY, Intent.PAY_BILL)
        )

    def _active_prompt(self, state: ConversationState, lang: Language) -> str | None:
        """Re-emit the question the in-progress flow is currently waiting on."""

        if state.status is ConversationStatus.CONFIRMING:
            text = self._confirm_text(state, lang)
            note = templates.warnings_note(state.preflight_warnings, lang)
            return f"{text} {note}" if note else text
        if state.status is ConversationStatus.DISAMBIGUATING:
            if state.disambiguation_kind == "beneficiary":
                options = [
                    (o.name, o.bank or "", _mask_account(o.account), o.currency)
                    for o in state.beneficiary_options
                ]
                return templates.choose_beneficiary(options, lang)
            return templates.choose_biller(
                [opt.name for opt in state.biller_options], lang
            )
        if state.pending_add_name:
            return templates.beneficiary_not_found(state.pending_add_name, lang)
        if state.pending_slot:
            return templates.slot_prompt(state.pending_slot, lang, state.intent)
        return None

    def _answer_balance_aside(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        """Answer a balance question mid-flow without disturbing the transaction."""

        account_type = _account_type(text)
        info = banking_core_client.get_balance(
            owner_user=self._owner(state), account_type=account_type
        )
        if info is None:
            balance_line = templates.balance_unavailable(lang)
        else:
            balance_line = templates.balance_reply(
                info.account_type, info.currency, self._fmt_amount(info.balance), lang
            )
        resume = self._active_prompt(state, lang)
        if resume:
            reply = f"{balance_line} {templates.resume_note(lang)} {resume}"
        else:
            reply = balance_line
        return self._finish(state, reply)

    def _handle_balance_inquiry(
        self, state: ConversationState, text: str, lang: Language
    ) -> ConversationResult:
        """Answer a balance inquiry using the external Banking Core API."""

        account_type = _account_type(text)
        info = banking_core_client.get_balance(
            owner_user=self._owner(state), account_type=account_type
        )
        state.intent = Intent.BALANCE_INQUIRY
        state.status = ConversationStatus.COMPLETED
        if info is None:
            return self._finish(state, templates.balance_unavailable(lang))
        reply = templates.balance_reply(
            info.account_type, info.currency, self._fmt_amount(info.balance), lang
        )
        return self._finish(state, reply)

    def _handle_list_beneficiaries(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult:
        """List the customer's saved beneficiaries (read-only; never transfers)."""

        state.intent = Intent.LIST_BENEFICIARIES
        state.status = ConversationStatus.COMPLETED
        directory = get_beneficiary_directory()
        if directory is None:
            return self._finish(state, templates.beneficiaries_unavailable(lang))
        hits = directory.list_all(self._owner(state))
        if hits is None:
            return self._finish(state, templates.beneficiaries_unavailable(lang))
        if not hits:
            return self._finish(state, templates.no_beneficiaries(lang))
        rows = self._option_rows(hits, lang)
        return self._finish(state, templates.list_beneficiaries(rows, lang))

    def _enter_confirmation(
        self, state: ConversationState, lang: Language
    ) -> ConversationResult:
        """Run pre-flight (advisory), then move to CONFIRMING with the review text."""

        state.pending_slot = None
        state.status = ConversationStatus.CONFIRMING
        self._run_preflight(state)
        reply = self._confirm_text(state, lang)
        note = templates.warnings_note(state.preflight_warnings, lang)
        if note:
            reply = f"{reply} {note}"
        return self._finish(state, reply)

    def _run_preflight(self, state: ConversationState) -> None:
        """Call the Banking Core pre-flight API; store advisory warnings on state."""

        state.preflight_warnings = []
        slots = state.slots
        if slots.amount is None or not slots.currency:
            return
        source_type = (
            _ACCOUNT_TYPES.get(slots.source_account.lower())
            if slots.source_account
            else None
        )
        result = None
        if state.intent is Intent.PAY_BILL:
            result = banking_core_client.preflight_bill(
                owner_user=self._owner(state),
                amount=slots.amount,
                currency=slots.currency,
                biller_code=slots.biller_code,
                reference_number=slots.reference_number or "",
                source_account=slots.source_account,
                source_account_type=source_type,
            )
        elif state.intent is Intent.TRANSFER_MONEY:
            result = banking_core_client.preflight_transfer(
                owner_user=self._owner(state),
                amount=slots.amount,
                currency=slots.currency,
                recipient_account=slots.account_number,
                source_account=slots.source_account,
                source_account_type=source_type,
            )
        if result is not None:
            state.preflight_warnings = list(result.warnings)

    def _confirm_text(self, state: ConversationState, lang: Language) -> str:
        slots = state.slots
        if state.pending_add_account:
            masked = _mask_account(state.pending_add_account)
            name = state.pending_add_name or ""
            if state.add_resumes_transfer:
                return templates.confirm_add_then_transfer(
                    name,
                    masked,
                    self._fmt_amount(slots.amount),
                    slots.currency or "",
                    lang,
                )
            return templates.confirm_add_beneficiary(name, masked, lang)
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
        flagged_terms: list[str] | None = None,
    ) -> ConversationResult:
        self._store.save(state)
        return ConversationResult(
            state, reply, transfer=transfer, bill=bill, flagged_terms=flagged_terms
        )


_engine: ConversationEngine | None = None


def get_engine() -> ConversationEngine:
    global _engine
    if _engine is None:
        _engine = ConversationEngine()
    return _engine
