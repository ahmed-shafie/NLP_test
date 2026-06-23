"""Bilingual (English/Arabic) response templates for the conversation engine."""

from __future__ import annotations

from app.schemas import Intent, Language

# Follow-up prompts for each missing slot, keyed by language.
_SLOT_PROMPTS: dict[str, dict[Language, str]] = {
    "amount": {
        Language.EN: "How much would you like to transfer?",
        Language.AR: "ما المبلغ الذي تريد تحويله؟",
    },
    "currency": {
        Language.EN: "Which currency should I use?",
        Language.AR: "بأي عملة تريد التحويل؟",
    },
    "recipient": {
        Language.EN: "Who should I send the money to?",
        Language.AR: "إلى من تريد إرسال المبلغ؟",
    },
    "biller": {
        Language.EN: "Which bill would you like to pay?",
        Language.AR: "أي فاتورة تريد دفعها؟",
    },
    "reference_number": {
        Language.EN: "What's the bill/reference number?",
        Language.AR: "ما هو رقم الفاتورة/المرجع؟",
    },
}

# Intent-specific overrides (e.g. "pay" instead of "transfer" for the amount).
_SLOT_PROMPTS_BY_INTENT: dict[Intent, dict[str, dict[Language, str]]] = {
    Intent.PAY_BILL: {
        "amount": {
            Language.EN: "How much would you like to pay?",
            Language.AR: "ما المبلغ الذي تريد دفعه؟",
        },
    },
}

_CHOOSE_ACTION: dict[Language, str] = {
    Language.EN: "What would you like to do — (1) Transfer money or (2) Pay a bill?",
    Language.AR: "ماذا تريد أن تفعل — (١) تحويل أموال أم (٢) دفع فاتورة؟",
}

_GREETING: dict[Language, str] = {
    Language.EN: "Sure — let's set up your transfer.",
    Language.AR: "تمام، لنجهّز عملية التحويل.",
}

_FALLBACK: dict[Language, str] = {
    Language.EN: "Hmm, I didn't quite catch that 🤔 — I can send money or pay a "
    'bill. For example, try "send 500 SAR to Ahmed" or "pay my STC bill".',
    Language.AR: "لم أفهم تماماً 🤔 — أستطيع تحويل الأموال أو دفع الفواتير. "
    'جرّب مثلاً: "حوّل ٥٠٠ ريال إلى أحمد" أو "ادفع فاتورة STC".',
}

# Warm chit-chat replies, keyed by a small-talk kind. Each one stays helpful by
# gently steering the customer back to what the assistant can do.
_SMALL_TALK: dict[str, dict[Language, str]] = {
    "greeting": {
        Language.EN: "Hey! 👋 Good to see you. I can send money or pay a bill "
        "for you — what's up?",
        Language.AR: "هلا والله! 👋 سعيد إني أشوفك. أقدر أحوّل لك فلوس أو أدفع "
        "فاتورة — وش تحتاج؟",
    },
    "thanks": {
        Language.EN: "Anytime! 😊 Need anything else — a transfer or a bill?",
        Language.AR: "على الرحب! 😊 تحتاج شي ثاني — تحويل أو فاتورة؟",
    },
    "how_are_you": {
        Language.EN: "I'm good, thanks for asking! 😄 So, wanna send some money "
        "or pay a bill?",
        Language.AR: "تمام والحمد لله، تسلم على السؤال! 😄 تبي تحوّل فلوس أو تدفع "
        "فاتورة؟",
    },
    "bye": {
        Language.EN: "Catch you later! 👋 I'm around whenever you wanna send "
        "money or pay a bill.",
        Language.AR: "نشوفك على خير! 👋 أنا موجود وقت ما تبي تحويل أو فاتورة.",
    },
    "default": {
        Language.EN: "Love a good chat! 😊 I'm best with money transfers and "
        "bills though — wanna give one a go?",
        Language.AR: "يسعدني السوالف! 😊 بس أنا أشطر في التحويلات ودفع الفواتير "
        "— نجرّب وحدة؟",
    },
}

# Keyword cues used to pick the right warm reply (matched on normalized tokens).
_SMALL_TALK_CUES: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "how_are_you",
        frozenset(
            {"how", "are", "you", "doing", "كيف", "حالك", "عامل", "ايه", "اخبارك"}
        ),
    ),
    (
        "thanks",
        frozenset({"thanks", "thank", "thx", "شكرا", "مشكور", "تسلم", "يعطيك"}),
    ),
    (
        "bye",
        frozenset({"bye", "goodbye", "وداعا", "باي"}),
    ),
    (
        "greeting",
        frozenset(
            {
                "hi",
                "hello",
                "hey",
                "yo",
                "morning",
                "evening",
                "مرحبا",
                "اهلا",
                "السلام",
                "هاي",
                "صباح",
                "مساء",
            }
        ),
    ),
)

_CANCELLED: dict[Language, str] = {
    Language.EN: "Okay, I've cancelled the transfer.",
    Language.AR: "حسناً، تم إلغاء عملية التحويل.",
}


def slot_prompt(slot: str, language: Language, intent: Intent | None = None) -> str:
    if intent is not None:
        override = _SLOT_PROMPTS_BY_INTENT.get(intent, {}).get(slot)
        if override is not None:
            return override[language]
    return _SLOT_PROMPTS.get(slot, {}).get(language, f"Please provide the {slot}.")


def choose_action(language: Language) -> str:
    return _CHOOSE_ACTION[language]


def choose_biller(names: list[str], language: Language) -> str:
    """Ask the customer which biller they meant when a term is ambiguous."""

    listing = "  ".join(f"({i + 1}) {name}" for i, name in enumerate(names))
    if language is Language.AR:
        return f"وجدت أكثر من جهة لهذه الفاتورة — أيها تقصد؟ {listing}"
    return f"I found more than one biller for that — which one? {listing}"


def greeting(language: Language) -> str:
    return _GREETING[language]


def fallback(language: Language) -> str:
    return _FALLBACK[language]


def cancelled(language: Language) -> str:
    return _CANCELLED[language]


# Filler words allowed in an otherwise pure chit-chat message ("hello there").
_SMALL_TALK_FILLERS: frozenset[str] = frozenset(
    {"there", "friend", "buddy", "sir", "dear", "please", "the", "a", "good"}
)
_ALL_SMALL_TALK_CUES: frozenset[str] = frozenset(
    tok for _, cues in _SMALL_TALK_CUES for tok in cues
)


def _small_talk_kind(text: str) -> str | None:
    from app.nlu.normalize import normalize_tokens

    tokens = set(normalize_tokens(text))
    if not tokens:
        return None
    # Only treat as chit-chat when the message is *purely* greeting/thanks/etc.
    # (cue tokens + a few fillers), so "hi I want to pay a bill" still routes.
    if not tokens & _ALL_SMALL_TALK_CUES:
        return None
    if tokens - _ALL_SMALL_TALK_CUES - _SMALL_TALK_FILLERS:
        return None
    for kind, cues in _SMALL_TALK_CUES:
        if tokens & cues:
            return kind
    return None


def is_small_talk(text: str) -> bool:
    """True when ``text`` is *purely* chit-chat (greeting/thanks/bye/etc.)."""

    return _small_talk_kind(text) is not None


def small_talk(text: str, language: Language) -> str:
    """Return a warm chit-chat reply, picking the kind from ``text`` cues."""

    kind = _small_talk_kind(text) or "default"
    return _SMALL_TALK[kind][language]


def confirm_prompt(
    amount: str, currency: str, recipient: str, language: Language
) -> str:
    if language is Language.AR:
        return (
            f"تأكيد: تحويل {amount} {currency} إلى {recipient}. " "هل أتابع؟ (نعم/لا)"
        )
    return (
        f"Please confirm: transfer {amount} {currency} to {recipient}. "
        "Shall I proceed? (yes/no)"
    )


def completed(amount: str, currency: str, recipient: str, language: Language) -> str:
    if language is Language.AR:
        return f"تم تجهيز التحويل: {amount} {currency} إلى {recipient}."
    return f"Done — your transfer of {amount} {currency} to {recipient} is ready."


def bill_confirm_prompt(
    amount: str, currency: str, biller: str, reference: str, language: Language
) -> str:
    if language is Language.AR:
        return (
            f"تأكيد: دفع {amount} {currency} لفاتورة {biller} (مرجع {reference}). "
            "هل أتابع؟ (نعم/لا)"
        )
    return (
        f"Please confirm: pay {amount} {currency} to {biller} (ref {reference}). "
        "Shall I proceed? (yes/no)"
    )


def bill_completed(
    amount: str, currency: str, biller: str, reference: str, language: Language
) -> str:
    if language is Language.AR:
        return (
            f"تم — تجهيز دفع فاتورة {biller} بمبلغ {amount} {currency} "
            f"(مرجع {reference})."
        )
    return (
        f"Done — your {biller} bill payment of {amount} {currency} "
        f"(ref {reference}) is ready."
    )


def alias_created(name: str, language: Language) -> str:
    """Tell the customer an alias was auto-created, with how to undo/rename it."""

    if language is Language.AR:
        return (
            f"✓ حفظت اختصارًا باسم '{name}' — في المرة القادمة يكفي أن تقول "
            f"'حوّل إلى {name}'. أرسل 'احذف {name}' لإلغائه."
        )
    return (
        f"✓ Saved a shortcut '{name}' — next time just say 'send to {name}'. "
        f"Reply 'forget {name}' to remove it."
    )


def alias_forgotten(name: str, language: Language) -> str:
    if language is Language.AR:
        return f"تم حذف الاختصار '{name}'."
    return f"Removed the shortcut '{name}'."


def alias_not_found(name: str, language: Language) -> str:
    if language is Language.AR:
        return f"لا يوجد اختصار باسم '{name}'."
    return f"You don't have a shortcut named '{name}'."
