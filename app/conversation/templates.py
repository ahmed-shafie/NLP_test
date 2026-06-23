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
    Language.EN: "I can help you transfer money. Try, for example, "
    '"send 500 USD to Ahmed".',
    Language.AR: 'أستطيع مساعدتك في تحويل الأموال. جرّب مثلاً: "حوّل ٥٠٠ دولار إلى أحمد".',
}

_CANCELLED: dict[Language, str] = {
    Language.EN: "Okay, I've cancelled the transfer.",
    Language.AR: "حسناً، تم إلغاء عملية التحويل.",
}


def slot_prompt(
    slot: str, language: Language, intent: Intent | None = None
) -> str:
    if intent is not None:
        override = _SLOT_PROMPTS_BY_INTENT.get(intent, {}).get(slot)
        if override is not None:
            return override[language]
    return _SLOT_PROMPTS.get(slot, {}).get(language, f"Please provide the {slot}.")


def choose_action(language: Language) -> str:
    return _CHOOSE_ACTION[language]


def greeting(language: Language) -> str:
    return _GREETING[language]


def fallback(language: Language) -> str:
    return _FALLBACK[language]


def cancelled(language: Language) -> str:
    return _CANCELLED[language]


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
