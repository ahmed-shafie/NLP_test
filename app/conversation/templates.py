"""Bilingual (English/Arabic) response templates for the conversation engine."""

from __future__ import annotations

from app.schemas import Language

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


def slot_prompt(slot: str, language: Language) -> str:
    return _SLOT_PROMPTS.get(slot, {}).get(language, f"Please provide the {slot}.")


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
