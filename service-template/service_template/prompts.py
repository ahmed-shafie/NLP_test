"""Bilingual (EN/AR) user-facing strings.

Mirrors ``app/conversation/templates.py``: every sentence the assistant says
lives here, keyed by language, so the engine never hard-codes copy. This keeps
localisation in one place and the engine logic clean.

To localise into more languages, add branches (or switch to a dict/gettext).
"""

from __future__ import annotations

from decimal import Decimal

from service_template.schemas import Language
from service_template.state import Candidate

# Human-readable prompt for each slot, per language. When you add a slot to a
# case, add its prompt here.
_SLOT_PROMPTS: dict[str, dict[Language, str]] = {
    "amount": {
        Language.EN: "How much would you like to send?",
        Language.AR: "كم المبلغ الذي تريد تحويله؟",
    },
    "currency": {
        Language.EN: "Which currency?",
        Language.AR: "بأي عملة؟",
    },
    "recipient": {
        Language.EN: "Who would you like to send money to?",
        Language.AR: "إلى من تريد التحويل؟",
    },
    "source_account": {
        Language.EN: "Which account should I use?",
        Language.AR: "من أي حساب؟",
    },
}


def slot_prompt(slot: str, language: Language) -> str:
    """Ask for a single missing slot."""

    prompts = _SLOT_PROMPTS.get(slot)
    if prompts is None:
        return (
            f"Could you provide the {slot}?"
            if language is Language.EN
            else f"هل يمكنك تزويدي بـ {slot}؟"
        )
    return prompts[language]


def confirm_transfer(
    amount: Decimal, currency: str, recipient: str, language: Language
) -> str:
    """The yes/no confirmation shown once all slots are gathered."""

    if language is Language.AR:
        return f"سأحوّل {amount} {currency} إلى {recipient}. هل أتابع؟ (نعم/لا)"
    return f"I'll send {amount} {currency} to {recipient}. Shall I proceed? (yes/no)"


def choose_candidate(candidates: list[Candidate], language: Language) -> str:
    """Disambiguation prompt listing candidates 1..N."""

    lines = []
    for i, c in enumerate(candidates, start=1):
        detail = f" · {c.detail}" if c.detail else ""
        lines.append(f"{i}. {c.name}{detail}")
    listing = "\n".join(lines)
    if language is Language.AR:
        header = f"لديك {len(candidates)} مستفيدين بهذا الاسم — أي واحد؟"
        return f"{header}\n{listing}"
    header = f"You have {len(candidates)} people by that name — which one?"
    return f"{header}\n{listing}"


def completed(recipient: str, language: Language) -> str:
    if language is Language.AR:
        return f"تم تجهيز التحويل إلى {recipient}. ✓"
    return f"Done — the transfer to {recipient} is ready. ✓"


def cancelled(language: Language) -> str:
    return "Okay, cancelled." if language is Language.EN else "تم الإلغاء."


def small_talk(language: Language) -> str:
    if language is Language.AR:
        return "مرحباً! يمكنني مساعدتك في تحويل الأموال. ماذا تريد أن تفعل؟"
    return "Hello! I can help you send money. What would you like to do?"


def fallback(language: Language) -> str:
    if language is Language.AR:
        return "لم أفهم ذلك تماماً. جرّب مثلاً: «حوّل ٥٠٠ ريال إلى أحمد»."
    return "I didn't quite get that. Try, e.g., 'send 500 SAR to Ahmed'."


def warnings_note(warnings: list[str], language: Language) -> str:
    """Render advisory (non-blocking) pre-flight notes for the confirmation."""

    if not warnings:
        return ""
    joined = "; ".join(warnings)
    if language is Language.AR:
        return f"⚠️ ملاحظة: {joined}"
    return f"⚠️ Note: {joined}"
