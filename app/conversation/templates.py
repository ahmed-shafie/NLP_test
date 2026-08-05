"""Bilingual (English/Arabic) response templates for the conversation engine."""

from __future__ import annotations

import random

from app.config import settings
from app.schemas import Intent, Language

_RNG = random.Random()

# Follow-up prompts for each missing slot, keyed by language.
_SLOT_PROMPTS: dict[str, dict[Language, str]] = {
    "amount": {
        Language.EN: "Sure — how much would you like to send?",
        Language.AR: "تمام — كم المبلغ اللي تحب تحوّله؟",
    },
    "currency": {
        Language.EN: "Got it — which currency are we using?",
        Language.AR: "تمام — بأي عملة نسوّيها؟",
    },
    "recipient": {
        Language.EN: "Sure — who would you like to send it to?",
        Language.AR: "أكيد — لمن تحب تحوّل؟",
    },
    "biller": {
        Language.EN: "Of course — which bill are we paying?",
        Language.AR: "أبشر — أي فاتورة نسدّدها؟",
    },
    "reference_number": {
        Language.EN: "Great — what's the bill or reference number?",
        Language.AR: "تمام — وش رقم الفاتورة أو المرجع؟",
    },
}

# Intent-specific overrides (e.g. "pay" instead of "transfer" for the amount).
_SLOT_PROMPTS_BY_INTENT: dict[Intent, dict[str, dict[Language, str]]] = {
    Intent.PAY_BILL: {
        "amount": {
            Language.EN: "Sure — how much would you like to pay?",
            Language.AR: "تمام — كم المبلغ اللي تحب تدفعه؟",
        },
    },
}

_CHOOSE_ACTION: dict[Language, str] = {
    Language.EN: "Happy to help! Would you like to (1) send money or (2) pay a bill?",
    Language.AR: "حياك الله! تحب (١) تحويل فلوس أو (٢) دفع فاتورة؟",
}

_GREETING: dict[Language, str] = {
    Language.EN: "Of course — let's get your transfer sorted.",
    Language.AR: "أبشر، خلنا نجهّز التحويل.",
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
    Language.EN: "No problem, I've cancelled that for you.",
    Language.AR: "ولا يهمّك، ألغيت العملية.",
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


def _option_lines(options: list[tuple[str, str, str, str]]) -> str:
    """Render ``(name, bank, masked_account, currency)`` tuples as numbered rows."""

    lines = []
    for i, (name, bank, masked, currency) in enumerate(options, start=1):
        parts = [name]
        if bank:
            parts.append(bank)
        parts.append(masked)
        parts.append(currency)
        lines.append(f"{i}. " + " · ".join(parts))
    return "\n".join(lines)


def choose_beneficiary(
    options: list[tuple[str, str, str, str]], language: Language
) -> str:
    """Ask which beneficiary is meant when a first name matches several people.

    ``options`` is a list of ``(name, bank, masked_account, currency)`` tuples.
    """

    listing = _option_lines(options)
    first = options[0][0].split()[0] if options else ""
    if language is Language.AR:
        return (
            f'لقيت أكثر من شخص باسم "{first}" 🙂 — أي واحد تقصد؟\n'
            f"{listing}\n"
            "قل لي الرقم (١/٢/٣)، أو الاسم الكامل، أو آخر ٤ أرقام."
        )
    return (
        f'I found a few people named "{first}" 🙂 — which one did you mean?\n'
        f"{listing}\n"
        "Just reply with the number (1/2/3), the full name, or the last 4 digits."
    )


def beneficiary_not_found(name: str, language: Language) -> str:
    """No beneficiary matched; offer to add one via the external API."""

    if language is Language.AR:
        return (
            f'ما لقيت أحد محفوظ باسم "{name}" 🤔 — تحب أضيفه؟ '
            'أرسل لي رقم الحساب/الآيبان، أو اكتب "لا" إذا تبي تتركها.'
        )
    return (
        f'Hmm, I don\'t have anyone saved as "{name}" yet 🤔 — want me to add '
        'them? Just send their account/IBAN, or say "no" to skip.'
    )


def beneficiary_added(name: str, language: Language) -> str:
    if language is Language.AR:
        return f'✓ تمت إضافة المستفيد "{name}".'
    return f'✓ Added beneficiary "{name}".'


def beneficiary_add_failed(
    name: str, language: Language, reason: str | None = None
) -> str:
    if reason:
        # Surface the specific reason from the banking service (e.g. duplicate
        # account) instead of a generic "try again later".
        if language is Language.AR:
            return f'تعذّرت إضافة المستفيد "{name}": {reason}'
        return f'I couldn\'t add "{name}": {reason}'
    if language is Language.AR:
        return (
            f'تعذّرت إضافة المستفيد "{name}" الآن. حاول لاحقًا أو استخدم '
            "مستفيدًا موجودًا."
        )
    return (
        f'I couldn\'t add "{name}" right now. Please try again later or use an '
        "existing beneficiary."
    )


def beneficiary_add_invalid_account(name: str, language: Language) -> str:
    """The reply wasn't a plausible account number / IBAN."""

    if language is Language.AR:
        return (
            "هذا لا يبدو رقم حساب أو آيبان صالحًا. أرسل آيبان كاملًا (مثل SA "
            f'متبوعة بأرقام) أو رقم حساب لإضافة "{name}"، أو اكتب "لا" للإلغاء.'
        )
    return (
        "That doesn't look like a valid account number or IBAN. Send a full "
        f'IBAN (e.g. SA followed by digits) or an account number to add "{name}", '
        'or reply "no" to cancel.'
    )


def list_beneficiaries(
    options: list[tuple[str, str, str, str]], language: Language
) -> str:
    """Show the customer's saved beneficiaries (read-only, no transfer started).

    ``options`` is a list of ``(name, bank, masked_account, currency)`` tuples.
    """

    listing = _option_lines(options)
    count = len(options)
    if language is Language.AR:
        word = "مستفيد" if count == 1 else "مستفيدين"
        return (
            f"عندك {count} {word} محفوظين 👇\n"
            f"{listing}\n"
            "تحب تحوّل لأحدهم؟ قل لي الاسم."
        )
    word = "beneficiary" if count == 1 else "beneficiaries"
    return (
        f"You have {count} saved {word} 👇\n"
        f"{listing}\n"
        "Want to send money to one of them? Just tell me the name."
    )


def no_beneficiaries(language: Language) -> str:
    """The customer has no saved beneficiaries yet."""

    if language is Language.AR:
        return (
            "ما عندك مستفيدين محفوظين للحين 🙂 — تحب تضيف واحد؟ "
            "قل لي الاسم ورقم الحساب/الآيبان."
        )
    return (
        "You don't have any saved beneficiaries yet 🙂 — want to add one? "
        "Just tell me the name and their account/IBAN."
    )


def beneficiaries_unavailable(language: Language) -> str:
    """The beneficiary directory could not be reached."""

    if language is Language.AR:
        return "تعذّر جلب قائمة المستفيدين الآن. حاول مرة أخرى لاحقًا."
    return "I couldn't fetch your beneficiaries right now. Please try again later."


_ACCOUNT_TYPE_AR: dict[str, str] = {
    "current": "الجاري",
    "savings": "التوفير",
    "credit": "الائتمان",
    "salary": "الراتب",
}


def balance_reply(
    account_type: str, currency: str, balance: str, language: Language
) -> str:
    if language is Language.AR:
        label = _ACCOUNT_TYPE_AR.get(account_type, account_type)
        return f"رصيد حساب {label} عندك {balance} {currency}. 💰"
    return f"You've got {balance} {currency} in your {account_type} account. 💰"


def balance_unavailable(language: Language) -> str:
    if language is Language.AR:
        return "تعذّر جلب الرصيد الآن. حاول مرة أخرى لاحقًا."
    return "I couldn't fetch your balance right now. Please try again later."


def resume_note(language: Language) -> str:
    """Short connector shown before re-emitting an in-progress prompt."""

    if language is Language.AR:
        return "طيّب، نرجع للتحويل —"
    return "Okay, back to where we were —"


def warnings_note(warnings: list[str], language: Language) -> str:
    """Turn raw pre-flight warning codes into a short bilingual advisory line."""

    notes: list[str] = []
    for warning in warnings:
        if warning.startswith("low_funds"):
            short = warning.split(":", 1)[1].strip() if ":" in warning else ""
            if language is Language.AR:
                notes.append(f"⚠️ الرصيد غير كافٍ ({short}) — يمكنك المتابعة.")
            else:
                notes.append(
                    f"⚠️ Balance may be insufficient ({short}) — "
                    "you can still proceed."
                )
        elif warning.startswith("fx"):
            detail = warning.split(":", 1)[1].strip() if ":" in warning else ""
            if language is Language.AR:
                notes.append(f"ℹ️ سيتم تحويل العملة ({detail}).")
            else:
                notes.append(f"ℹ️ A currency conversion applies ({detail}).")
    return " ".join(notes)


def greeting(language: Language) -> str:
    return _GREETING[language]


def fallback(language: Language) -> str:
    return _FALLBACK[language]


def cancelled(language: Language) -> str:
    return _CANCELLED[language]


# Calm, professional replies to abusive ("ribald") input. Multiple variants per
# (severity, language) so the assistant doesn't sound canned; one is picked at
# random, avoiding an immediate repeat. ``mild`` variants may name the flagged
# word(s) via ``{words}``; ``severe`` variants never echo the language back.
_INAPPROPRIATE: dict[str, dict[Language, list[str]]] = {
    "mild": {
        Language.EN: [
            "Let's keep it friendly 🙂 — I'll set {words} aside. I can send money "
            "or pay a bill; which would you like?",
            "No worries, but let's leave {words} out. Want to make a transfer or "
            "pay a bill?",
            "I'd rather skip {words} and keep things respectful. What can I help "
            "you with — a transfer or a bill?",
        ],
        Language.AR: [
            "خلينا نتكلم بأدب 🙂 — بتجاوز عن {words}. أقدر أحوّل فلوس أو أدفع فاتورة، "
            "وش تحب؟",
            "ولا يهمك، بس نخلي {words} على جنب. تبي تحويل أو دفع فاتورة؟",
            "أفضّل نتجاوز {words} ونكمل باحترام. كيف أقدر أساعدك — تحويل أو فاتورة؟",
        ],
    },
    "severe": {
        Language.EN: [
            "Let's keep this respectful. I can help with a transfer or a bill "
            "payment — what would you like to do?",
            "I'm here to help with your banking. Shall we do a transfer or pay a "
            "bill?",
            "I'd like to keep our chat professional. I can send money or pay a "
            "bill for you — which one?",
        ],
        Language.AR: [
            "خلينا نحافظ على الاحترام. أقدر أساعدك في تحويل أو دفع فاتورة — وش تحب؟",
            "أنا هنا لمساعدتك في معاملاتك البنكية. نبدأ بتحويل أو دفع فاتورة؟",
            "أحب نكمل المحادثة باحترام. أقدر أحوّل لك فلوس أو أدفع فاتورة — أيها؟",
        ],
    },
}

# Firm message when a session exceeds the abuse strike limit.
_REPEAT_OFFENSE: dict[Language, str] = {
    Language.EN: "I'm not able to continue this conversation. Please contact "
    "support if you need help.",
    Language.AR: "ما أقدر أكمل هذه المحادثة. تواصل مع خدمة العملاء إذا تحتاج "
    "مساعدة.",
}


def _pick_variant(count: int, last_index: int | None) -> int:
    """Pick a variant index, avoiding ``last_index`` so it isn't repeated."""

    if count <= 1:
        return 0
    seed = settings.reply_variation_seed
    rng = random.Random(seed) if seed is not None else _RNG
    choices = [i for i in range(count) if i != last_index]
    return rng.choice(choices or list(range(count)))


def inappropriate(
    language: Language,
    severity: str,
    flagged: tuple[str, ...] = (),
    last_index: int | None = None,
) -> tuple[str, int]:
    """Return ``(reply, index)`` for an abusive turn.

    A ``mild`` reply names the flagged word(s); ``severe`` (or a flagged-but-
    untermed semantic catch) uses a generic redirect that never echoes the text.
    """

    bucket = "mild" if severity == "mild" and flagged else "severe"
    variants = _INAPPROPRIATE[bucket][language]
    index = _pick_variant(len(variants), last_index)
    reply = variants[index]
    if "{words}" in reply:
        fallback_word = "that" if language is Language.EN else "ذلك"
        words = ", ".join(f'"{w}"' for w in flagged) or fallback_word
        reply = reply.replace("{words}", words)
    return reply, index


def repeat_offense(language: Language) -> str:
    return _REPEAT_OFFENSE[language]


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
        return f"للتأكيد — أحوّل {amount} {currency} إلى {recipient}؟ (نعم/لا)"
    return f"Just to confirm — send {amount} {currency} to {recipient}? (yes/no)"


_COMPLETED: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "All set! ✅ Your transfer of {amount} {currency} to {recipient} is ready "
        "to go.",
        "Done! ✅ I've prepared your transfer of {amount} {currency} to {recipient}.",
        "Perfect — {amount} {currency} to {recipient} is ready to go. ✅",
    ),
    Language.AR: (
        "تمّ ✅ — جهّزت تحويل {amount} {currency} إلى {recipient}.",
        "تمام! ✅ حوّلت {amount} {currency} إلى {recipient} وكل شي جاهز.",
        "خلاص، جهّزت {amount} {currency} إلى {recipient} ✅",
    ),
}


def completed(amount: str, currency: str, recipient: str, language: Language) -> str:
    variants = _COMPLETED[language]
    template = variants[_pick_variant(len(variants), None)]
    return template.format(amount=amount, currency=currency, recipient=recipient)


def bill_confirm_prompt(
    amount: str, currency: str, biller: str, reference: str, language: Language
) -> str:
    if language is Language.AR:
        return (
            f"للتأكيد — أدفع {amount} {currency} لفاتورة {biller} "
            f"(مرجع {reference})؟ (نعم/لا)"
        )
    return (
        f"Just to confirm — pay {amount} {currency} to {biller} "
        f"(ref {reference})? (yes/no)"
    )


def bill_completed(
    amount: str, currency: str, biller: str, reference: str, language: Language
) -> str:
    if language is Language.AR:
        return (
            f"تمّ ✅ — جهّزت دفع فاتورة {biller} بمبلغ {amount} {currency} "
            f"(مرجع {reference})."
        )
    return (
        f"All set! ✅ Your {biller} bill payment of {amount} {currency} "
        f"(ref {reference}) is ready to go."
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
