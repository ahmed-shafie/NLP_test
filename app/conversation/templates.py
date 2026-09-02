"""Bilingual (English/Arabic) response templates for the conversation engine.

Replies come in two tiers (see :mod:`app.conversation.phrasing`). Money-critical
replies — confirmations, amounts, accounts, write outcomes, rendered lists — have
exactly one wording here and are asserted verbatim by tests. Conversational
replies carry no money fact, so each one has several equivalent phrasings and is
rendered through :func:`phrasing.varied`, which rotates between them and may hand
them to the local LLM for re-wording when that is switched on.
"""

from __future__ import annotations

import random

from app.config import settings
from app.conversation import phrasing, topic_replies
from app.nlu import accounts
from app.nlu.normalize import normalize, normalize_tokens
from app.nlu.semantic_intents import TopicEvidence
from app.schemas import Intent, Language

_RNG = random.Random()

# Follow-up prompts for each missing slot, keyed by language. Asked on almost
# every conversation, so this is where repetition is most noticeable.
_SLOT_PROMPTS: dict[str, dict[Language, tuple[str, ...]]] = {
    "amount": {
        Language.EN: (
            "Sure — how much would you like to send?",
            "Happy to — what amount are we sending?",
            "Of course. How much should I send?",
        ),
        Language.AR: (
            "تمام — كم المبلغ اللي تحب تحوّله؟",
            "أبشر — كم تبي تحوّل؟",
            "طيب، وش المبلغ؟",
        ),
    },
    "currency": {
        Language.EN: (
            "Got it — which currency are we using?",
            "Noted. Which currency should that be in?",
            "Sure — what currency are we sending?",
        ),
        Language.AR: (
            "تمام — بأي عملة نسوّيها؟",
            "طيب، وش العملة؟",
            "أبشر — نحوّلها بأي عملة؟",
        ),
    },
    "recipient": {
        Language.EN: (
            "Sure — who would you like to send it to?",
            "Of course. Who's it going to?",
            "Happy to — who should I send it to?",
        ),
        Language.AR: (
            "أكيد — لمن تحب تحوّل؟",
            "تمام — لمن نحوّلها؟",
            "أبشر — وش اسم اللي تحوّل له؟",
        ),
    },
    "biller": {
        Language.EN: (
            "Of course — which bill are we paying?",
            "Sure — which biller is it?",
            "Happy to — whose bill are we paying?",
        ),
        Language.AR: (
            "أبشر — أي فاتورة نسدّدها؟",
            "تمام — فاتورة أي جهة؟",
            "طيب، وش الفاتورة اللي نسدّدها؟",
        ),
    },
    "reference_number": {
        Language.EN: (
            "Great — what's the bill or reference number?",
            "Perfect. What's the reference number on the bill?",
            "Sure — send me the bill number, please.",
        ),
        Language.AR: (
            "تمام — وش رقم الفاتورة أو المرجع؟",
            "حلو — عطني رقم المرجع لو سمحت.",
            "طيب، رقم الفاتورة كم؟",
        ),
    },
}

# Intent-specific overrides (e.g. "pay" instead of "transfer" for the amount).
_SLOT_PROMPTS_BY_INTENT: dict[Intent, dict[str, dict[Language, tuple[str, ...]]]] = {
    Intent.PAY_BILL: {
        "amount": {
            Language.EN: (
                "Sure — how much would you like to pay?",
                "Of course. How much are we paying?",
                "Happy to — what amount should I pay?",
            ),
            Language.AR: (
                "تمام — كم المبلغ اللي تحب تدفعه؟",
                "أبشر — كم تبي تدفع؟",
                "طيب، وش المبلغ المطلوب؟",
            ),
        },
    },
}

_CHOOSE_ACTION: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "Happy to help! Would you like to (1) send money or (2) pay a bill?",
        "Sure, I can help with that — (1) a transfer or (2) a bill payment?",
        "I'm on it. Are we (1) sending money or (2) paying a bill?",
    ),
    Language.AR: (
        "حياك الله! تحب (١) تحويل فلوس أو (٢) دفع فاتورة؟",
        "أبشر — نسوّي (١) تحويل ولا (٢) دفع فاتورة؟",
        "تمام، أنا معك. (١) تحويل فلوس ولا (٢) فاتورة؟",
    ),
}

# A question about a service the assistant does not carry. It must not invent
# the answer (fees, policies) — it says so, points the customer at customer
# service for the information, and names what it can actually do.
_OUT_OF_SCOPE: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "I don't have that information \U0001f642 — customer service can help you "
        "with it. I can send money, pay bills, check your balance, and manage "
        "your beneficiaries.",
        "That one isn't with me \U0001f642 — please ask customer service about it. "
        "What I can do is transfers, bill payments, your balance, and your "
        "beneficiaries.",
        "I can't answer that one \U0001f642 — customer service is the place for it. "
        "I'm here for transfers, bills, balances, and beneficiaries.",
    ),
    Language.AR: (
        "المعلومة دي مو عندي \U0001f642 — كلّم خدمة العملاء يفيدونك فيها، "
        "وأنا أقدر أساعدك في التحويل، سداد الفواتير، الرصيد، "
        "والمستفيدين.",
        "هالموضوع مو من خدماتي \U0001f642 — تراجع خدمة العملاء للاستفسار، "
        "واللي أقدر عليه: تحويل فلوس، سداد فواتير، الرصيد، "
        "والمستفيدين.",
        "ما عندي إجابة لهالسؤال \U0001f642 — خدمة العملاء هم الأقدر عليه، "
        "وأنا هنا للتحويل، الفواتير، الرصيد، والمستفيدين.",
    ),
}

# A "how do I …" question about something the assistant *does* carry: explain
# the flow by example instead of claiming the service is missing.
_HOW_TO: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "I can do that for you right here \U0001f642 — just tell me the amount "
        "and who it's for, e.g. \"send 500 to Omar\", and I'll take it from "
        "there.",
    ),
    Language.AR: (
        "أقدر أسويها لك من هنا \U0001f642 — قل لي المبلغ واسم اللي تحول له، "
        'مثلاً "حول ٥٠٠ لعمر"، وأنا أكمل معك البقية.',
    ),
}

_GREETING: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "Of course — let's get your transfer sorted.",
        "Sure thing — let's set up that transfer.",
        "Happy to help — let's get this transfer done.",
    ),
    Language.AR: (
        "أبشر، خلنا نجهّز التحويل.",
        "حياك الله، نجهّز التحويل الآن.",
        "تمام، خلنا نكمّل التحويل.",
    ),
}

# Shown by a channel when the conversation opens, before the customer has said
# anything. Names only what the assistant can actually do, so the first line is
# never a promise the engine can't keep.
_OPENING: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "Hey there 👋 I can transfer money, tell you your balance, or pay a "
        "bill for you — what do you need?",
        "Hi 👋 Transfers, your balance, bill payments — I'm at your service. "
        "What can I do?",
        "Welcome 👋 I can send money, check your balance, or settle a bill — "
        "just say the word.",
    ),
    Language.AR: (
        "هلا والله 👋 حيّاك! أقدر أحوّل لك فلوس، أقولك كم رصيدك، أو أسدّد "
        "فاتورتك — آمر.",
        "أهلين وسهلين 👋 تحويل، رصيد، سداد فواتير — أنا جاهز. وش تبي؟",
        "حيّاك الله 👋 أقدر أحوّل، أعلّمك برصيدك، أو أدفع فاتورتك — أمرك.",
    ),
}

_FALLBACK: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "Hmm, I didn't quite catch that 🤔 — I can send money or pay a "
        'bill. For example, try "send 500 SAR to Ahmed" or "pay my STC bill".',
        "Sorry, that one's outside what I do 🤔 — I handle transfers and bills. "
        'Try "send 500 SAR to Ahmed" or "pay my STC bill".',
        "I'm not sure I follow 🤔 — transfers and bill payments are my thing. "
        'For example: "send 500 SAR to Ahmed" or "pay my STC bill".',
    ),
    Language.AR: (
        "لم أفهم تماماً 🤔 — أستطيع تحويل الأموال أو دفع الفواتير. "
        'جرّب مثلاً: "حوّل ٥٠٠ ريال إلى أحمد" أو "ادفع فاتورة STC".',
        "ما ضبطت عليك 🤔 — أنا أساعدك في التحويلات ودفع الفواتير. "
        'جرّب: "حوّل ٥٠٠ ريال إلى أحمد" أو "ادفع فاتورة STC".',
        "هذي بره اللي أقدر عليه 🤔 — تخصصي التحويلات والفواتير. "
        'مثلاً: "حوّل ٥٠٠ ريال إلى أحمد" أو "ادفع فاتورة STC".',
    ),
}

# Warm chit-chat replies, keyed by a small-talk kind. Each one stays helpful by
# gently steering the customer back to what the assistant can do.
_SMALL_TALK: dict[str, dict[Language, tuple[str, ...]]] = {
    "greeting": {
        Language.EN: (
            "Hey! 👋 Good to see you. I can send money or pay a bill "
            "for you — what's up?",
            "Hi there! 👋 I can move money or settle a bill — what do you need?",
            "Hello! 👋 Transfers and bill payments are my thing — how can I help?",
        ),
        Language.AR: (
            "هلا والله! 👋 سعيد إني أشوفك. أقدر أحوّل لك فلوس أو أدفع "
            "فاتورة — وش تحتاج؟",
            "حياك الله! 👋 أقدر أحوّل فلوس أو أسدّد فاتورة — وش تبي؟",
            "أهلين! 👋 تحويل ولا فاتورة؟ أنا جاهز.",
        ),
    },
    "thanks": {
        Language.EN: (
            "Anytime! 😊 Need anything else — a transfer or a bill?",
            "Anytime, glad to help 😊 Anything else — a transfer or a bill?",
            "Anytime you like 😊 Want to do a transfer or pay a bill?",
        ),
        Language.AR: (
            "على الرحب! 😊 تحتاج شي ثاني — تحويل أو فاتورة؟",
            "لا شكر على واجب 😊 تبي شي ثاني — تحويل ولا فاتورة؟",
            "حاضر دايمًا 😊 عندك شي ثاني — تحويل أو فاتورة؟",
        ),
    },
    "how_are_you": {
        Language.EN: (
            "I'm good, thanks for asking! 😄 So, wanna send some money "
            "or pay a bill?",
            "Doing great, thanks for asking! 😄 Shall we send money or pay a bill?",
            "All good here, thanks for asking 😄 Fancy a transfer or a bill payment?",
        ),
        Language.AR: (
            "تمام والحمد لله، تسلم على السؤال! 😄 تبي تحوّل فلوس أو تدفع " "فاتورة؟",
            "بخير والحمد لله، تسلم على السؤال 😄 نسوّي تحويل ولا فاتورة؟",
            "كله تمام، الله يسلمك 😄 تحب تحويل أو دفع فاتورة؟",
        ),
    },
    "bye": {
        Language.EN: (
            "Catch you later! 👋 I'm around whenever you wanna send "
            "money or pay a bill.",
            "See you! 👋 I'm here whenever you need a transfer or a bill paid.",
            "Take care! 👋 Come back anytime for a transfer or a bill.",
        ),
        Language.AR: (
            "نشوفك على خير! 👋 أنا موجود وقت ما تبي تحويل أو فاتورة.",
            "في أمان الله! 👋 تلقاني وقت ما تحتاج تحويل أو فاتورة.",
            "مع السلامة! 👋 أنا حاضر أي وقت لتحويل أو فاتورة.",
        ),
    },
    "capability": {
        Language.EN: (
            "I'm your banking assistant 🙌 I can send money, pay bills, "
            "check your balance and manage your beneficiaries — which one?",
            "I'm your banking assistant 🙌 transfers, bill payments, your balance "
            "and your beneficiaries — what would you like?",
            "Banking assistant here 🙌 I move money, settle bills, read your "
            "balance and look after your beneficiaries — where do we start?",
        ),
        Language.AR: (
            "أنا مساعدك البنكي 🙌 أقدر أحوّل فلوس، أدفع فواتير، "
            "أجيب رصيدك، وأدير مستفيدينك — وش تحب؟",
            "أنا مساعدك البنكي 🙌 تحويلات، فواتير، رصيدك، ومستفيدينك — وش تبي؟",
            "مساعدك البنكي حاضر 🙌 أحوّل، أسدّد فواتير، أجيب الرصيد، وأرتّب "
            "المستفيدين — نبدأ بوش؟",
        ),
    },
    "default": {
        Language.EN: (
            "Love a good chat! 😊 I'm best with money transfers and "
            "bills though — wanna give one a go?",
            "I enjoy the chat 😊 but transfers and bills are what I do best — "
            "shall we try one?",
            "Happy to chat 😊 though I shine at transfers and bill payments — "
            "want to start one?",
        ),
        Language.AR: (
            "يسعدني السوالف! 😊 بس أنا أشطر في التحويلات ودفع الفواتير " "— نجرّب وحدة؟",
            "السوالف حلوة 😊 بس شغلتي التحويلات والفواتير — نجرّب؟",
            "أنا معك في السوالف 😊 لكن قوّتي في التحويل ودفع الفواتير — نبدأ؟",
        ),
    },
}

# Keyword cues used to pick the right warm reply (matched on normalized tokens).
# Order matters: the first kind whose cues appear wins, so the kinds run from the
# least ambiguous cues to the most. "thanks for your help" shares "help" with the
# capability cues, and "who are you" shares "are"/"you" with the how-are-you
# cues, so thanks precedes capability and capability precedes how-are-you.
_SMALL_TALK_CUES: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "thanks",
        frozenset({"thanks", "thank", "thx", "شكرا", "مشكور", "تسلم", "يعطيك"}),
    ),
    (
        # "who are you" / "what can you do" / "وش تقدر تسوي": asks what the
        # assistant is for. Safe next to real requests because a message only
        # counts as chit-chat when *every* token is a cue or a filler.
        "capability",
        frozenset(
            {
                "who",
                "what",
                "can",
                "do",
                "help",
                "talking",
                "مين",
                "انت",
                "من",
                "تقدر",
                "تسوي",
                "تسويها",
                "تساعدني",
                "تعاوني",
                "الاشياء",
            }
        ),
    ),
    (
        "how_are_you",
        frozenset(
            {
                "how",
                "are",
                "you",
                "doing",
                "going",
                "كيف",
                "حالك",
                "عامل",
                "ايه",
                "اخبارك",
            }
        ),
    ),
    (
        "bye",
        frozenset({"bye", "goodbye", "night", "later", "وداعا", "باي", "اشوفك"}),
    ),
    (
        "greeting",
        frozenset(
            {
                "hi",
                "hallo",
                "afternoon",
                "اهلين",
                "هلا",
                "سلام",
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
                "حياك",
                "حياكم",
            }
        ),
    ),
)

_CANCELLED: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "No problem, I've cancelled that for you.",
        "Done — I've cancelled it.",
        "Sure, cancelled. Nothing has been sent.",
    ),
    Language.AR: (
        "ولا يهمّك، ألغيت العملية.",
        "تمام، ألغيتها.",
        "أبشر، ألغيت العملية وما انرسل شي.",
    ),
}


def slot_prompt(slot: str, language: Language, intent: Intent | None = None) -> str:
    if intent is not None:
        override = _SLOT_PROMPTS_BY_INTENT.get(intent, {}).get(slot)
        if override is not None:
            return phrasing.varied(
                f"slot_prompt:{intent.value}:{slot}", override, language
            )
    variants = _SLOT_PROMPTS.get(slot)
    if variants is None:
        return f"Please provide the {slot}."
    return phrasing.varied(f"slot_prompt:{slot}", variants, language)


def choose_action(language: Language) -> str:
    return phrasing.varied("choose_action", _CHOOSE_ACTION, language)


def choose_biller(names: list[str], language: Language) -> str:
    """Ask the customer which biller they meant when a term is ambiguous."""

    listing = "  ".join(f"({i + 1}) {name}" for i, name in enumerate(names))
    if language is Language.AR:
        return f"وجدت أكثر من جهة لهذه الفاتورة — أيها تقصد؟ {listing}"
    return f"I found more than one biller for that — which one? {listing}"


_BILLER_CATEGORY_AR: dict[str, str] = {
    "Utilities": "الخدمات (كهرباء/مياه/غاز)",
    "Telecom & Internet": "الاتصالات والإنترنت",
    "Government Services": "الخدمات الحكومية",
    "Education": "التعليم",
    "Banking & Finance": "البنوك والتمويل",
    "Insurance": "التأمين",
    "Travel & Transportation": "السفر والنقل",
    "Media & Entertainment": "الإعلام والترفيه",
    "Services": "خدمات أخرى",
}


def biller_not_found(name: str, categories: list[str], language: Language) -> str:
    """The named biller isn't in the SADAD catalogue — say so and re-ask.

    States plainly that this is a bill payment and the biller is unknown, then
    lists the supported categories so the customer can name a valid one.
    """

    if language is Language.AR:
        listing = "، ".join(_BILLER_CATEGORY_AR.get(c, c) for c in categories)
        return (
            f'هذه عملية دفع فاتورة، لكن "{name}" غير موجود في قائمة المزوّدين '
            f"لدينا. الفئات المتاحة: {listing}. "
            "اكتب اسم المزوّد الصحيح أو رقم سداد الخاص به."
        )
    listing = ", ".join(categories)
    return (
        f'This is a bill payment, but "{name}" isn\'t in our list of billers. '
        f"Available categories: {listing}. "
        "Please give the correct biller name or its SADAD code."
    )


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
    options: list[tuple[str, str, str, str]],
    language: Language,
    asked: str = "",
) -> str:
    """Ask which beneficiary is meant when a name matches several people.

    ``options`` is a list of ``(name, bank, masked_account, currency)`` tuples.
    ``asked`` is the name the customer typed: quoting a candidate instead would
    name a person they never mentioned, since a match can be on any part of a
    saved name.
    """

    listing = _option_lines(options)
    first = asked.strip() or (options[0][0].split()[0] if options else "")
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


def confirm_beneficiary_match(
    options: list[tuple[str, str, str, str]],
    language: Language,
    asked: str = "",
) -> str:
    """Ask whether the one match — matched deeper in its name — is the person.

    Nobody is saved under the word the customer typed; it turned up inside
    somebody else's name. Naming that person and waiting is the only safe move:
    silently confirming a transfer to them would put a name in the confirmation
    the customer never asked for.
    """

    listing = _option_lines(options)
    if language is Language.AR:
        return (
            f'ما عندي أحد محفوظ باسم "{asked}" بالضبط — أقرب واحد عندي:\n'
            f"{listing}\n"
            "هو المقصود؟ قل لي الرقم (١)، أو اكتب الاسم الصحيح."
        )
    return (
        f'Nobody is saved as "{asked}" exactly — the closest I have is:\n'
        f"{listing}\n"
        "Is that who you meant? Reply with the number (1), or type the right name."
    )


_BENEFICIARY_NOT_FOUND: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        'Hmm, I don\'t have anyone saved as "{name}" yet 🤔 — send their '
        "account/IBAN to add them, type another beneficiary name, or say "
        '"cancel".',
        'I can\'t find "{name}" in your beneficiaries 🤔 — you can send the '
        "account/IBAN, give me a different beneficiary name, or cancel this "
        "transfer.",
        '"{name}" isn\'t saved yet 🤔 — send the account/IBAN to add them, '
        'enter another beneficiary, or say "cancel".',
    ),
    Language.AR: (
        'ما لقيت أحد محفوظ باسم "{name}" 🤔 — إذا تبي تضيفه أرسل رقم الحساب/'
        'الآيبان، أو اكتب اسم مستفيد ثاني، أو قل "إلغاء".',
        'ما لقيت "{name}" في مستفيدينك 🤔 — عطني رقم الحساب/الآيبان لإضافته، '
        'أو اكتب اسم مستفيد ثاني، أو قل "إلغاء".',
        '"{name}" مو محفوظ عندك للحين 🤔 — أرسل رقم الحساب أو الآيبان لإضافته، '
        'أو اكتب اسم ثاني، أو قل "إلغاء".',
    ),
}


def beneficiary_not_found(name: str, language: Language) -> str:
    """No beneficiary matched; offer to add one via the external API."""

    return phrasing.varied(
        "beneficiary_not_found", _BENEFICIARY_NOT_FOUND, language, name=name
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


_ASK_BENEFICIARY_NAME: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "Sure — what's the name of the beneficiary you'd like to add?",
        "Of course. Who are we adding?",
        "Happy to — what name should I save them under?",
    ),
    Language.AR: (
        "تمام — وش اسم المستفيد اللي تبي تضيفه؟",
        "أبشر — مين اللي نضيفه؟",
        "طيب، عطني اسم المستفيد.",
    ),
}

_ASK_BENEFICIARY_ACCOUNT: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        'And what\'s the IBAN or account number for "{name}"?',
        'Great — what IBAN or account number should I save for "{name}"?',
        "Now the account: what's \"{name}\"'s IBAN or account number?",
    ),
    Language.AR: (
        'وش رقم الآيبان أو الحساب الخاص بـ "{name}"؟',
        'تمام — عطني آيبان "{name}" أو رقم حسابه.',
        'والحساب؟ وش آيبان "{name}" أو رقم حسابه؟',
    ),
}


def ask_beneficiary_name(language: Language) -> str:
    """Opening question of the standalone add-beneficiary flow."""

    return phrasing.varied("ask_beneficiary_name", _ASK_BENEFICIARY_NAME, language)


def ask_beneficiary_account(name: str, language: Language) -> str:
    return phrasing.varied(
        "ask_beneficiary_account", _ASK_BENEFICIARY_ACCOUNT, language, name=name
    )


# Why an account was rejected -> how to fix it, per language.
_ACCOUNT_ERRORS: dict[str, dict[Language, str]] = {
    "iban_length": {
        Language.EN: (
            "that IBAN isn't the right length — a Saudi IBAN is 24 characters "
            "(SA + 22 digits)"
        ),
        Language.AR: ("طول الآيبان غير صحيح — الآيبان السعودي 24 خانة (SA + 22 رقمًا)"),
    },
    "iban_checksum": {
        Language.EN: (
            "that IBAN failed its checksum, so there's a typo in it somewhere"
        ),
        Language.AR: "الآيبان ما نجح في التحقق الحسابي، يعني فيه خطأ مطبعي",
    },
    "too_short": {
        Language.EN: "that's too short for an account number",
        Language.AR: "هذا قصير جدًا ليكون رقم حساب",
    },
    "not_an_account": {
        Language.EN: "that doesn't look like an IBAN or an account number",
        Language.AR: "هذا ما يبدو آيبان ولا رقم حساب",
    },
}


_INVALID_ACCOUNT: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "Sorry, {detail}. Please send a valid IBAN (e.g. SA0380000000608010167519) "
        'or an account number to add "{name}", or reply "no" to cancel.',
        "Hmm, {detail}. Send a valid IBAN (e.g. SA0380000000608010167519) or an "
        'account number for "{name}", or reply "no" to cancel.',
        "That didn't work — {detail}. Try a valid IBAN (e.g. "
        'SA0380000000608010167519) or an account number for "{name}", or say "no".',
    ),
    Language.AR: (
        "{detail}. أرسل آيبان صحيح (مثل SA0380000000608010167519) أو رقم "
        'حساب لإضافة "{name}"، أو اكتب "لا" للإلغاء.',
        "عذرًا، {detail}. عطني آيبان صحيح (مثل SA0380000000608010167519) أو رقم "
        'حساب لـ "{name}"، أو اكتب "لا" للإلغاء.',
        "ما مشى — {detail}. جرّب آيبان صحيح (مثل SA0380000000608010167519) أو رقم "
        'حساب لـ "{name}"، أو اكتب "لا".',
    ),
}


def beneficiary_invalid_account(name: str, reason: str, language: Language) -> str:
    """Explain precisely why the account was rejected and re-ask."""

    detail = _ACCOUNT_ERRORS.get(reason, _ACCOUNT_ERRORS["not_an_account"])[language]
    return phrasing.varied(
        "beneficiary_invalid_account",
        _INVALID_ACCOUNT,
        language,
        detail=detail,
        name=name,
    )


def _typo_location(hint: accounts.IbanTypoHint, language: Language) -> str:
    """Name the spot only when the arithmetic pins it down, never a guess."""

    if hint.swapped is not None:
        first, second = hint.swapped
        if language is Language.AR:
            return f"يبدو أن الخانتين {first} و{second} متبادلتان"
        return f"characters {first} and {second} look swapped"
    if len(hint.positions) == 1:
        position = hint.positions[0]
        if language is Language.AR:
            return f"الخطأ على الأرجح في الخانة رقم {position}"
        return f"the typo is most likely at character {position}"
    if language is Language.AR:
        return "خانة واحدة فيها خطأ، لكن ما أقدر أحدد أيها بالضبط"
    return "one character is wrong, though I can't tell which"


def beneficiary_iban_typo(
    name: str, hint: accounts.IbanTypoHint, language: Language
) -> str:
    """A failed checksum reads as a typo, with a way through if it isn't one."""

    where = _typo_location(hint, language)
    if language is Language.AR:
        return (
            f"الآيبان ما ضبط في التحقق الحسابي — {where}. راجعه مع كشف الحساب "
            f'وأرسله من جديد لإضافة "{name}"، أو اكتب "أنا متأكد" لأكمل بالرقم '
            'زي ما كتبته، أو "لا" للإلغاء.'
        )
    return (
        f"That IBAN doesn't pass its checksum — {where}. Check it against your "
        f'statement and send it again to add "{name}", or reply "I\'m sure" and '
        'I\'ll use it exactly as you typed it, or "no" to cancel.'
    )


def unchecked_account_note(overridden: bool, language: Language) -> str:
    """Restate, at the point of no return, that the IBAN was never verified."""

    if not overridden:
        return ""
    if language is Language.AR:
        return "⚠️ الآيبان ما نجح في التحقق الحسابي وهنستخدمه زي ما كتبته."
    return "⚠️ This IBAN failed its checksum; I'll use it exactly as you typed it."


def confirm_add_beneficiary(name: str, masked: str, language: Language) -> str:
    if language is Language.AR:
        return f"للتأكيد — أضيف المستفيد {name} على الحساب {masked}؟ (نعم/لا)"
    return f"Just to confirm — add {name} with account {masked}? (yes/no)"


def confirm_add_then_transfer(
    name: str, masked: str, amount: str, currency: str, language: Language
) -> str:
    """One question for the mid-transfer case: save them *and* send the money."""

    if language is Language.AR:
        return (
            f"للتأكيد — أضيف {name} على الحساب {masked} وأحوّل له "
            f"{amount} {currency}؟ (نعم/لا)"
        )
    return (
        f"Just to confirm — add {name} with account {masked} and send them "
        f"{amount} {currency}? (yes/no)"
    )


def beneficiary_add_completed(name: str, masked: str, language: Language) -> str:
    if language is Language.AR:
        return f"تمّ ✅ — أضفت {name} ({masked}) لقائمة المستفيدين. " "تحب تحوّل له الآن؟"
    return (
        f"Done ✅ — {name} ({masked}) is saved to your beneficiaries. "
        "Want to send them money now?"
    )


_ADD_INVALID_ACCOUNT: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "That doesn't look like a valid account number or IBAN. Send a full "
        'IBAN (e.g. SA followed by digits) or an account number to add "{name}", '
        'or reply "no" to cancel.',
        "I couldn't read that as a valid account number or IBAN. Send a full IBAN "
        '(SA followed by digits) or an account number for "{name}", or reply "no".',
        "That isn't a valid account number or IBAN yet. A full IBAN (SA followed by "
        'digits) or an account number works for "{name}" — or reply "no" to cancel.',
    ),
    Language.AR: (
        "هذا لا يبدو رقم حساب أو آيبان صالحًا. أرسل آيبان كاملًا (مثل SA "
        'متبوعة بأرقام) أو رقم حساب لإضافة "{name}"، أو اكتب "لا" للإلغاء.',
        "ما قدرت أقراه كرقم حساب أو آيبان صالح. أرسل آيبان كامل (SA وبعدها "
        'أرقام) أو رقم حساب لـ "{name}"، أو اكتب "لا".',
        "لسه ما وصلني رقم حساب أو آيبان صالح. آيبان كامل (SA وبعدها أرقام) "
        'أو رقم حساب يكفي لـ "{name}"، أو اكتب "لا" للإلغاء.',
    ),
}


def beneficiary_add_invalid_account(name: str, language: Language) -> str:
    """The reply wasn't a plausible account number / IBAN."""

    return phrasing.varied(
        "beneficiary_add_invalid_account", _ADD_INVALID_ACCOUNT, language, name=name
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


_NO_BENEFICIARIES: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "You don't have any saved beneficiaries yet 🙂 — want to add one? "
        "Just tell me the name and their account/IBAN.",
        "You don't have any saved beneficiaries so far 🙂 — shall we add one? "
        "Give me a name and an account/IBAN.",
        "Nothing saved yet — you don't have any saved beneficiaries 🙂 Want to add "
        "the first one? Send me a name and an account/IBAN.",
    ),
    Language.AR: (
        "ما عندك مستفيدين محفوظين للحين 🙂 — تحب تضيف واحد؟ "
        "قل لي الاسم ورقم الحساب/الآيبان.",
        "قائمة المستفيدين فارغة للحين 🙂 — نضيف واحد؟ عطني الاسم ورقم "
        "الحساب أو الآيبان.",
        "ما عندك مستفيدين محفوظين بعد 🙂 تبي نضيف أول واحد؟ أرسل الاسم " "والآيبان.",
    ),
}

_BENEFICIARIES_UNAVAILABLE: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "I couldn't fetch your beneficiaries right now. Please try again later.",
        "I couldn't fetch your beneficiaries at the moment — please try again "
        "shortly.",
        "Your beneficiary list isn't reachable right now. Please try again in a bit.",
    ),
    Language.AR: (
        "تعذّر جلب قائمة المستفيدين الآن. حاول مرة أخرى لاحقًا.",
        "ما قدرت أجيب قائمة المستفيدين حاليًا — جرّب بعد شوية.",
        "قائمة المستفيدين ما هي متاحة اللحظة. حاول مرة ثانية بعد قليل.",
    ),
}


def no_beneficiaries(language: Language) -> str:
    """The customer has no saved beneficiaries yet."""

    return phrasing.varied("no_beneficiaries", _NO_BENEFICIARIES, language)


def beneficiaries_unavailable(language: Language) -> str:
    """The beneficiary directory could not be reached."""

    return phrasing.varied(
        "beneficiaries_unavailable", _BENEFICIARIES_UNAVAILABLE, language
    )


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


_BALANCE_UNAVAILABLE: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "I couldn't fetch your balance right now. Please try again later.",
        "I couldn't reach your balance at the moment — please try again shortly.",
        "Your balance isn't available right now. Please try again in a bit.",
    ),
    Language.AR: (
        "تعذّر جلب الرصيد الآن. حاول مرة أخرى لاحقًا.",
        "ما قدرت أجيب الرصيد حاليًا — جرّب بعد شوية.",
        "الرصيد ما هو متاح اللحظة. حاول مرة ثانية بعد قليل.",
    ),
}

_RESUME_NOTE: dict[Language, tuple[str, ...]] = {
    Language.EN: (
        "Okay, back to where we were —",
        "Right, back to your transfer —",
        "Now, where we left off —",
    ),
    Language.AR: (
        "طيّب، نرجع للتحويل —",
        "تمام، نكمّل من وين وقّفنا —",
        "أبشر، نرجع للطلب —",
    ),
}


def _account_label(account_type: str) -> str:
    """The account's display name, in English in both languages.

    A display name identifies the account on the customer's statements, so it
    is printed as the bank writes it rather than translated per turn.
    """

    return f"{account_type.capitalize()} Account"


def choose_source_account(
    accounts: list[tuple[str, str, str, str]], language: Language
) -> str:
    """Ask which account the money leaves from.

    ``accounts`` is a list of ``(account_type, masked_number, balance,
    currency)`` tuples, every value straight from the Banking Core: the
    assistant numbers them and prints nothing of its own.
    """

    lines = [
        f"{i}. {_account_label(account_type)} {masked} — {balance} {currency}"
        for i, (account_type, masked, balance, currency) in enumerate(accounts, start=1)
    ]
    listing = "\n".join(lines)
    if language is Language.AR:
        return f"من أي حساب تبي تحول؟\n{listing}"
    return f"Which account do you want to transfer from?\n{listing}"


def choose_transfer_purpose(
    purposes: list[str], language: Language, recipient: str = ""
) -> str:
    """Ask what the transfer is for, as a pick from the numbered list."""

    listing = "\n".join(f"{i}. {label}" for i, label in enumerate(purposes, start=1))
    if language is Language.AR:
        title = (
            f"وش غرض التحويل لـ {recipient}؟ اختر من القائمة:"
            if recipient
            else "وش غرض التحويل؟ اختر من القائمة:"
        )
        return f"{title}\n{listing}"
    title = (
        f"What's the transfer purpose for {recipient}? Choose from the list:"
        if recipient
        else "What's the transfer purpose? Choose from the list:"
    )
    return f"{title}\n{listing}"


def balance_unavailable(language: Language) -> str:
    return phrasing.varied("balance_unavailable", _BALANCE_UNAVAILABLE, language)


def resume_note(language: Language) -> str:
    """Short connector shown before re-emitting an in-progress prompt."""

    return phrasing.varied("resume_note", _RESUME_NOTE, language)


def warnings_note(warnings: list[str], language: Language) -> str:
    """Turn raw pre-flight warning codes into a short bilingual advisory line."""

    notes: list[str] = []
    for warning in warnings:
        if warning.startswith("fx"):
            detail = warning.split(":", 1)[1].strip() if ":" in warning else ""
            if language is Language.AR:
                notes.append(f"ℹ️ سيتم تحويل العملة ({detail}).")
            else:
                notes.append(f"ℹ️ A currency conversion applies ({detail}).")
    return " ".join(notes)


def insufficient_funds(
    requested: str, available: str, currency: str, language: Language
) -> str:
    """Refuse a debit the balance can't fund and offer the balance instead.

    Both figures come from the Banking Core and are printed exactly as given.
    """

    if language is Language.AR:
        return (
            f"رصيدك {available} {currency} وما يكفي لـ{requested} {currency}، "
            f"فما أقدر أنفّذ التحويل. أحوّل {available} {currency} بدالها؟ "
            "(نعم/لا) أو اكتب مبلغ تاني."
        )
    return (
        f"Your balance is {available} {currency}, which doesn't cover "
        f"{requested} {currency}, so I can't put this through. Shall I send "
        f"{available} {currency} instead? (yes/no) — or give me another amount."
    )


def preflight_blocked(language: Language) -> str:
    """Refuse for any other Banking Core reason (e.g. an inactive account)."""

    if language is Language.AR:
        return "ما أقدر أكمل العملية على هذا الحساب — خدمة العملاء تقدر توضّح السبب."
    return (
        "I can't put this through on that account — customer support can tell "
        "you why."
    )


def greeting(language: Language) -> str:
    return phrasing.varied("greeting", _GREETING, language)


def opening(language: Language) -> str:
    """The line a channel shows before the customer has typed anything."""

    return phrasing.varied("opening", _OPENING, language)


def fallback(language: Language) -> str:
    return phrasing.varied("fallback", _FALLBACK, language)


def how_to_transact(language: Language) -> str:
    """Explain how to start a flow the assistant does carry."""

    return phrasing.varied("how_to_transact", _HOW_TO, language)


def out_of_scope(language: Language, turn: str | None = None) -> str:
    """Answer a question about a service the assistant does not carry.

    With the customer's ``turn`` the LLM writes the decline itself, so it can
    open on what they said before saying the information is not ours; without it
    (or with the LLM off) the fixed wording is sent.
    """

    template = phrasing.pick("out_of_scope", _OUT_OF_SCOPE[language])
    if turn is None:
        return phrasing.rewrite("out_of_scope", template, language)
    return phrasing.declined("out_of_scope", turn, template, language)


def cancelled(language: Language) -> str:
    return phrasing.varied("cancelled", _CANCELLED, language)


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

    if count <= 1 or not settings.reply_variation_enabled:
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
    return phrasing.rewrite("inappropriate", reply, language), index


def repeat_offense(language: Language) -> str:
    return _REPEAT_OFFENSE[language]


# Filler words allowed in an otherwise pure chit-chat message ("hello there").
# Normalized so Arabic letter-form folding (ة → ه, ى → ي) matches the tokens.
_SMALL_TALK_FILLERS: frozenset[str] = frozenset(
    normalize(w)
    for w in {
        "there",
        "for",
        "your",
        "وش",
        "خير",
        "العافية",
        "ورحمة",
        "friend",
        "buddy",
        "sir",
        "dear",
        "please",
        "the",
        "a",
        "good",
        "lot",
        "much",
        "many",
        "so",
        "very",
        "nice",
        "me",
        "my",
        "i",
        "am",
        "is",
        "it",
        "to",
        "out",
        "could",
        "would",
        "see",
        "later",
        "night",
        "لك",
        "على",
        "المساعدة",
        "وسهلا",
        "الله",
        "عليكم",
        "الحال",
        "الخير",
        "اللي",
        "فيه",
        "بك",
    }
)
_SMALL_TALK_CUES_NORM: tuple[tuple[str, frozenset[str]], ...] = tuple(
    (kind, frozenset(normalize(cue) for cue in cues)) for kind, cues in _SMALL_TALK_CUES
)
_ALL_SMALL_TALK_CUES: frozenset[str] = frozenset(
    tok for _, cues in _SMALL_TALK_CUES_NORM for tok in cues
)


def _small_talk_kind(text: str) -> str | None:
    tokens = set(normalize_tokens(text))
    if not tokens:
        return None
    # Only treat as chit-chat when the message is *purely* greeting/thanks/etc.
    # (cue tokens + a few fillers), so "hi I want to pay a bill" still routes.
    if not tokens & _ALL_SMALL_TALK_CUES:
        return None
    if tokens - _ALL_SMALL_TALK_CUES - _SMALL_TALK_FILLERS:
        return None
    for kind, cues in _SMALL_TALK_CUES_NORM:
        if tokens & cues:
            return kind
    return None


def is_small_talk(text: str) -> bool:
    """True when ``text`` is *purely* chit-chat (greeting/thanks/bye/etc.)."""

    return _small_talk_kind(text) is not None


def small_talk(text: str, language: Language) -> str:
    """Return a warm chit-chat reply, picking the kind from ``text`` cues."""

    kind = _small_talk_kind(text) or "default"
    return phrasing.varied(f"small_talk:{kind}", _SMALL_TALK[kind], language)


def one_payee_at_a_time(names: list[str], language: Language) -> str:
    """Ask which of the payees the sentence listed to send to now.

    Which amount belongs to which name is the customer's to state, so the names
    are read back exactly as typed and nothing is assumed about the amounts.
    """

    if language is Language.AR:
        listing = " ولا ".join(names)
        return (
            f"أقدر أنفّذ تحويل واحد في كل مرة. لمين نبدأ — {listing}؟ "
            "والباقي نسويه بطلب ثاني."
        )
    listing = " or ".join(names)
    return (
        f"I can make one transfer at a time. Who should this one go to — {listing}? "
        "We'll do the rest in a separate request."
    )


def confirm_prompt(
    amount: str, currency: str, recipient: str, language: Language
) -> str:
    if language is Language.AR:
        return f"تأكيد بس — أحوّل {amount} {currency} لـ {recipient}؟ (إيه/لا)"
    return f"Just to confirm — send {amount} {currency} to {recipient}? (yes/no)"


# One wording per language: the executed action is what the customer's history,
# the logs and the tests all quote, so it is never varied or paraphrased.
_COMPLETED: dict[Language, str] = {
    Language.EN: (
        "All set! {amount} {currency} was transferred to {recipient} successfully."
    ),
    Language.AR: "تم التحويل بنجاح ✅ حوّلت {amount} {currency} إلى {recipient}.",
}


def completed(amount: str, currency: str, recipient: str, language: Language) -> str:
    return _COMPLETED[language].format(
        amount=amount, currency=currency, recipient=recipient
    )


def bill_confirm_prompt(
    amount: str, currency: str, biller: str, reference: str, language: Language
) -> str:
    if language is Language.AR:
        return (
            f"تأكيد بس — أسدّد {amount} {currency} لفاتورة {biller} "
            f"(مرجع {reference})؟ (إيه/لا)"
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
            f"تم سداد فاتورة {biller} بنجاح ✅ بمبلغ {amount} {currency} "
            f"(مرجع {reference})."
        )
    return (
        f"Bill paid successfully ✅ {amount} {currency} was paid to {biller} "
        f"(ref {reference})."
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


def topic_answer(
    text: str, evidence: TopicEvidence, language: Language
) -> topic_replies.TopicAnswer | None:
    """Answer a refused customer-service question in its own topic.

    ``None`` when the retrieval is not decisive enough or the topic has no
    reviewed answer, so the caller keeps the generic prompt. Deliberately
    money-critical even though it carries no figure: each answer states what
    this assistant can and cannot do, and a re-worded "I can't reverse a charge"
    would be a false promise.
    """

    return topic_replies.decide(
        text,
        evidence.top_score,
        evidence.votes,
        evidence.retrieved,
        language,
        evidence.prediction,
    )
