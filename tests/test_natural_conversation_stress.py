"""Ordinary conversation, at volume, must never invent a money movement.

Every reported fault so far arrived as one sentence at a time: a greeting read
as a payee, a transfer request read as an admin task. This gate composes the
surface forms instead of storing them — ten families of Saudi/Gulf, Egyptian,
MSA, English, code-switched, noisy, ambiguous and everyday talk — and asserts
the invariants those faults broke: chit-chat never opens a financial write, and
an aside mid-transfer never loses the slot the assistant is waiting on.

``NLU_STRESS_CASES_PER_FAMILY`` scales the gate: the suite runs a sample, the
release run sets it to 2000 for the full twenty thousand conversations.
"""

from __future__ import annotations

import os
import re

import pytest

from app.conversation.engine import ConversationEngine, ConversationResult
from app.conversation.state import ConversationStatus
from app.schemas import Intent

FAMILY_COUNT = 10
CASES_PER_FAMILY = int(os.getenv("NLU_STRESS_CASES_PER_FAMILY", "120"))

WRITE_INTENTS = frozenset(
    {Intent.TRANSFER_MONEY, Intent.PAY_BILL, Intent.ADD_BENEFICIARY}
)


@pytest.fixture(scope="module")
def engine() -> ConversationEngine:
    """One engine for the whole gate: this stresses state, not model loading."""

    return ConversationEngine()


def _turn(engine: ConversationEngine, session: str, text: str) -> ConversationResult:
    return engine.handle(text, session)


def _compose(variant: int, *groups: tuple[str, ...]) -> str:
    """Walk the Cartesian product of the phrase groups, deterministically."""

    parts: list[str] = []
    n = variant
    for group in groups:
        parts.append(group[n % len(group)])
        n //= len(group)
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


def _no_money_movement(result: ConversationResult, text: str) -> None:
    assert result.state.intent not in WRITE_INTENTS, text
    assert result.state.status is not ConversationStatus.CONFIRMING, text
    assert result.transfer is None, text
    assert result.bill is None, text
    assert result.state.slots.recipient is None, text


def _answered(result: ConversationResult, text: str) -> None:
    assert result.reply.strip(), text


_SAUDI = (
    ("هلا", "يا هلا", "هلا والله", "مرحبا", "مساء الخير", "صباح الخير", "أهلين"),
    (
        "وش أخبارك اليوم",
        "علومك اليوم",
        "كيف يومك ماشي",
        "عساك بخير",
        "كيف الأمور معك",
        "ودي نسولف شوي",
        "خلنا ناخذها سوالف",
        "كيفك يا صاحبي",
        "طمني عنك",
    ),
    ("", "يا بعدي", "بس كذا", "إذا تسمح", "شوي", "اليوم", "الحين"),
)
_EGYPTIAN = (
    ("هاي", "أهلا", "مساء الخير", "صباح الفل", "يا معلم", "بقولك", "معلش"),
    (
        "عامل ايه النهارده",
        "اخبارك ايه",
        "يومك ماشي ازاي",
        "عايز أرغي شوية",
        "خلينا نتكلم كلام عادي",
        "طمني عليك",
        "ينفع ندردش شوية",
        "محتاج أفصل شوية",
        "حاسس اليوم طويل",
    ),
    ("", "كده", "شوية", "لو سمحت", "بس", "النهارده", "دلوقتي"),
)
_MSA = (
    ("مرحباً", "أهلاً", "مساء الخير", "صباح الخير", "لو سمحت", "بالمناسبة", "حسناً"),
    (
        "كيف حالك اليوم",
        "أود أن نتحدث قليلاً",
        "هل يمكن أن نتبادل الحديث",
        "أخبرني كيف تسير الأمور",
        "أريد محادثة عادية فقط",
        "دعنا نتحدث دون أي معاملة",
        "لدي بعض الوقت للدردشة",
        "أريد أن أبدأ بسؤال بسيط",
    ),
    ("", "من فضلك", "قليلاً", "الآن", "إن أمكن", "اليوم"),
)
_ENGLISH = (
    ("hey", "hello", "hi there", "good morning", "good evening", "by the way"),
    (
        "how are things going today",
        "how has your day been",
        "can we just chat for a bit",
        "I feel like having a normal conversation",
        "let's talk about something ordinary",
        "I just want to chat, no transaction",
        "can I talk to you for a minute",
        "let's have a casual conversation first",
    ),
    ("", "please", "for a moment", "if that's okay", "today", "right now"),
)
_MIXED = (
    ("hello", "hey", "okay", "thanks", "please", "بالمناسبة", "هلا"),
    (
        "كيفك today",
        "خلنا chat شوي",
        "عايز normal conversation بس",
        "وش أخبارك today",
        "can we نسولف شوي",
        "ممكن talk معاك دقيقة",
        "today نفسي ندردش",
        "how are الأمور معك",
    ),
    ("", "لو سمحت", "please", "الحين", "دلوقتي", "بس", "for a bit"),
)
_NOISY = (
    (
        "هلااا كيفك",
        "وش اخباارك",
        "كيفكك اليوم",
        "عايز اتكلمم شويه",
        "خلينا نرغيى شوية",
        "اهلاا عامل اي",
        "hellooo how r u",
        "heyy can we chat",
        "how r thingss",
        "وش علومكك",
        "مساء الخيير كيف الحال",
    ),
    ("", "يا صاحبي", "لو سمحت", "بس", "today", "دلوقتي", "الحين"),
)
_VAGUE = (
    ("بقولك", "اسمع", "معلش", "لو سمحت", "عندي موضوع", "quick thing", "so"),
    (
        "مش عارف أبدأ منين",
        "محتاج رأيك في حاجة",
        "في موضوع محيرني",
        "ممكن تساعدني أفكر",
        "ودي أقولك شيء بس مو مرتب",
        "I am not sure how to phrase this",
        "I need your opinion on something",
        "can you hear me out",
    ),
    ("", "شوية", "بس", "لو تقدر", "for a moment", "please"),
)
_EVERYDAY = (
    ("بالمناسبة", "على فكرة", "اليوم", "بقولك", "you know", "honestly"),
    (
        "الدوام كان طويل وتعبت",
        "نفسي أشرب قهوة وأروق",
        "الزحمة اليوم كانت كثيرة",
        "أفكر أقرأ كتاب جديد",
        "كنت أتكلم مع صاحبي عن السفر",
        "work was exhausting today",
        "I could really use some coffee",
        "I feel like reading something new",
        "I want a quiet weekend",
    ),
    ("", "وش رايك", "what do you think", "بس كده", "today"),
)
_CHAT_FAMILIES = (_SAUDI, _EGYPTIAN, _MSA, _ENGLISH, _MIXED, _NOISY, _VAGUE, _EVERYDAY)

_FOLLOWUP_STARTS = (
    "خلنا نسولف شوي",
    "عايز أتكلم شوية",
    "can we chat for a minute",
    "ممكن نتكلم كلام عادي",
    "هلا والله وش أخبارك",
    "hello how are you doing",
)
_FOLLOWUPS = (
    "وإنت؟",
    "طيب كمل",
    "وبعدين؟",
    "وش بعد",
    "كمل كلامك",
    "what about you?",
    "go on",
    "tell me a bit more",
)
_ASIDES = (
    "بالمناسبة كيفك اليوم؟",
    "على فكرة وش أخبارك؟",
    "قبل ما نكمل عامل ايه؟",
    "quick side note, how are you?",
    "خلنا نسولف ثانية",
    "one second, how's your day?",
    "طيب قبلها طمني عليك",
)
_RESUMES = (
    "تمام نكمل",
    "يلا كمل",
    "خلاص نرجع للتحويل",
    "okay continue",
    "let's resume",
    "كمل العملية",
)


def test_the_gate_covers_every_family() -> None:
    assert FAMILY_COUNT == len(_CHAT_FAMILIES) + 2
    assert CASES_PER_FAMILY >= 1


@pytest.mark.parametrize("family", range(len(_CHAT_FAMILIES)))
def test_ordinary_talk_never_opens_a_money_flow(
    engine: ConversationEngine, family: int
) -> None:
    """Chit-chat, in five languages and with typos, stays chit-chat."""

    groups = _CHAT_FAMILIES[family]
    for variant in range(CASES_PER_FAMILY):
        text = _compose(variant, *groups)
        result = _turn(engine, f"stress-chat-{family}-{variant:05d}", text)
        _no_money_movement(result, text)
        _answered(result, text)


def test_a_conversational_follow_up_is_still_not_a_money_flow(
    engine: ConversationEngine,
) -> None:
    """ "وبعدين؟" answers the previous turn; it is not a new instruction."""

    for variant in range(CASES_PER_FAMILY):
        session = f"stress-followup-{variant:05d}"
        opener = _FOLLOWUP_STARTS[variant % len(_FOLLOWUP_STARTS)]
        follow = _FOLLOWUPS[(variant // len(_FOLLOWUP_STARTS)) % len(_FOLLOWUPS)]
        first = _turn(engine, session, opener)
        _no_money_movement(first, opener)
        second = _turn(engine, session, follow)
        _no_money_movement(second, follow)
        _answered(second, follow)


def test_an_aside_mid_transfer_keeps_the_slot_it_is_waiting_on(
    engine: ConversationEngine,
) -> None:
    """Small talk inside a transfer must not drop, or answer, the amount."""

    for variant in range(CASES_PER_FAMILY):
        session = f"stress-aside-{variant:05d}"
        opened = _turn(engine, session, "حول لمحمد")
        assert opened.state.intent is Intent.TRANSFER_MONEY
        assert opened.state.pending_slot == "amount"

        aside = _ASIDES[variant % len(_ASIDES)]
        interrupted = _turn(engine, session, aside)
        assert interrupted.state.intent is Intent.TRANSFER_MONEY, aside
        assert interrupted.state.pending_slot == "amount", aside
        assert interrupted.state.slots.amount is None, aside

        resume = _RESUMES[(variant // len(_ASIDES)) % len(_RESUMES)]
        resumed = _turn(engine, session, resume)
        assert resumed.state.intent is Intent.TRANSFER_MONEY, resume
        assert resumed.state.pending_slot == "amount", resume
        _answered(resumed, resume)
