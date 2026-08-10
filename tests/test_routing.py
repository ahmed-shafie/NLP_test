"""Regression tests for first-turn routing (``route_fresh_turn``).

These lock in the two failure shapes the expanded gold set exposed: supported
requests that were falling through to the "transfer or bill?" prompt, and
unrelated questions that were opening a payment flow because the biller/name
extractors are deliberately generous.
"""

from __future__ import annotations

import pytest

from app.conversation import templates
from app.conversation.engine import route_fresh_turn
from app.nlu.lang import detect_language
from app.schemas import Intent, Language


def _route(text: str) -> Intent:
    lang = detect_language(text)
    return route_fresh_turn(text, lang, Intent.FALLBACK)


@pytest.mark.parametrize(
    "text",
    [
        "what is left in my account",
        "my balance?",
        "whats my ballance",
        "how much money is in my account",
        "كم رصيدى",
        "كم فلوسي",
        "كم باقي عندي في الحساب",
        "وش اللي موجود بحسابي",
    ],
)
def test_balance_questions_reach_the_balance_flow(text: str) -> None:
    assert _route(text) is Intent.BALANCE_INQUIRY


@pytest.mark.parametrize(
    "text",
    [
        "thanks for your help",
        "many thanks",
        "good night",
        "who am I talking to",
        "could you help me out",
        "شكرا على المساعدة",
        "سلام عليكم ورحمة الله",
        "يعطيك العافية",
        "انت مين",
        "وش الاشياء اللي تسويها",
    ],
)
def test_chit_chat_gets_a_chit_chat_reply(text: str) -> None:
    assert _route(text) is Intent.SMALL_TALK


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        # "are"/"you" are how-are-you cues, so these must not be answered with
        # "I'm good, thanks for asking!" — they ask what the assistant does.
        ("who are you", "I can send money"),
        ("what can you do", "I can send money"),
        ("can you help me", "I can send money"),
        ("من أنت", "أقدر أحوّل فلوس"),
        ("وش تقدر تسوي", "أقدر أحوّل فلوس"),
    ],
)
def test_capability_questions_list_what_the_assistant_does(
    text: str, phrase: str
) -> None:
    kind = templates._small_talk_kind(text)
    assert kind == "capability"
    assert phrase in templates.small_talk(text, detect_language(text))


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        # These share cue tokens with each other ("help" is also a capability
        # cue, "are"/"you" are also how-are-you cues), so each one pins the
        # cue-order that picks its reply.
        ("thanks for your help", "thanks"),
        ("شكرا على المساعدة", "thanks"),
        ("how are you doing", "how_are_you"),
        ("good night", "bye"),
        ("hi there", "greeting"),
    ],
)
def test_each_chit_chat_kind_keeps_its_own_reply(text: str, kind: str) -> None:
    assert templates._small_talk_kind(text) == kind


def test_thanks_and_how_are_you_replies_are_distinct() -> None:
    assert "Anytime" in templates.small_talk("thanks for your help", Language.EN)
    assert "thanks for asking" in templates.small_talk("how are you doing", Language.EN)


@pytest.mark.parametrize(
    "text",
    [
        # "mobile"/"internet" resolve to a telecom biller; without a pay verb
        # these are not bill payments.
        "change my mobile number",
        "I forgot my internet banking password",
        # The 20k-name gazetteer matches most Arabic words, so a bare "name"
        # must not be read as a transfer recipient.
        "قل لي نكتة",
        "وش رقم السويفت للبنك",
    ],
)
def test_unrelated_questions_never_open_a_payment_flow(text: str) -> None:
    assert _route(text) not in (Intent.TRANSFER_MONEY, Intent.PAY_BILL)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pay my STC bill", Intent.PAY_BILL),
        ("ادفع ل اس تي سي", Intent.PAY_BILL),
        ("I'd like to settle an invoice", Intent.PAY_BILL),
        ("حولي الي عبدالله 500", Intent.TRANSFER_MONEY),
        ("send 500 SAR to Ahmed", Intent.TRANSFER_MONEY),
        ("my transfer contacts", Intent.LIST_BENEFICIARIES),
        ("أضف مستفيداً جديداً", Intent.ADD_BENEFICIARY),
    ],
)
def test_supported_requests_still_route(text: str, expected: Intent) -> None:
    assert _route(text) is expected
