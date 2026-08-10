"""Regression tests for first-turn routing (``route_fresh_turn``).

These lock in the two failure shapes the expanded gold set exposed: supported
requests that were falling through to the "transfer or bill?" prompt, and
unrelated questions that were opening a payment flow because the biller/name
extractors are deliberately generous.
"""

from __future__ import annotations

import pytest

from app.conversation.engine import route_fresh_turn
from app.nlu.lang import detect_language
from app.schemas import Intent


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
