"""Contextual answers for refused customer-service questions.

The load-bearing tests are the gate ones: a topical answer is only worth sending
when the retrieval is decisive, and it must never be able to open, fill or
confirm a money flow.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.conversation import templates, topic_replies
from app.conversation.engine import ConversationEngine
from app.conversation.topic_replies import (
    FAMILY_REPLIES,
    TOPIC_FAMILIES,
    TOPIC_REPLIES,
    decide,
    topic_reply_top_k,
)
from app.embeddings import get_embedder
from app.schemas import Intent, Language

QUESTION = "ليش انخصم مني مرتين؟"
CHARGED_TWICE = "تم التحصيل مرتين"
TIMING = "توقيت التحويل"
NOT_RECEIVED = "لم يستلم المستلم التحويل"


# ------------------------------------------------------------------- the table


def test_every_family_has_a_reply_in_both_languages() -> None:
    for family in set(TOPIC_FAMILIES.values()):
        assert set(FAMILY_REPLIES[family]) == {Language.AR, Language.EN}


def test_every_detailed_topic_belongs_to_a_family() -> None:
    """A specific reply must still degrade to a family answer if it is removed."""

    for topic in TOPIC_REPLIES:
        assert topic in TOPIC_FAMILIES, topic


def test_answers_promise_no_policy() -> None:
    """No answer may state a fee, a duration or a limit we cannot verify."""

    forbidden = ("يوم عمل", "business day", "%", "ريال", "SAR", "ساعة", "hour")
    replies = [
        text
        for table in (FAMILY_REPLIES, TOPIC_REPLIES)
        for by_language in table.values()
        for text in by_language.values()
    ]
    assert replies
    for text in replies:
        for word in forbidden:
            assert word not in text, f"{word!r} in {text!r}"


def test_specific_reply_wins_over_its_family() -> None:
    family = FAMILY_REPLIES[TOPIC_FAMILIES[CHARGED_TWICE]][Language.AR]
    assert topic_replies.topic_reply(CHARGED_TWICE, Language.AR) != family


def test_topic_without_a_specific_reply_falls_back_to_its_family() -> None:
    topic = next(t for t in TOPIC_FAMILIES if t not in TOPIC_REPLIES)
    assert (
        topic_replies.topic_reply(topic, Language.EN)
        == FAMILY_REPLIES[TOPIC_FAMILIES[topic]][Language.EN]
    )


def test_meta_labels_have_no_answer() -> None:
    """ "confirm"/"deny" are not questions about a subject."""

    for label in ("confirm", "deny", "out_of_scope", "repeat_request"):
        assert topic_replies.topic_reply(label, Language.AR) is None


# -------------------------------------------------------------------- the gate


def test_unanimous_neighbours_answer_the_topic() -> None:
    answer = decide(QUESTION, 0.80, {CHARGED_TWICE: 10}, 10, Language.AR)
    assert answer is not None
    assert answer.subject == CHARGED_TWICE


def test_unanimous_but_distant_neighbours_stay_generic() -> None:
    assert decide(QUESTION, 0.70, {CHARGED_TWICE: 10}, 10, Language.AR) is None


def test_a_split_vote_needs_the_higher_bar() -> None:
    """Eight of ten is enough at 0.95 similarity, not at 0.80."""

    votes = {CHARGED_TWICE: 8, TIMING: 1, NOT_RECEIVED: 1}
    assert decide(QUESTION, 0.95, votes, 10, Language.AR) is not None
    assert decide(QUESTION, 0.80, votes, 10, Language.AR) is None


def test_a_weak_majority_stays_generic() -> None:
    """Six of ten name the topic — agreement the gate does not accept."""

    votes = {CHARGED_TWICE: 6, TIMING: 2, NOT_RECEIVED: 2}
    assert decide(QUESTION, 0.99, votes, 10, Language.AR) is None


def test_disagreeing_topics_stay_generic_even_within_one_family() -> None:
    """Measured: answering these at family level quadruples wrong answers."""

    votes = {TIMING: 4, NOT_RECEIVED: 4, "انتظار التحويل": 2}
    assert decide(QUESTION, 0.95, votes, 10, Language.AR) is None


def test_no_topic_at_all_stays_generic() -> None:
    """Nearest rows are executable requests, so there is no subject to answer."""

    assert decide(QUESTION, 0.99, {}, 10, Language.EN) is None


def test_gate_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "topic_reply_unanimous_threshold", 0.99)
    assert decide(QUESTION, 0.90, {CHARGED_TWICE: 10}, 10, Language.AR) is None


# --------------------------------------------------- subjects the gate confuses


def test_the_rate_and_its_fee_get_the_same_exchange_answer() -> None:
    """Retrieval cannot tell them apart, so neither answer may exclude the other."""

    assert TOPIC_FAMILIES["سعر الصرف"] == TOPIC_FAMILIES["رسوم الصرف"] == "fx"
    reply = topic_replies.topic_reply("رسوم الصرف", Language.EN)
    assert reply is not None
    assert "rate" in reply and "fee" in reply


def test_a_blocked_card_and_a_blocked_pin_get_the_same_answer() -> None:
    assert TOPIC_FAMILIES["البطاقة لا تعمل"] == "card_blocked"
    assert TOPIC_FAMILIES["رمز التعريف الشخصي محظور"] == "card_blocked"


@pytest.mark.parametrize(
    "question",
    [
        "my card is not working",
        "أظن إني ماني قادر استخدم بطاقتي",
        "علاش ما خدماش البطاقة الافتراضية ديالي؟",
    ],
)
def test_a_card_that_does_not_work_is_not_answered_as_a_stolen_card(
    question: str,
) -> None:
    """ "Call support now to block it" is only right for a theft."""

    answer = decide(question, 0.95, {"بطاقة مخترقة": 10}, 10, Language.AR)

    assert answer is not None
    assert answer.reply == FAMILY_REPLIES["card_blocked"][Language.AR]


@pytest.mark.parametrize(
    "question",
    ["كيف أقدر أجمد بطاقتي من التطبيق؟", "how do I freeze my card in the app?"],
)
def test_freezing_a_card_is_answered_as_urgent(question: str) -> None:
    """Retrieval reads this as a question about the app; the answer is urgent."""

    language = Language.EN if question.isascii() else Language.AR
    answer = decide(question, 0.95, {"ربط البطاقة": 10}, 10, language)

    assert answer is not None
    assert answer.reply == FAMILY_REPLIES["security"][language]


def test_unblocking_a_card_is_not_a_theft_report() -> None:
    answer = decide(
        "كيف أقدر ألغي حظر بطاقتي بالتطبيق؟",
        0.95,
        {"البطاقة لا تعمل": 10},
        10,
        Language.AR,
    )

    assert answer is not None
    assert answer.reply == FAMILY_REPLIES["card_blocked"][Language.AR]


def test_the_cues_cannot_answer_a_question_the_gate_refused() -> None:
    """A cue corrects the subject; it never lowers the bar for answering."""

    assert (
        decide("my card is not working", 0.50, {"بطاقة مخترقة": 10}, 10, Language.EN)
        is None
    )


# ------------------------------------------------------------ through the engine


@pytest.mark.skipif(get_embedder() is None, reason="embedding model unavailable")
def test_a_refused_question_is_answered_in_its_own_context() -> None:
    result = ConversationEngine().handle(
        text="ليش انخصم مني مرتين؟", session_id="topic-1", user_id="demo"
    )
    assert result.state.intent is None  # no flow was opened
    assert result.reply == topic_replies.topic_reply(CHARGED_TWICE, Language.AR)
    assert "(١) تحويل" not in result.reply


@pytest.mark.skipif(get_embedder() is None, reason="embedding model unavailable")
def test_a_topical_answer_never_starts_a_money_flow() -> None:
    """The answer ends the turn: no slot is filled and nothing is pending."""

    result = ConversationEngine().handle(
        text="my card payment was reversed, why?", session_id="topic-2", user_id="demo"
    )
    assert result.transfer is None
    assert result.bill is None
    assert result.state.slots.amount is None
    assert result.state.slots.recipient is None


@pytest.mark.skipif(get_embedder() is None, reason="embedding model unavailable")
def test_switching_the_feature_off_restores_the_generic_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "topic_replies_enabled", False)
    result = ConversationEngine().handle(
        text="ليش انخصم مني مرتين؟", session_id="topic-3", user_id="demo"
    )
    assert result.reply == templates.choose_action(Language.AR)


# ------------------------------------------------- the gate an English question meets


def test_an_english_question_reads_a_narrower_window() -> None:
    """English retrieval is cross-lingual; its window is calibrated separately."""

    assert topic_reply_top_k(Language.EN) == settings.topic_reply_top_k_en
    assert topic_reply_top_k(Language.AR) == settings.topic_reply_top_k


def test_a_unanimous_english_retrieval_clears_its_own_bar() -> None:
    """0.79 is under the Arabic bar for the same evidence, and over the English one.

    An English question is answered by Arabic rows, which score lower than the
    same-language retrieval the Arabic bar was set from.
    """

    votes = {CHARGED_TWICE: 7}

    answered = decide("why was i charged twice", 0.79, votes, 7, Language.EN)
    assert answered is not None
    assert answered.reply == topic_replies.topic_reply(CHARGED_TWICE, Language.EN)

    assert decide("why was i charged twice", 0.75, votes, 7, Language.EN) is None


def test_a_split_english_vote_still_meets_the_full_bar() -> None:
    """Only unanimity moves; a majority vote is as demanding as it was."""

    votes = {CHARGED_TWICE: 6, TIMING: 1}

    assert decide("why was i charged twice", 0.90, votes, 7, Language.EN) is None
    assert decide("why was i charged twice", 0.95, votes, 7, Language.EN) is not None


@pytest.mark.skipif(get_embedder() is None, reason="embedding model unavailable")
def test_an_english_card_question_is_answered_not_menued() -> None:
    """The reported gap: this retrieves "البطاقة لا تعمل" and used to get the menu."""

    result = ConversationEngine().handle(
        text="my card is not working", session_id="topic-en-1", user_id="demo"
    )
    assert result.state.intent is None
    assert result.reply == FAMILY_REPLIES["card_blocked"][Language.EN]
    assert "(1) send money" not in result.reply


@pytest.mark.skipif(get_embedder() is None, reason="embedding model unavailable")
def test_the_english_gate_does_not_answer_a_transfer_request() -> None:
    """A wider English gate must not turn an executable request into an answer."""

    result = ConversationEngine().handle(
        text="send 100 sar to ahmed", session_id="topic-en-2", user_id="demo"
    )
    assert result.state.intent is Intent.TRANSFER_MONEY
