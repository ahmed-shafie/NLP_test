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
)
from app.embeddings import get_embedder
from app.schemas import Language

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
    answer = decide(0.80, {CHARGED_TWICE: 10}, 10, Language.AR)
    assert answer is not None
    assert answer.subject == CHARGED_TWICE


def test_unanimous_but_distant_neighbours_stay_generic() -> None:
    assert decide(0.70, {CHARGED_TWICE: 10}, 10, Language.AR) is None


def test_a_split_vote_needs_the_higher_bar() -> None:
    """Eight of ten is enough at 0.95 similarity, not at 0.80."""

    votes = {CHARGED_TWICE: 8, TIMING: 1, NOT_RECEIVED: 1}
    assert decide(0.95, votes, 10, Language.AR) is not None
    assert decide(0.80, votes, 10, Language.AR) is None


def test_a_weak_majority_stays_generic() -> None:
    """Six of ten name the topic — agreement the gate does not accept."""

    votes = {CHARGED_TWICE: 6, TIMING: 2, NOT_RECEIVED: 2}
    assert decide(0.99, votes, 10, Language.AR) is None


def test_disagreeing_topics_stay_generic_even_within_one_family() -> None:
    """Measured: answering these at family level quadruples wrong answers."""

    votes = {TIMING: 4, NOT_RECEIVED: 4, "انتظار التحويل": 2}
    assert decide(0.95, votes, 10, Language.AR) is None


def test_no_topic_at_all_stays_generic() -> None:
    """Nearest rows are executable requests, so there is no subject to answer."""

    assert decide(0.99, {}, 10, Language.EN) is None


def test_gate_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "topic_reply_unanimous_threshold", 0.99)
    assert decide(0.90, {CHARGED_TWICE: 10}, 10, Language.AR) is None


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
