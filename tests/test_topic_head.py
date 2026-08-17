"""The trained head that answers customer-service questions retrieval refused.

The load-bearing tests are the confinement ones: the head only speaks where the
retrieval vote already refused, it needs the retrieved majority to agree on the
same answer, and no probability it can produce may turn an executable request
into a topic answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.conversation.engine import ConversationEngine
from app.conversation.topic_replies import (
    FAMILY_REPLIES,
    TOPIC_FAMILIES,
    answer_key,
    decide,
    reply_for_key,
)
from app.embeddings import get_embedder
from app.nlu import topic_head
from app.nlu.topic_head import NO_ANSWER, Prediction, TopicHead, get_topic_head
from app.schemas import Intent, Language

QUESTION = "علاش تحسبو رسوم على السحب؟"
# A topic answered by its family (it has no specific reply of its own).
CASH_FEE = "رسوم السحب النقدي"
CHARGED_TWICE = "تم التحصيل مرتين"
SURE = Prediction(key=answer_key(CASH_FEE), probability=0.9999)


def votes_for(topic: str, count: int = 6) -> dict[str, int]:
    """A majority too split for the retrieval gate to answer on its own."""

    return {topic: count, CHARGED_TWICE: 4}


# ------------------------------------------------------------------ answer keys


def test_a_topic_with_its_own_reply_is_its_own_answer() -> None:
    assert answer_key(CHARGED_TWICE) == CHARGED_TWICE


def test_a_topic_without_one_is_answered_by_its_family() -> None:
    assert answer_key(CASH_FEE) == TOPIC_FAMILIES[CASH_FEE]


def test_a_label_with_no_reviewed_answer_maps_to_no_answer() -> None:
    for label in ("confirm", "deny", "out_of_scope", ""):
        assert answer_key(label) == NO_ANSWER
    assert reply_for_key(NO_ANSWER, "anything", Language.AR) is None


def test_an_answer_key_resolves_to_the_reviewed_reply() -> None:
    key = TOPIC_FAMILIES[CASH_FEE]
    assert reply_for_key(key, QUESTION, Language.AR) == (
        key,
        FAMILY_REPLIES[key][Language.AR],
    )


def test_the_question_words_still_correct_the_answer() -> None:
    """The cue rules apply to the head's answer exactly as to a retrieved one.

    And the *corrected* subject comes back, so a trace of the turn names the
    subject the customer actually read.
    """

    assert reply_for_key("card_ordering", "how do I freeze my card?", Language.EN) == (
        "security",
        FAMILY_REPLIES["security"][Language.EN],
    )


# ----------------------------------------------------------- the forward pass


def head_of(keys: tuple[str, ...], winner: int) -> TopicHead:
    """A two-layer head that always names ``keys[winner]``."""

    w1 = np.eye(4, dtype="float32")
    b1 = np.zeros(4, dtype="float32")
    w2 = np.zeros((4, len(keys)), dtype="float32")
    w2[0, winner] = 40.0
    b2 = np.zeros(len(keys), dtype="float32")
    return TopicHead((w1, b1, w2, b2), keys)


def test_the_forward_pass_returns_a_probability() -> None:
    head = head_of(("a", "b", "c"), winner=1)
    prediction = head.predict(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    assert prediction.key == "b"
    assert 0.99 < prediction.probability <= 1.0
    assert head.dimension == 4


def test_the_shipped_head_answers_only_reviewed_keys() -> None:
    head = get_topic_head()
    assert head is not None, "the trained head ships with the app"
    for key in head.answers:
        if key == NO_ANSWER:
            continue
        assert reply_for_key(key, "", Language.AR) is not None, key


def test_a_head_trained_for_another_embedder_is_refused(tmp_path: Path) -> None:
    """Vectors from a different model mean nothing to these weights."""

    path = tmp_path / "topic_head.npz"
    np.savez_compressed(
        path,
        w1=np.zeros((4, 2), dtype="float32"),
        b1=np.zeros(2, dtype="float32"),
        w2=np.zeros((2, 2), dtype="float32"),
        b2=np.zeros(2, dtype="float32"),
        keys=np.array(["a", "b"]),
        embedding_model=np.array("some/other-model"),
    )
    get_topic_head.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(topic_head, "WEIGHTS_PATH", path)
        assert get_topic_head() is None
    get_topic_head.cache_clear()


def test_a_missing_head_leaves_the_retrieval_gate_alone(tmp_path: Path) -> None:
    get_topic_head.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(topic_head, "WEIGHTS_PATH", tmp_path / "absent.npz")
        assert get_topic_head() is None
    get_topic_head.cache_clear()


def test_the_head_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    get_topic_head.cache_clear()
    monkeypatch.setattr(settings, "topic_head_enabled", False)
    assert get_topic_head() is None
    get_topic_head.cache_clear()


# -------------------------------------------------------------------- the gate


def test_a_certain_head_answers_where_the_vote_was_too_split() -> None:
    answer = decide(QUESTION, 0.85, votes_for(CASH_FEE), 10, Language.AR, SURE)

    assert answer is not None
    assert answer.reply == FAMILY_REPLIES[TOPIC_FAMILIES[CASH_FEE]][Language.AR]


def test_retrieval_keeps_the_first_word() -> None:
    """A question the shipped gate answers keeps its measured answer."""

    disagreeing = Prediction(key=TOPIC_FAMILIES["سعر الصرف"], probability=0.9999)
    answer = decide(QUESTION, 0.99, {CHARGED_TWICE: 10}, 10, Language.AR, disagreeing)

    assert answer is not None
    assert answer.subject == CHARGED_TWICE


def test_the_head_must_agree_with_the_retrieved_majority() -> None:
    """The head alone is measurably three times as wrong; agreement is the guard."""

    votes = votes_for("سعر الصرف")
    assert decide(QUESTION, 0.85, votes, 10, Language.AR, SURE) is None


def test_an_unsure_head_stays_generic() -> None:
    unsure = Prediction(key=SURE.key, probability=0.98)
    votes = votes_for(CASH_FEE)
    assert decide(QUESTION, 0.85, votes, 10, Language.AR, unsure) is None


def test_a_question_far_from_everything_indexed_stays_generic() -> None:
    """Below the floor the question is out of scope, however sure the head is."""

    assert decide(QUESTION, 0.60, votes_for(CASH_FEE), 10, Language.AR, SURE) is None


def test_the_head_cannot_answer_when_it_predicts_no_answer() -> None:
    executable = Prediction(key=NO_ANSWER, probability=1.0)
    votes = votes_for(CASH_FEE)
    assert decide(QUESTION, 0.95, votes, 10, Language.AR, executable) is None


def test_no_head_at_all_reproduces_the_retrieval_gate() -> None:
    assert decide(QUESTION, 0.85, votes_for(CASH_FEE), 10, Language.AR, None) is None


def test_the_thresholds_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "topic_head_threshold", 0.5)
    unsure = Prediction(key=SURE.key, probability=0.6)
    assert (
        decide(QUESTION, 0.85, votes_for(CASH_FEE), 10, Language.AR, unsure) is not None
    )


# ------------------------------------------------------------ through the engine


@pytest.mark.skipif(get_embedder() is None, reason="embedding model unavailable")
def test_the_head_never_turns_a_transfer_into_an_answer() -> None:
    """Executable rows train the "no answer" class for exactly this."""

    for text in ("send 100 sar to ahmed", "حول ٥٠٠ ريال لأحمد", "كم رصيدي"):
        result = ConversationEngine().handle(
            text=text, session_id=f"head-{text}", user_id="demo"
        )
        assert result.state.intent in {
            Intent.TRANSFER_MONEY,
            Intent.BALANCE_INQUIRY,
        }, text


@pytest.mark.skipif(get_embedder() is None, reason="embedding model unavailable")
def test_a_bill_request_is_still_a_bill_request() -> None:
    result = ConversationEngine().handle(
        text="pay my mobily bill 100 sar", session_id="head-bill", user_id="demo"
    )
    assert result.state.intent is Intent.PAY_BILL
