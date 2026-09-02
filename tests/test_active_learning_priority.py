"""The review queue is ordered by risk, not by clock."""

from __future__ import annotations

import pytest

from app.active_learning import priority, service
from app.active_learning.priority import INTENT_RISK, TurnSignals, score
from app.active_learning.schemas import CaseStatus
from app.active_learning.store import ActiveLearningStore, get_engine, get_sessionmaker
from app.config import settings
from app.conversation.reasons import ReasonCode
from app.conversation.state import ConversationState, ConversationStatus
from app.schemas import Intent, Language


@pytest.fixture()
def store(tmp_path, monkeypatch) -> ActiveLearningStore:
    monkeypatch.setattr(
        settings, "active_learning_store_url", f"sqlite:///{tmp_path}/al.db"
    )
    get_sessionmaker.cache_clear()
    get_engine.cache_clear()
    yield ActiveLearningStore()
    get_sessionmaker.cache_clear()
    get_engine.cache_clear()


def _log(store: ActiveLearningStore, intent: Intent, priority_value: float, **kw):
    return store.log_case(
        text="whatever",
        language=Language.EN,
        predicted_intent=intent,
        confidence=0.5,
        intent_source="semantic",
        llm_assisted=False,
        clarification=None,
        status=CaseStatus.PENDING,
        source="nlu.parse",
        priority=priority_value,
        **kw,
    )


# --------------------------------- scoring ---------------------------------- #


def test_every_intent_has_a_stated_risk():
    """A new intent must be priced deliberately, not silently defaulted."""

    assert set(INTENT_RISK) == set(Intent)


def test_a_money_flow_outranks_chit_chat_at_equal_uncertainty():
    transfer = score(TurnSignals(intent=Intent.TRANSFER_MONEY, confidence=0.5))
    small_talk = score(TurnSignals(intent=Intent.SMALL_TALK, confidence=0.5))

    assert transfer > small_talk


def test_being_less_sure_scores_higher_on_the_same_intent():
    unsure = score(TurnSignals(intent=Intent.PAY_BILL, confidence=0.2))
    confident = score(TurnSignals(intent=Intent.PAY_BILL, confidence=0.9))

    assert unsure > confident


def test_asking_the_llm_counts_as_uncertainty_whatever_confidence_says():
    """A reported confidence of 1.0 after an LLM call is not certainty."""

    assisted = score(
        TurnSignals(intent=Intent.TRANSFER_MONEY, confidence=1.0, llm_assisted=True)
    )
    plain = score(TurnSignals(intent=Intent.TRANSFER_MONEY, confidence=1.0))

    assert assisted > plain


def test_a_failed_transfer_dialogue_cannot_be_averaged_out_of_the_top():
    """This is the floor's whole purpose: a severe signal must not be diluted."""

    signals = TurnSignals(
        intent=Intent.TRANSFER_MONEY,
        confidence=1.0,  # everything else looks fine
        status=ConversationStatus.FAILED,
    )
    weighted_sum = sum(priority.explain(signals).values())

    assert score(signals) >= priority._FLOOR_DIALOGUE_FAILURE
    assert score(signals) > weighted_sum


def test_a_misread_reply_on_a_money_flow_lands_near_the_top():
    signals = TurnSignals(
        intent=Intent.PAY_BILL,
        confidence=1.0,
        reason_code=ReasonCode.CONFIRMATION_NOT_RECOGNISED.value,
    )

    assert score(signals) >= priority._FLOOR_MISUNDERSTOOD


def test_the_floors_do_not_apply_to_a_low_risk_intent():
    """A misread yes/no in chit-chat is not a payment incident."""

    signals = TurnSignals(
        intent=Intent.SMALL_TALK,
        confidence=1.0,
        reason_code=ReasonCode.CHOICE_NOT_RECOGNISED.value,
    )

    assert score(signals) < priority._FLOOR_MISUNDERSTOOD


def test_walking_away_mid_transfer_scores_above_a_completed_one():
    abandoned = score(
        TurnSignals(
            intent=Intent.TRANSFER_MONEY,
            confidence=1.0,
            status=ConversationStatus.CANCELLED,
        )
    )
    completed = score(
        TurnSignals(
            intent=Intent.TRANSFER_MONEY,
            confidence=1.0,
            status=ConversationStatus.COMPLETED,
        )
    )

    assert abandoned > completed


def test_the_same_slot_asked_twice_raises_the_score():
    repeated = score(
        TurnSignals(intent=Intent.TRANSFER_MONEY, confidence=0.9, repeated_prompt=True)
    )
    once = score(
        TurnSignals(intent=Intent.TRANSFER_MONEY, confidence=0.9, repeated_prompt=False)
    )

    assert repeated > once


def test_an_unknown_repeat_signal_is_not_read_as_no():
    """Turn store off means unknown; it must score like the signal is absent."""

    unknown = score(
        TurnSignals(intent=Intent.PAY_BILL, confidence=0.9, repeated_prompt=None)
    )
    absent = score(
        TurnSignals(intent=Intent.PAY_BILL, confidence=0.9, repeated_prompt=False)
    )

    assert unknown == absent


def test_a_missing_intent_outranks_chit_chat():
    """The case where we know least is not the case that matters least."""

    unknown = score(TurnSignals(intent=None, confidence=0.4))
    small_talk = score(TurnSignals(intent=Intent.SMALL_TALK, confidence=0.4))

    assert unknown > small_talk


def test_scores_stay_within_range():
    worst = score(
        TurnSignals(
            intent=Intent.TRANSFER_MONEY,
            confidence=0.0,
            llm_assisted=True,
            reason_code=ReasonCode.INTENT_UNCLEAR.value,
            status=ConversationStatus.FAILED,
            repeated_prompt=True,
        )
    )
    best = score(TurnSignals(intent=Intent.SMALL_TALK, confidence=1.0))

    assert 0.0 <= best < worst <= 1.0


def test_explain_accounts_for_every_weighted_signal():
    """An order a reviewer cannot interrogate is an order they will not trust."""

    contributions = priority.explain(
        TurnSignals(intent=Intent.TRANSFER_MONEY, confidence=0.3)
    )

    assert set(contributions) == set(priority.WEIGHTS)


# ---------------------------------- queue ----------------------------------- #


def test_the_queue_is_read_worst_first_not_newest_first(store):
    _log(store, Intent.TRANSFER_MONEY, 0.9)  # oldest, riskiest
    _log(store, Intent.SMALL_TALK, 0.1)
    _log(store, Intent.SMALL_TALK, 0.2)  # newest, routine

    queue = store.list_cases()

    assert [case.priority for case in queue] == [0.9, 0.2, 0.1]


def test_equal_priority_still_falls_back_to_newest_first(store):
    first = _log(store, Intent.SMALL_TALK, 0.3)
    second = _log(store, Intent.SMALL_TALK, 0.3)

    ids = [case.id for case in store.list_cases()]

    assert ids.index(second.id) < ids.index(first.id)


def test_the_turn_outcome_raises_the_case_it_belongs_to(store):
    case = _log(store, Intent.TRANSFER_MONEY, 0.4, trace_id="trace-a")
    other = _log(store, Intent.TRANSFER_MONEY, 0.4, trace_id="trace-b")

    moved = store.raise_priority("trace-a", 0.85)

    assert moved == 1
    assert store.get_case(case.id).priority == 0.85
    assert store.get_case(other.id).priority == 0.4


def test_the_turn_outcome_never_lowers_a_case(store):
    """The outcome pass adds signals; it must not bury what parsing already flagged."""

    case = _log(store, Intent.TRANSFER_MONEY, 0.9, trace_id="trace-a")

    moved = store.raise_priority("trace-a", 0.2)

    assert moved == 0
    assert store.get_case(case.id).priority == 0.9


def test_a_queue_written_before_priority_existed_is_still_readable(
    tmp_path, monkeypatch
):
    """The old table has no priority column; create_all would not add one."""

    from sqlalchemy import create_engine, text

    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE active_learning_cases ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at DATETIME, "
                "text TEXT, language VARCHAR(8), predicted_intent VARCHAR(32), "
                "confidence FLOAT, intent_source VARCHAR(16), llm_assisted BOOLEAN, "
                "clarification TEXT, status VARCHAR(16), corrected_intent VARCHAR(32), "
                "reviewer VARCHAR(128), reviewed_at DATETIME, source VARCHAR(32))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO active_learning_cases (created_at, text, language, "
                "predicted_intent, confidence, intent_source, llm_assisted, status, "
                "source) VALUES ('2026-01-01 00:00:00', 'old case', 'en', "
                "'transfer_money', 0.5, 'semantic', 0, 'pending', 'nlu.parse')"
            )
        )

    from app.active_learning import store as store_module

    store_module.get_sessionmaker.cache_clear()
    store_module.get_engine.cache_clear()
    monkeypatch.setattr(settings, "active_learning_store_url", url)
    try:
        cases = ActiveLearningStore().list_cases()
    finally:
        store_module.get_sessionmaker.cache_clear()
        store_module.get_engine.cache_clear()

    assert [case.text for case in cases] == ["old case"]
    assert cases[0].priority == 0.0


# -------------------------------- wiring ------------------------------------ #


def test_a_transfer_case_is_logged_above_a_small_talk_case(store, monkeypatch):
    """End to end through the service: the risky case is ahead in the queue."""

    from app.schemas import NLUResponse, TransferEntities

    monkeypatch.setattr("app.active_learning.service.get_store", lambda: store)
    for intent in (Intent.SMALL_TALK, Intent.TRANSFER_MONEY):
        service.record_case(
            NLUResponse(
                text="…",
                language=Language.EN,
                intent=intent,
                confidence=0.4,
                intent_source="semantic",
                entities=TransferEntities(),
            ),
            source="nlu.parse",
        )

    queue = store.list_cases()

    assert queue[0].predicted_intent is Intent.TRANSFER_MONEY


def test_re_scoring_survives_a_store_that_throws(store, monkeypatch):
    """A failure while re-ordering a queue must not surface on a payment turn."""

    def boom() -> ActiveLearningStore:
        raise RuntimeError("queue is down")

    monkeypatch.setattr("app.active_learning.service.get_store", boom)
    state = ConversationState(session_id="s1", intent=Intent.TRANSFER_MONEY)
    state.status = ConversationStatus.FAILED

    service.record_turn_outcome(state, ReasonCode.INSUFFICIENT_FUNDS.value)


def test_re_scoring_does_nothing_when_active_learning_is_off(store, monkeypatch):
    monkeypatch.setattr(settings, "active_learning_enabled", False)
    case = _log(store, Intent.TRANSFER_MONEY, 0.1, trace_id="trace-a")
    monkeypatch.setattr("app.active_learning.service.get_store", lambda: store)
    monkeypatch.setattr("app.active_learning.service.get_request_id", lambda: "trace-a")

    state = ConversationState(session_id="s1", intent=Intent.TRANSFER_MONEY)
    state.status = ConversationStatus.FAILED
    service.record_turn_outcome(state, None)

    assert store.get_case(case.id).priority == 0.1


def test_the_turn_outcome_promotes_the_parse_time_case(store, monkeypatch):
    case = _log(store, Intent.TRANSFER_MONEY, 0.3, trace_id="trace-a")
    monkeypatch.setattr("app.active_learning.service.get_store", lambda: store)
    monkeypatch.setattr("app.active_learning.service.get_request_id", lambda: "trace-a")
    monkeypatch.setattr(
        "app.active_learning.service.previous_pending_slot", lambda state: None
    )

    state = ConversationState(session_id="s1", intent=Intent.TRANSFER_MONEY)
    state.status = ConversationStatus.FAILED
    service.record_turn_outcome(state, ReasonCode.INSUFFICIENT_FUNDS.value)

    assert store.get_case(case.id).priority >= priority._FLOOR_DIALOGUE_FAILURE


def test_a_re_asked_slot_is_detected_from_the_previous_turn(store, monkeypatch):
    case = _log(store, Intent.TRANSFER_MONEY, 0.1, trace_id="trace-a")
    monkeypatch.setattr("app.active_learning.service.get_store", lambda: store)
    monkeypatch.setattr("app.active_learning.service.get_request_id", lambda: "trace-a")
    monkeypatch.setattr(
        "app.active_learning.service.previous_pending_slot", lambda state: "amount"
    )

    state = ConversationState(session_id="s1", intent=Intent.TRANSFER_MONEY)
    state.pending_slot = "amount"
    service.record_turn_outcome(state, ReasonCode.SLOT_REQUIRED.value)

    raised = store.get_case(case.id).priority
    assert raised > 0.1


def test_a_different_slot_this_turn_is_progress_not_a_repeat(store, monkeypatch):
    """Moving from "who to?" to "how much?" is the flow working, not failing."""

    monkeypatch.setattr("app.active_learning.service.get_store", lambda: store)
    state = ConversationState(session_id="s1", intent=Intent.TRANSFER_MONEY)
    state.pending_slot = "amount"

    scores: dict[str, float] = {}
    for trace, previous in (("progress", "recipient"), ("repeat", "amount")):
        case = _log(store, Intent.TRANSFER_MONEY, 0.1, trace_id=trace)
        monkeypatch.setattr(
            "app.active_learning.service.get_request_id", lambda t=trace: t
        )
        monkeypatch.setattr(
            "app.active_learning.service.previous_pending_slot",
            lambda state, p=previous: p,
        )
        service.record_turn_outcome(state, ReasonCode.SLOT_REQUIRED.value)
        scores[trace] = store.get_case(case.id).priority

    assert scores["repeat"] > scores["progress"]


def test_the_repeat_signal_is_unknown_when_the_turn_store_is_off(monkeypatch):
    monkeypatch.setattr(settings, "turn_observability_enabled", False)
    state = ConversationState(session_id="s1")
    state.pending_slot = "amount"

    assert service._repeated_prompt(state) is None


def test_scoring_never_takes_the_customer_s_words():
    """No cue list can creep in if the score cannot see the text in the first place."""

    fields = set(TurnSignals.__dataclass_fields__)

    assert fields == {
        "intent",
        "confidence",
        "llm_assisted",
        "reason_code",
        "status",
        "repeated_prompt",
    }
