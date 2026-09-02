"""Active Learning policy: decide which cases to log and which to auto-approve.

Runs passively at the tail of the pipeline. A case is *interesting* (worth
logging) when the deterministic layers were uncertain — the LLM was invoked, the
intent fell back, or the intent confidence was below
``NLU_ACTIVE_LEARNING_LOG_CONFIDENCE``. Confident, deterministic results are not
logged (there is nothing to learn). An interesting case with a concrete,
high-confidence intent is auto-approved; otherwise it waits for human review.
"""

from __future__ import annotations

import logging

from app.active_learning.priority import TurnSignals, score
from app.active_learning.schemas import CaseStatus, ReviewCase
from app.active_learning.store import get_store
from app.config import settings
from app.conversation.state import ConversationState
from app.observability.turns import previous_pending_slot
from app.request_context import get_request_id
from app.schemas import Intent, NLUResponse

logger = logging.getLogger(__name__)


def _classify(response: NLUResponse) -> CaseStatus | None:
    """Return the status to log this case under, or ``None`` to skip logging."""

    is_uncertain = (
        response.llm_assisted
        or response.intent is Intent.FALLBACK
        or response.confidence < settings.active_learning_log_confidence
    )
    if not is_uncertain:
        return None

    auto_ok = (
        response.intent is not Intent.FALLBACK
        and response.confidence >= settings.active_learning_auto_approve_confidence
    )
    return CaseStatus.AUTO_APPROVED if auto_ok else CaseStatus.PENDING


def record_case(response: NLUResponse, source: str) -> ReviewCase | None:
    """Log ``response`` to the review queue if it is worth learning from.

    Returns the logged :class:`ReviewCase`, or ``None`` when the case was skipped
    or active learning is disabled. Never raises: a logging failure must not break
    the request that produced it.
    """

    if not settings.active_learning_enabled:
        return None
    status = _classify(response)
    if status is None:
        return None
    # What is known at parse time: how risky this intent is, and how unsure the
    # layer was. The dialogue-level signals arrive at the end of the turn.
    priority = score(
        TurnSignals(
            intent=response.intent,
            confidence=response.confidence,
            llm_assisted=response.llm_assisted,
        )
    )
    try:
        return get_store().log_case(
            text=response.text,
            language=response.language,
            predicted_intent=response.intent,
            confidence=response.confidence,
            intent_source=response.intent_source,
            llm_assisted=response.llm_assisted,
            clarification=response.clarification,
            status=status,
            source=source,
            priority=priority,
            trace_id=get_request_id(),
        )
    except Exception:  # noqa: BLE001 - active learning must never break a request
        logger.warning("Failed to log active-learning case", exc_info=True)
        return None


def _repeated_prompt(state: ConversationState) -> bool | None:
    """Is this turn still waiting on the slot the previous turn already asked for?

    ``None`` when the turn store is off: the previous prompt is then unknown, and
    unknown is not the same as "no".
    """

    if not settings.turn_observability_enabled:
        return None
    pending = state.pending_slot
    if pending is None:
        return False
    return previous_pending_slot(state) == pending


def record_turn_outcome(state: ConversationState, reason_code: str | None) -> None:
    """Re-score this turn's case now that the dialogue outcome is known.

    Parsing sees one utterance; the queue order needs what happened next — the
    bank refused the action, the customer walked away mid-flow, the same slot had
    to be asked twice. Those only exist once the turn has run, so the case logged
    during parsing is scored again here and only ever moves up.

    Never raises: re-ordering a review queue is not worth failing a payment for.
    """

    if not settings.active_learning_enabled:
        return
    trace_id = get_request_id()
    if not trace_id:
        return
    try:
        priority = score(
            TurnSignals(
                intent=state.intent,
                # Confidence belongs to the parse; this pass only adds signals.
                confidence=1.0,
                reason_code=reason_code,
                status=state.status,
                repeated_prompt=_repeated_prompt(state),
            )
        )
        get_store().raise_priority(trace_id, priority)
    except Exception:  # noqa: BLE001 - active learning must never break a turn
        logger.warning("Failed to re-score active-learning case", exc_info=True)
