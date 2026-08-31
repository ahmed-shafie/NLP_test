"""What role a turn plays in an in-progress flow.

A slot-filling flow reads the next message as the answer to the question it
asked. That is right for "250" and wrong for "transfer 250 riyals to Sara": a
fresh instruction absorbed as an answer becomes financial data (a bill's
reference number, a recipient's name) the customer never gave.

This decides the *role* of the turn — answer, new request, or aside — from
structural facts only (does it carry an action verb, is it question-shaped, does
it name its own object). The caller supplies the facts; nothing here inspects
wording, so widening the vocabulary of the deterministic extractors is enough to
widen this too.

The role never decides money. A new request only pauses the flow to ask the
customer which one they meant, and an aside only answers and re-asks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.schemas import Intent


class TurnRole(str, Enum):
    """The part this message plays while a flow waits for a slot."""

    # It answers the question the flow asked ("250", "Sara", "option 2").
    ANSWER = "answer"
    # It instructs a different flow ("transfer 250 to Sara" mid bill payment).
    NEW_REQUEST = "new_request"
    # It asks a question instead of answering one ("why was I charged a fee?").
    ASIDE = "aside"


@dataclass(frozen=True, slots=True)
class TurnRoleEvidence:
    """The chosen role, the flow a new request points at, and why."""

    role: TurnRole
    requested_flow: Intent | None
    signals: tuple[str, ...]


_MONEY_FLOWS = (Intent.TRANSFER_MONEY, Intent.PAY_BILL)


def classify_turn_role(
    *,
    active_intent: Intent | None,
    pending_slot: str | None,
    requested_flow: Intent | None,
    is_instruction: bool,
    is_question: bool,
    control_only: bool,
) -> TurnRoleEvidence:
    """Decide the role of one turn.

    ``requested_flow`` is the flow the deterministic chooser reads in this
    message, ``is_instruction`` that it carries an action verb and its own
    object (not a bare value), ``is_question`` that it is question-shaped, and
    ``control_only`` that it is made solely of control tokens ("yes", "2").

    Anything short of clear evidence stays :attr:`TurnRole.ANSWER`, because the
    flow's own reading of the turn is the behaviour that is already tested.
    """

    if active_intent not in _MONEY_FLOWS or pending_slot is None:
        return TurnRoleEvidence(TurnRole.ANSWER, None, ("not_mid_flow",))
    # A menu pick or a yes/no answers the prompt by construction.
    if control_only:
        return TurnRoleEvidence(TurnRole.ANSWER, None, ("control_only",))
    if is_question and not is_instruction:
        return TurnRoleEvidence(TurnRole.ASIDE, None, ("question_form",))
    if (
        is_instruction
        and requested_flow is not None
        and requested_flow is not active_intent
    ):
        return TurnRoleEvidence(
            TurnRole.NEW_REQUEST,
            requested_flow,
            ("instruction", f"flow:{requested_flow.value}"),
        )
    return TurnRoleEvidence(TurnRole.ANSWER, None, ("default_answer",))
