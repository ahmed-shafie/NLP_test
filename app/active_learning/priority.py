"""How urgent is this case to review?

The queue used to be chronological, which is the wrong order for a bank: a
misread chit-chat line sat at the top while a transfer the layer did not
understand waited behind it. A case is scored instead, and the queue is read
worst-first.

Every input is a signal the layer already produces deterministically — the
intent, its confidence, whether the LLM was involved, the ``ReasonCode`` the
engine attached to the turn, its terminal status. Nothing here reads the
customer's words, and no cue list ("I meant", "no I said") decides anything: a
misunderstanding is what ``ReasonCode`` says it is.

The weighted sum alone is not enough. Averaging pushes a single severe signal
towards the middle, so a transfer that the engine could not understand at all
would score about the same as a low-confidence FAQ. The floors below stop that:
when a high-risk flow misfires, the case cannot land outside the top of the
queue no matter how the rest of the signals read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.conversation.reasons import ReasonCode
from app.conversation.state import ConversationStatus
from app.schemas import Intent

# What it costs to get this intent wrong. A misclassified FAQ is an unhelpful
# answer; a misclassified transfer moves money, or refuses to. These are the
# ratios the review order is argued from, so they are stated once, here.
INTENT_RISK: Mapping[Intent, float] = {
    Intent.TRANSFER_MONEY: 1.0,
    Intent.PAY_BILL: 1.0,
    Intent.ADD_BENEFICIARY: 0.85,
    Intent.BALANCE_INQUIRY: 0.35,
    Intent.LIST_BENEFICIARIES: 0.3,
    Intent.INAPPROPRIATE: 0.15,
    Intent.SMALL_TALK: 0.1,
    Intent.FALLBACK: 0.4,
}

# An unrecognised (or missing) intent is not harmless — it is the case where we
# know least — so it scores above chit-chat rather than below it.
UNKNOWN_INTENT_RISK = 0.4

WEIGHTS: Mapping[str, float] = {
    "business_risk": 0.30,
    "uncertainty": 0.22,
    "misunderstood_reply": 0.18,
    "dialogue_failure": 0.12,
    "repeated_prompt": 0.10,
    "abandoned_mid_flow": 0.08,
}

# The engine said, in its own codes, that it could not read what the customer
# replied. This is the honest equivalent of their "customer correction" signal.
_MISUNDERSTOOD: frozenset[str] = frozenset(
    {
        ReasonCode.INTENT_UNCLEAR.value,
        ReasonCode.CHOICE_NOT_RECOGNISED.value,
        ReasonCode.CONFIRMATION_NOT_RECOGNISED.value,
        ReasonCode.INVALID_SLOT_VALUE.value,
    }
)

# Above this, an intent counts as a money flow for the floors below.
HIGH_RISK = 0.8

# Floors, worst first. Each says: when this happened on a money flow, the case
# belongs at the top of the queue whatever the average works out to.
_FLOOR_DIALOGUE_FAILURE = 0.85
_FLOOR_MISUNDERSTOOD = 0.80
_FLOOR_ABANDONED = 0.75


@dataclass(frozen=True, slots=True)
class TurnSignals:
    """Deterministic signals about one turn, as the layer already recorded them.

    ``repeated_prompt`` is three-valued on purpose: ``None`` means the turn store
    is switched off, so we do not know whether the same slot was asked twice. An
    unknown signal contributes nothing and is not silently read as "no".
    """

    intent: Intent | None = None
    confidence: float = 0.0
    llm_assisted: bool = False
    reason_code: str | None = None
    status: ConversationStatus | None = None
    repeated_prompt: bool | None = None

    @property
    def business_risk(self) -> float:
        if self.intent is None:
            return UNKNOWN_INTENT_RISK
        return INTENT_RISK.get(self.intent, UNKNOWN_INTENT_RISK)

    @property
    def high_risk(self) -> bool:
        return self.business_risk >= HIGH_RISK


def _components(signals: TurnSignals) -> dict[str, float]:
    """Each signal as a 0..1 magnitude, before weighting."""

    uncertainty = 1.0 - max(0.0, min(1.0, signals.confidence))
    # The LLM being asked at all means the deterministic layers were not sure,
    # whatever confidence ended up being reported.
    if signals.llm_assisted:
        uncertainty = max(uncertainty, 0.5)
    return {
        "business_risk": signals.business_risk,
        "uncertainty": uncertainty,
        "misunderstood_reply": (1.0 if signals.reason_code in _MISUNDERSTOOD else 0.0),
        "dialogue_failure": (
            1.0 if signals.status is ConversationStatus.FAILED else 0.0
        ),
        "repeated_prompt": 1.0 if signals.repeated_prompt else 0.0,
        "abandoned_mid_flow": (
            1.0 if signals.status is ConversationStatus.CANCELLED else 0.0
        ),
    }


def score(signals: TurnSignals) -> float:
    """Return this case's review priority, 0.0 (routine) to 1.0 (review first)."""

    components = _components(signals)
    total = sum(WEIGHTS[name] * value for name, value in components.items())

    if signals.high_risk:
        if components["dialogue_failure"]:
            total = max(total, _FLOOR_DIALOGUE_FAILURE)
        if components["misunderstood_reply"]:
            total = max(total, _FLOOR_MISUNDERSTOOD)
        if components["abandoned_mid_flow"]:
            total = max(total, _FLOOR_ABANDONED)

    return round(min(1.0, max(0.0, total)), 4)


def explain(signals: TurnSignals) -> dict[str, float]:
    """The weighted contribution of each signal, so an order can be argued with."""

    return {
        name: round(WEIGHTS[name] * value, 4)
        for name, value in _components(signals).items()
    }
