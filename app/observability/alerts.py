"""The SLO catalogue, and how a measurement becomes an alert.

Deliberately free of transport and of storage: an objective is a name, a window,
a ratio to watch and a bound, and :func:`evaluate` turns counts into breaches.
Whoever ships the breach — an endpoint, a cron, a pager — is somebody else's job,
which is what lets the catalogue be reviewed by the people who own the numbers
rather than by whoever owns the HTTP layer.

Every ratio is defined over ``ReasonCode`` values the engine attaches itself, so
an objective is measurable without re-deriving intent from the reply text.

Alongside the ratios there is a second, smaller catalogue of *operational*
objectives — latency, model availability, dependency health — which are plain
numbers with a ceiling rather than shares of turns. They are kept in a separate
type instead of being forced into :class:`Slo`, so the test that holds the ratio
catalogue to the ``ReasonCode``/``ConversationStatus`` enums keeps its meaning.
A reading that is missing is reported as unavailable and never silently skipped:
an objective nobody can read is a broken objective, not a passing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.conversation.reasons import ReasonCode
from app.conversation.state import ConversationStatus


class Severity(str, Enum):
    """How loudly a breach should be raised."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Slo:
    """One objective: a ratio over turns, and the bound it must stay under."""

    key: str
    title: str
    # Count keys from ``turns.counts`` summed for the numerator.
    numerator: tuple[str, ...]
    # Highest acceptable share of turns in the window.
    max_ratio: float
    window_minutes: int
    severity: Severity
    rationale: str


@dataclass(frozen=True, slots=True)
class Measurement:
    """What an objective actually reads, breach or not."""

    slo: Slo
    observed: int
    total: int
    ratio: float
    breached: bool


def _reason(code: ReasonCode) -> str:
    return f"reason:{code.value}"


def _status(status: ConversationStatus) -> str:
    return f"status:{status.value}"


CATALOGUE: tuple[Slo, ...] = (
    Slo(
        key="failed_writes",
        title="Turns that ended in a refused write",
        numerator=(_status(ConversationStatus.FAILED),),
        max_ratio=0.01,
        window_minutes=60,
        severity=Severity.CRITICAL,
        rationale=(
            "A refused write is the bank saying no after the customer said yes. "
            "It is never a completion, and a rising share of them is an incident "
            "in the Core or in what we send it, not a conversation problem."
        ),
    ),
    Slo(
        key="unresolved_identity",
        title="Turns stopped because who is being paid is unsettled",
        numerator=(
            _reason(ReasonCode.AMBIGUOUS_BENEFICIARY),
            _reason(ReasonCode.BENEFICIARY_NOT_FOUND),
            _reason(ReasonCode.AMBIGUOUS_BILLER),
            _reason(ReasonCode.BILLER_NOT_IN_CATALOGUE),
        ),
        max_ratio=0.15,
        window_minutes=1440,
        severity=Severity.WARNING,
        rationale=(
            "These are the turns where the directory or the catalogue could not "
            "name the payee. A jump usually means a directory outage or a new "
            "phrasing we do not read — never a reason to loosen resolution."
        ),
    ),
    Slo(
        key="understanding_gap",
        title="Turns we could not read as an answer or a request",
        numerator=(
            _reason(ReasonCode.INTENT_UNCLEAR),
            _reason(ReasonCode.CHOICE_NOT_RECOGNISED),
            _reason(ReasonCode.CONFIRMATION_NOT_RECOGNISED),
            _reason(ReasonCode.INVALID_SLOT_VALUE),
        ),
        max_ratio=0.2,
        window_minutes=1440,
        severity=Severity.WARNING,
        rationale=(
            "The share of turns where the customer said something and we asked "
            "again. This is the number the release loop is meant to move."
        ),
    ),
    Slo(
        key="core_unavailable",
        title="Turns where the Banking Core or the directory could not answer",
        numerator=(
            _reason(ReasonCode.BALANCE_UNAVAILABLE),
            _reason(ReasonCode.DIRECTORY_UNAVAILABLE),
            _reason(ReasonCode.PREFLIGHT_BLOCKED),
        ),
        max_ratio=0.02,
        window_minutes=60,
        severity=Severity.CRITICAL,
        rationale=(
            "A dependency being down looks like a polite refusal to the customer, "
            "so it has to be visible as a dependency breach here."
        ),
    ),
    Slo(
        key="abandoned_flows",
        title="Turns where the customer stopped the request",
        numerator=(
            _reason(ReasonCode.CANCELLED_BY_CUSTOMER),
            _reason(ReasonCode.SESSION_ENDED),
        ),
        max_ratio=0.1,
        window_minutes=1440,
        severity=Severity.WARNING,
        rationale=(
            "Cancelling is always the customer's right, so this never blocks "
            "anything. It is watched because a rising share of it is how a bad "
            "prompt or a wrong reading shows up before anybody complains."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class OpsSignal:
    """One operational objective: a measured number and the ceiling it must respect."""

    key: str
    title: str
    unit: str
    max_value: float
    window_minutes: int
    severity: Severity
    rationale: str


@dataclass(frozen=True, slots=True)
class SignalReading:
    """What an operational objective read, or that it could not be read."""

    signal: OpsSignal
    value: float | None
    available: bool
    breached: bool


P95_LATENCY = "turn_latency_p95_ms"
LLM_UNAVAILABLE_RATE = "llm_unavailable_rate_pct"
UNHEALTHY_DEPENDENCIES = "unhealthy_dependencies"


SIGNALS: tuple[OpsSignal, ...] = (
    OpsSignal(
        key=P95_LATENCY,
        title="Time to reply, ninety-fifth percentile",
        unit="ms",
        max_value=1500.0,
        window_minutes=60,
        severity=Severity.WARNING,
        rationale=(
            "Measured over turns, not over every HTTP request: a slow operations "
            "query is not a slow conversation, and mixing them lets one hide the "
            "other. The percentile is read from the turn store, so the number "
            "survives a restart and can be quoted for a past month."
        ),
    ),
    OpsSignal(
        key=LLM_UNAVAILABLE_RATE,
        title="Model calls that never came back",
        unit="%",
        max_value=15.0,
        window_minutes=0,
        severity=Severity.WARNING,
        rationale=(
            "Counts calls that failed or timed out — not rewrites the guard "
            "rejected, which are the guard working. Money replies are "
            "deterministic either way, so this is a quality-of-wording signal "
            "and never a correctness one."
        ),
    ),
    OpsSignal(
        key=UNHEALTHY_DEPENDENCIES,
        title="Dependencies whose last real call failed",
        unit="count",
        max_value=0.0,
        window_minutes=0,
        severity=Severity.CRITICAL,
        rationale=(
            "Read from the outcome of real calls rather than a probe, so it "
            "reflects what customers actually hit and adds no traffic to the "
            "Core. Silence is therefore not health, which is why each reading "
            "carries when the dependency was last seen."
        ),
    ),
)


def read_signals(
    readings: dict[str, float | None],
    catalogue: tuple[OpsSignal, ...] = SIGNALS,
) -> list[SignalReading]:
    """Compare each operational objective with its reading.

    A ``None`` reading means there was nothing to measure (no traffic yet, no
    call made). That is reported as unavailable rather than as zero, because a
    zero would read as a passing objective.
    """

    return [
        SignalReading(
            signal=signal,
            value=readings.get(signal.key),
            available=readings.get(signal.key) is not None,
            breached=(value := readings.get(signal.key)) is not None
            and value > signal.max_value,
        )
        for signal in catalogue
    ]


def evaluate(
    counts_for_window: dict[int, dict[str, int]],
    catalogue: tuple[Slo, ...] = CATALOGUE,
) -> list[Measurement]:
    """Read every objective against pre-fetched counts, keyed by window.

    Counts are passed in rather than queried so the catalogue stays testable
    against numbers on paper and reads the store once per distinct window.
    """

    measurements: list[Measurement] = []
    for slo in catalogue:
        window = counts_for_window.get(slo.window_minutes, {})
        total = window.get("turns", 0)
        observed = sum(window.get(key, 0) for key in slo.numerator)
        ratio = observed / total if total else 0.0
        measurements.append(
            Measurement(
                slo=slo,
                observed=observed,
                total=total,
                ratio=round(ratio, 4),
                # An empty window is not a breach: no traffic is not a failure.
                breached=bool(total) and ratio > slo.max_ratio,
            )
        )
    return measurements


def windows(catalogue: tuple[Slo, ...] = CATALOGUE) -> tuple[int, ...]:
    """The distinct windows the catalogue needs counts for."""

    return tuple(sorted({slo.window_minutes for slo in catalogue}))
