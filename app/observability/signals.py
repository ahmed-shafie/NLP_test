"""Operational signals: is the layer fast, and are the things it depends on up.

Separate from the turn store on purpose. A turn row is a *decision*, kept for
ninety days because somebody may need to count it later. What is here is a
*liveness* fact — "the Core answered a moment ago", "the model refused three
calls in a row" — which is only ever read as "now" and is therefore held in
process and lost on restart. Persisting it would invite reading a stale row as
the current state of a dependency.

Dependency health is recorded from real calls rather than a periodic probe: the
question that matters is whether the customer's request reached the Core, not
whether a synthetic ping did, and a probe adds traffic to the Core to answer a
question its own traffic already answers. The cost is that silence is not
health, so every reading carries when it was last observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

BANKING_CORE = "banking_core"
LLM = "llm"


@dataclass(frozen=True, slots=True)
class DependencyState:
    """What the last real call to a dependency did."""

    name: str
    healthy: bool
    # When a call last succeeded, and when one was last attempted at all.
    last_success: datetime | None
    last_attempt: datetime | None
    consecutive_failures: int
    attempts: int
    failures: int


@dataclass
class _Tally:
    healthy: bool = True
    last_success: datetime | None = None
    last_attempt: datetime | None = None
    consecutive_failures: int = 0
    attempts: int = 0
    failures: int = 0


_lock = Lock()
_tallies: dict[str, _Tally] = {}


def record_call(name: str, *, ok: bool) -> None:
    """Note the outcome of one real call to a dependency."""

    now = datetime.now(UTC)
    with _lock:
        tally = _tallies.setdefault(name, _Tally())
        tally.attempts += 1
        tally.last_attempt = now
        if ok:
            tally.healthy = True
            tally.last_success = now
            tally.consecutive_failures = 0
        else:
            tally.healthy = False
            tally.failures += 1
            tally.consecutive_failures += 1


def dependencies() -> list[DependencyState]:
    """Every dependency that has been called since this process started."""

    with _lock:
        return [
            DependencyState(
                name=name,
                healthy=tally.healthy,
                last_success=tally.last_success,
                last_attempt=tally.last_attempt,
                consecutive_failures=tally.consecutive_failures,
                attempts=tally.attempts,
                failures=tally.failures,
            )
            for name, tally in sorted(_tallies.items())
        ]


def unhealthy_count() -> int:
    """How many dependencies last answered with a failure."""

    return sum(1 for state in dependencies() if not state.healthy)


def failure_ratio(name: str) -> tuple[int, int]:
    """``(failures, attempts)`` for one dependency since this process started.

    For the model this is the number that matters: a rewrite the guard rejected
    is a normal outcome, but a call that never came back is the model being
    unavailable, and only the second one is counted as a failure here.
    """

    with _lock:
        tally = _tallies.get(name)
        return (tally.failures, tally.attempts) if tally else (0, 0)


def reset() -> None:
    """Forget every reading. For tests, and for a deliberate operator reset."""

    with _lock:
        _tallies.clear()
