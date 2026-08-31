"""Operational signals: reply latency, model availability, dependencies, tracing.

These are the four signals worth taking from the imported design. What is *not*
taken is its store: no customer text, no unauthenticated read, no in-process
deque standing in for history. The latency percentile here is read from the
durable turn store, and the read stays behind the operations key.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import banking_core_client
from app.config import settings
from app.llm import LLMExceptionHandler
from app.main import app
from app.observability import alerts, signals, turns
from app.observability.store import ConversationTurnRow, get_sessionmaker, reset_engine
from app.request_context import (
    RequestContextMiddleware,
    get_trace_context,
    outbound_traceparent,
    parse_traceparent,
)

client = TestClient(app)

OPS_KEY = "test-ops-key"
TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_SPAN = "00f067aa0ba902b7"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        settings, "turn_store_url", f"sqlite:///{tmp_path / 'turns.db'}"
    )
    monkeypatch.setattr(settings, "turn_observability_enabled", True)
    reset_engine()
    signals.reset()
    yield
    reset_engine()
    signals.reset()


def _row(latency_ms: float | None, minutes_ago: float = 1.0) -> ConversationTurnRow:
    return ConversationTurnRow(
        timestamp=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        trace_id=None,
        session_ref="ref",
        customer_ref=None,
        language="en",
        intent=None,
        status="collecting",
        pending_slot=None,
        reason_code=None,
        latency_ms=latency_ms,
        slots_masked="{}",
    )


def _store(*rows: ConversationTurnRow) -> None:
    with get_sessionmaker()() as session:
        session.add_all(rows)
        session.commit()


# --------------------------------------------------------------------------- #
# p95 — measured over turns, from the durable store
# --------------------------------------------------------------------------- #
def test_the_percentile_is_a_latency_a_turn_actually_took() -> None:
    """Nearest-rank, so the number quoted is always an observed one."""

    _store(*(_row(float(value)) for value in range(1, 101)))

    assert turns.latency_percentile(60, 95.0) == 95.0
    assert turns.latency_percentile(60, 50.0) == 50.0


def test_turns_outside_the_window_do_not_count() -> None:
    _store(_row(50.0), _row(9000.0, minutes_ago=180))

    assert turns.latency_percentile(60) == 50.0


def test_no_traffic_reads_as_unavailable_not_as_zero() -> None:
    """A silent hour must not present itself as a perfect one."""

    assert turns.latency_percentile(60) is None

    readings = alerts.read_signals({alerts.P95_LATENCY: None})
    latency = next(r for r in readings if r.signal.key == alerts.P95_LATENCY)
    assert latency.available is False
    assert latency.breached is False


def test_a_slow_hour_breaches_the_latency_objective() -> None:
    ceiling = next(s for s in alerts.SIGNALS if s.key == alerts.P95_LATENCY).max_value
    readings = alerts.read_signals({alerts.P95_LATENCY: ceiling + 1})

    latency = next(r for r in readings if r.signal.key == alerts.P95_LATENCY)
    assert latency.breached is True


# --------------------------------------------------------------------------- #
# Dependency health — from real calls, never a probe
# --------------------------------------------------------------------------- #
def test_health_follows_the_last_real_call() -> None:
    signals.record_call(signals.BANKING_CORE, ok=True)
    signals.record_call(signals.BANKING_CORE, ok=False)
    assert signals.unhealthy_count() == 1

    signals.record_call(signals.BANKING_CORE, ok=True)
    assert signals.unhealthy_count() == 0

    state = signals.dependencies()[0]
    assert state.attempts == 3
    assert state.failures == 1
    assert state.consecutive_failures == 0
    assert state.last_success is not None


def test_an_unreachable_core_is_recorded_without_breaking_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client still degrades to ``None``; the failure is merely visible."""

    monkeypatch.setattr(settings, "banking_core_enabled", True)

    def _boom(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _boom)

    assert banking_core_client.get_balance("u-1") is None
    assert signals.unhealthy_count() == 1


def test_a_core_answering_no_is_not_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown account is the Core working, so health must stay green."""

    monkeypatch.setattr(settings, "banking_core_enabled", True)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(404, request=httpx.Request("POST", "http://x")),
    )

    assert banking_core_client.get_balance("u-1") is None
    assert signals.unhealthy_count() == 0
    assert signals.dependencies()[0].healthy is True


def test_a_dependency_never_called_reads_as_unknown() -> None:
    """Silence is not health: with no calls there is no reading to report."""

    assert signals.dependencies() == []

    readings = alerts.read_signals({alerts.UNHEALTHY_DEPENDENCIES: None})
    deps = next(r for r in readings if r.signal.key == alerts.UNHEALTHY_DEPENDENCIES)
    assert deps.available is False
    assert deps.breached is False


def test_one_unhealthy_dependency_breaches() -> None:
    readings = alerts.read_signals({alerts.UNHEALTHY_DEPENDENCIES: 1.0})
    deps = next(r for r in readings if r.signal.key == alerts.UNHEALTHY_DEPENDENCIES)
    assert deps.breached is True
    assert deps.signal.severity is alerts.Severity.CRITICAL


# --------------------------------------------------------------------------- #
# Model availability — a call that never came back, not a rejected rewrite
# --------------------------------------------------------------------------- #
def _handler() -> LLMExceptionHandler:
    return LLMExceptionHandler(
        model="m", api_base="http://llm.invalid", timeout=1.0, temperature=0.0
    )


def test_a_model_that_answers_is_not_counted_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wording the guard later refuses is the guard working, not an outage."""

    import litellm

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: {"choices": [{"message": {"content": "a nicer wording"}}]},
    )

    assert _handler().rephrase("Your balance is 1,000 SAR.", "en", 1.0)
    assert signals.failure_ratio(signals.LLM) == (0, 1)
    assert signals.unhealthy_count() == 0


def test_a_model_call_that_never_returns_is_counted_and_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    def _boom(**kwargs: object) -> dict[str, object]:
        raise TimeoutError("no answer")

    monkeypatch.setattr(litellm, "completion", _boom)

    assert _handler().rephrase("Your balance is 1,000 SAR.", "en", 1.0) is None
    assert signals.failure_ratio(signals.LLM) == (1, 1)
    assert signals.unhealthy_count() == 1


def test_the_rate_is_calls_that_failed_over_calls_attempted() -> None:
    signals.record_call(signals.LLM, ok=True)
    signals.record_call(signals.LLM, ok=False)
    signals.record_call(signals.LLM, ok=True)
    signals.record_call(signals.LLM, ok=True)

    failures, attempts = signals.failure_ratio(signals.LLM)
    assert (failures, attempts) == (1, 4)


# --------------------------------------------------------------------------- #
# W3C traceparent
# --------------------------------------------------------------------------- #
def test_a_valid_traceparent_is_joined_with_a_fresh_span() -> None:
    context = parse_traceparent(f"00-{TRACE}-{PARENT_SPAN}-01")

    assert context is not None
    assert context.trace_id == TRACE
    assert context.parent_span_id == PARENT_SPAN
    assert context.span_id != PARENT_SPAN
    assert context.header() == f"00-{TRACE}-{context.span_id}-01"


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "garbage",
        f"01-{TRACE}-{PARENT_SPAN}-01",  # unknown version
        f"00-{'0' * 32}-{PARENT_SPAN}-01",  # all-zero trace id
        f"00-{TRACE}-{'0' * 16}-01",  # all-zero span id
        f"00-{TRACE[:31]}-{PARENT_SPAN}-01",  # wrong length
    ],
)
def test_an_unusable_traceparent_is_not_joined(header: str | None) -> None:
    """Better a new local id than a malformed one propagated onwards."""

    assert parse_traceparent(header) is None


def test_the_request_adopts_the_inbound_trace_and_echoes_it() -> None:
    response = client.post(
        "/conversation/text",
        json={"text": "hello", "session_id": "trace-1"},
        headers={"traceparent": f"00-{TRACE}-{PARENT_SPAN}-01"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == TRACE
    echoed = response.headers["traceparent"]
    assert echoed.startswith(f"00-{TRACE}-")
    assert echoed.endswith("-01")
    # The turn row is filed against the caller's trace, so an APM trace and a
    # decision row can be lined up without either service inventing an id.
    assert turns.list_turns(limit=1)[0].trace_id == TRACE


def test_x_request_id_still_works_when_no_trace_is_sent() -> None:
    response = client.post(
        "/conversation/text",
        json={"text": "hello", "session_id": "trace-2"},
        headers={"x-request-id": "legacy-id-1"},
    )

    assert response.headers["x-request-id"] == "legacy-id-1"
    assert "traceparent" not in response.headers


def test_the_trace_is_passed_on_to_the_banking_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One customer request is followable across both services."""

    monkeypatch.setattr(settings, "banking_core_enabled", True)
    seen: dict[str, str] = {}

    def _capture(url: str, **kwargs: object) -> httpx.Response:
        seen.update(dict(kwargs.get("headers") or {}))  # type: ignore[arg-type]
        return httpx.Response(404, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _capture)

    # Drive the middleware directly: the trace lives in a ContextVar, so the
    # outbound header has to be produced inside a request scope.
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    def _endpoint(request):  # type: ignore[no-untyped-def]
        banking_core_client.get_balance("u-1")
        return JSONResponse({"traceparent": outbound_traceparent()})

    inner = Starlette(routes=[Route("/probe", _endpoint)])
    inner.add_middleware(RequestContextMiddleware)
    with TestClient(inner) as probe:
        response = probe.get(
            "/probe", headers={"traceparent": f"00-{TRACE}-{PARENT_SPAN}-01"}
        )

    assert seen["traceparent"].startswith(f"00-{TRACE}-")
    assert response.json()["traceparent"] == seen["traceparent"]


def test_no_trace_outside_a_request() -> None:
    assert get_trace_context() is None
    assert outbound_traceparent() is None


# --------------------------------------------------------------------------- #
# The authenticated read
# --------------------------------------------------------------------------- #
def test_the_signal_read_is_closed_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ops_api_key", None)
    assert client.get("/ops/observability/signals").status_code == 503

    monkeypatch.setattr(settings, "ops_api_key", OPS_KEY)
    assert client.get("/ops/observability/signals").status_code == 401
    assert (
        client.get(
            "/ops/observability/signals", headers={"x-ops-key": "wrong"}
        ).status_code
        == 401
    )


def test_the_signal_read_reports_every_objective_and_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ops_api_key", OPS_KEY)
    _store(_row(120.0), _row(240.0))
    signals.record_call(signals.BANKING_CORE, ok=False)
    signals.record_call(signals.LLM, ok=False)
    signals.record_call(signals.LLM, ok=True)

    body = client.get(
        "/ops/observability/signals", headers={"x-ops-key": OPS_KEY}
    ).json()

    by_key = {entry["key"]: entry for entry in body["signals"]}
    assert set(by_key) == {
        alerts.P95_LATENCY,
        alerts.LLM_UNAVAILABLE_RATE,
        alerts.UNHEALTHY_DEPENDENCIES,
    }
    assert by_key[alerts.P95_LATENCY]["value"] == 240.0
    assert by_key[alerts.LLM_UNAVAILABLE_RATE]["value"] == 50.0
    assert by_key[alerts.UNHEALTHY_DEPENDENCIES]["value"] == 1.0
    assert by_key[alerts.UNHEALTHY_DEPENDENCIES]["breached"] is True
    assert body["breaching"] >= 1

    core = next(d for d in body["dependencies"] if d["name"] == signals.BANKING_CORE)
    assert core["healthy"] is False
    assert core["last_attempt"] is not None
    # Health is inferred from real calls, so a reader is told when it was seen.
    assert core["last_success"] is None


def test_the_signal_read_carries_no_customer_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The imported design shipped conversation text here. This one cannot."""

    monkeypatch.setattr(settings, "ops_api_key", OPS_KEY)
    client.post(
        "/conversation/text",
        json={"text": "transfer 250 SAR to Sara Adel", "session_id": "sig-pii"},
    )

    body = client.get("/ops/observability/signals", headers={"x-ops-key": OPS_KEY}).text

    for secret in ("Sara", "250", "SAR", "transfer 250"):
        assert secret not in body


def test_the_ratio_catalogue_is_untouched_by_the_new_signals() -> None:
    """Operational numbers live in their own catalogue, not in ``Slo``."""

    assert all(isinstance(slo, alerts.Slo) for slo in alerts.CATALOGUE)
    assert all(isinstance(signal, alerts.OpsSignal) for signal in alerts.SIGNALS)
    assert not {slo.key for slo in alerts.CATALOGUE} & {
        signal.key for signal in alerts.SIGNALS
    }
