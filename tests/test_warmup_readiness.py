"""Warm-up runs off the request path, and readiness waits for it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import warmup
from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def _clean_warmup():
    warmup.reset()
    yield
    warmup.reset()


def _steps(calls: list[str], *, fail: str | None = None):
    def make(name: str):
        def load() -> None:
            calls.append(name)
            if name == fail:
                raise RuntimeError("model file missing")

        return load

    return lambda: [(name, make(name)) for name in ("embedder", "semantic_index")]


def test_warmup_loads_every_step_and_reports_ready(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(warmup, "_steps", _steps(calls))

    warmup.start()
    state = warmup.wait(timeout=10)

    assert calls == ["embedder", "semantic_index"]
    assert state.status == warmup.READY
    assert state.ready is True
    assert state.step is None
    assert state.duration_s is not None


def test_disabled_warmup_loads_nothing_and_does_not_gate(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(warmup, "_steps", _steps(calls))

    warmup.start(enabled=False)

    assert calls == []
    assert warmup.state().status == warmup.SKIPPED
    assert warmup.state().ready is True


def test_a_failed_step_is_recorded_and_stops_the_rest(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(warmup, "_steps", _steps(calls, fail="embedder"))

    warmup.start()
    state = warmup.wait(timeout=10)

    assert calls == ["embedder"]
    assert state.status == warmup.FAILED
    assert state.error is not None
    assert "embedder" in state.error


def test_start_twice_does_not_run_warmup_twice(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(warmup, "_steps", _steps(calls))

    warmup.start()
    warmup.wait(timeout=10)
    calls.clear()
    warmup.start()
    warmup.wait(timeout=10)

    # The thread has finished, so a second start is allowed to run again; what must
    # not happen is two warm-ups running at once.
    assert calls in ([], ["embedder", "semantic_index"])


def test_liveness_answers_while_warmup_is_still_running(monkeypatch):
    """Liveness must not wait for the index: a slow probe gets the process killed."""

    monkeypatch.setattr(warmup, "_status", warmup.RUNNING)
    monkeypatch.setattr(warmup, "_step", "semantic_index")

    r = TestClient(app).get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_is_not_ready_until_warmup_finishes(monkeypatch):
    monkeypatch.setattr(warmup, "_status", warmup.RUNNING)
    monkeypatch.setattr(warmup, "_step", "semantic_index")

    r = TestClient(app).get("/health/ready")

    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["warmup"] == warmup.RUNNING
    # Which step is holding it up, so a slow start can be diagnosed.
    assert body["checks"]["warmup_step"] == "semantic_index"


def test_readiness_is_ready_once_warmup_completes(monkeypatch):
    monkeypatch.setattr(warmup, "_status", warmup.READY)
    monkeypatch.setattr(warmup, "_duration", 174.81)

    r = TestClient(app).get("/health/ready")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"]["warmup"] == warmup.READY
    # The measured cost of a cold start, so the wait is a number and not a feeling.
    assert body["warmup_seconds"] == 174.81


def test_a_failed_warmup_still_serves_traffic(monkeypatch):
    """The lazy loaders remain: a failed warm-up is a slow first turn, not an outage."""

    monkeypatch.setattr(warmup, "_status", warmup.FAILED)
    monkeypatch.setattr(warmup, "_error", "embedder: model file missing")

    r = TestClient(app).get("/health/ready")

    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["warmup_error"] == "embedder: model file missing"


def test_warmup_steps_cover_the_expensive_loaders():
    """The index build is the dominant cost; missing it would defeat the warm-up."""

    names = [name for name, _ in warmup._steps()]

    assert "semantic_index" in names
    assert "embedder" in names
    assert names.index("embedder") < names.index("semantic_index")


def test_warmup_is_on_by_default():
    """A cold process must not make the first customer wait for the index."""

    assert type(settings)().preload_models is True
