"""Tests for audit recording, querying, and store-backed observability stats."""

from __future__ import annotations

import pytest

import app.admin.store as store
from app.admin import audit
from app.config import settings


@pytest.fixture()
def isolated_audit(tmp_path, monkeypatch):
    """Isolate the audit store and disable network sinks for deterministic tests."""

    monkeypatch.setattr(settings, "admin_store_url", f"sqlite:///{tmp_path/'audit.db'}")
    monkeypatch.setattr(settings, "audit_enabled", True)
    monkeypatch.setattr(settings, "audit_sink", "none")
    store.get_engine.cache_clear()
    store.get_sessionmaker.cache_clear()
    yield
    store.get_engine.cache_clear()
    store.get_sessionmaker.cache_clear()


def test_record_and_list(isolated_audit):
    audit.record("nlu.parse", category="nlu", outcome="success", duration_ms=12.0)
    audit.record("GET /x", category="http", status_code=200, duration_ms=5.0)
    audit.record("GET /y", category="http", status_code=500, outcome="error")

    events = audit.list_events()
    assert len(events) == 3
    # Newest first.
    assert events[0].action == "GET /y"


def test_list_filters(isolated_audit):
    audit.record("nlu.parse", category="nlu")
    audit.record("GET /x", category="http", status_code=200)
    audit.record("GET /y", category="http", status_code=500, outcome="error")

    assert len(audit.list_events(category="http")) == 2
    assert len(audit.list_events(outcome="error")) == 1
    assert len(audit.list_events(action="parse")) == 1


def test_stats_from_store(isolated_audit):
    audit.record("GET /x", category="http", status_code=200, duration_ms=10.0)
    audit.record("GET /x", category="http", status_code=200, duration_ms=30.0)
    audit.record(
        "GET /y", category="http", status_code=404, outcome="error", duration_ms=20.0
    )

    stats = audit.stats_from_store()
    assert stats.source == "store"
    assert stats.total == 3
    assert stats.success == 2
    assert stats.errors == 1
    assert stats.by_status["200"] == 2
    assert stats.by_status["404"] == 1
    assert stats.avg_duration_ms == pytest.approx(20.0, abs=0.1)
    assert stats.by_action["GET /x"] == 2


def test_get_stats_falls_back_to_store(isolated_audit, monkeypatch):
    # Even with the elasticsearch sink selected, an unavailable ES falls back.
    monkeypatch.setattr(settings, "audit_sink", "elasticsearch")
    monkeypatch.setattr(audit.elk, "fetch_stats", lambda window_minutes=1440: None)
    audit.record("nlu.parse", category="nlu", status_code=200)
    stats = audit.get_stats()
    assert stats.source == "store"
    assert stats.total == 1


def test_disabled_audit_records_nothing(isolated_audit, monkeypatch):
    monkeypatch.setattr(settings, "audit_enabled", False)
    audit.record("nlu.parse", category="nlu")
    assert audit.list_events() == []
