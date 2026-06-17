"""Shared pytest fixtures.

By default we disable the live LiteLLM exception handler so the suite is fast and
deterministic and does not require a running Ollama server. Tests that exercise the
LLM path explicitly install a fake handler via monkeypatch.
"""

from __future__ import annotations

import pytest

import app.orchestration as orchestration


@pytest.fixture(autouse=True)
def _disable_live_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the orchestration pipeline skip the real LLM unless a test opts in."""

    monkeypatch.setattr(orchestration, "get_llm_handler", lambda: None)
    orchestration.get_nlu_pipeline.cache_clear()
    yield
    orchestration.get_nlu_pipeline.cache_clear()
