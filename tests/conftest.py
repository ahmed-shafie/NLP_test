"""Shared pytest fixtures.

By default we disable the live LiteLLM exception handler so the suite is fast and
deterministic and does not require a running Ollama server. Tests that exercise the
LLM path explicitly install a fake handler via monkeypatch.

Reply variation is disabled for the same reason: with it on, a conversational reply
is any one of its hand-written phrasings, which makes wording assertions flaky.
Off means every reply renders its first phrasing, so the suite compares exact text.
Variation itself is covered explicitly in ``test_phrasing.py``.
"""

from __future__ import annotations

import pytest

import app.orchestration as orchestration
from app.config import settings


@pytest.fixture(autouse=True)
def _disable_live_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the orchestration pipeline skip the real LLM unless a test opts in."""

    monkeypatch.setattr(orchestration, "get_llm_handler", lambda: None)
    orchestration.get_nlu_pipeline.cache_clear()
    yield
    orchestration.get_nlu_pipeline.cache_clear()


@pytest.fixture(autouse=True)
def _deterministic_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render one fixed phrasing per reply unless a test opts into variation."""

    monkeypatch.setattr(settings, "reply_variation_enabled", False)
    monkeypatch.setattr(settings, "reply_rewrite_enabled", False)
