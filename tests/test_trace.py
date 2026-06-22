"""Tests for the block_trace observability field and the BlockTracer helper."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.orchestration as orchestration
from app.main import app
from app.trace import BlockTracer

client = TestClient(app)


def test_tracer_records_ok_skipped_and_error():
    tracer = BlockTracer()
    with tracer.block("a"):
        pass
    with tracer.block("b") as span:
        span.skip("nothing to do")
    with pytest.raises(ValueError):
        with tracer.block("c"):
            raise ValueError("boom")

    statuses = {e.block: e.status for e in tracer.entries}
    assert statuses == {"a": "ok", "b": "skipped", "c": "error"}
    by_name = {e.block: e for e in tracer.entries}
    assert by_name["b"].note == "nothing to do"
    assert "boom" in by_name["c"].note
    assert all(e.duration_ms >= 0 for e in tracer.entries)


def test_run_pipeline_emits_full_block_trace():
    result = orchestration.run_pipeline("send 500 dollars to Ahmed")
    names = [e.block for e in result.block_trace]
    assert names == [
        "language_detection",
        "intent_classification",
        "entity_extraction",
        "contact_resolution",
        "beneficiary_lookup",
        "llm_fallback",
        "active_learning",
    ]
    # A complete deterministic transfer skips the beneficiary + LLM blocks.
    by_name = {e.block: e for e in result.block_trace}
    assert by_name["language_detection"].status == "ok"
    assert by_name["beneficiary_lookup"].status == "skipped"
    assert by_name["llm_fallback"].status == "skipped"


def test_non_transfer_marks_entity_block_skipped():
    result = orchestration.run_pipeline("what is the weather today")
    by_name = {e.block: e for e in result.block_trace}
    assert by_name["entity_extraction"].status == "skipped"
    assert by_name["entity_extraction"].note == "non-transfer intent"


def test_parse_endpoint_includes_block_trace():
    resp = client.post("/nlu/parse", json={"text": "send 500 dollars to Ahmed"})
    assert resp.status_code == 200
    blocks = resp.json()["block_trace"]
    assert [b["block"] for b in blocks][0] == "language_detection"
    assert blocks[-1]["block"] == "active_learning"


def test_conversation_endpoint_includes_block_trace():
    resp = client.post("/conversation/text", json={"text": "send 500 dollars to Ahmed"})
    assert resp.status_code == 200
    names = [b["block"] for b in resp.json()["block_trace"]]
    # The turn restores memory, runs the NLU blocks, then the orchestrator FSM.
    assert names[0] == "memory_restore"
    assert "intent_classification" in names
    assert names[-1] == "orchestrator"
