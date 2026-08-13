"""Tests for P1 observability: readiness, metrics, request-id, /v1."""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from app.logging_config import JsonFormatter
from app.main import app
from app.request_context import REQUEST_ID_HEADER, set_request_id

client = TestClient(app)


def test_readiness_ok():
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["store"] == "ok"


def test_liveness_still_ok():
    assert client.get("/health").json()["status"] == "ok"


def test_metrics_endpoint_exposes_prometheus():
    client.post("/nlu/parse", json={"text": "send 5 dollars to Ahmed"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "nlu_http_requests_total" in resp.text
    assert "nlu_http_request_duration_seconds" in resp.text


def test_request_id_generated_and_returned():
    resp = client.get("/health")
    rid = resp.headers.get(REQUEST_ID_HEADER)
    assert rid and len(rid) == 16


def test_request_id_honours_inbound_header():
    resp = client.get("/health", headers={REQUEST_ID_HEADER: "trace-123"})
    assert resp.headers.get(REQUEST_ID_HEADER) == "trace-123"


def test_v1_alias_works():
    resp = client.post("/v1/nlu/parse", json={"text": "send 5 dollars to Ahmed"})
    assert resp.status_code == 200
    assert resp.json()["intent"] == "transfer_money"


def test_json_formatter_includes_request_id():
    set_request_id("req-xyz")
    try:
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1, "hello %s", ("world",), None
        )
        rendered = json.loads(JsonFormatter().format(record))
        assert rendered["message"] == "hello world"
        assert rendered["request_id"] == "req-xyz"
        assert rendered["level"] == "INFO"
    finally:
        set_request_id(None)
