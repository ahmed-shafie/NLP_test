"""Tests for the uniform error envelope and request body-size limit."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_public_endpoint_open():
    resp = client.post("/nlu/parse", json={"text": "send 10 dollars to Ahmed"})
    assert resp.status_code == 200


def test_error_envelope_on_not_found():
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == "not_found"


def test_body_size_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "max_request_bytes", 50)
    big = "x" * 200
    resp = client.post("/nlu/parse", json={"text": big})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"
