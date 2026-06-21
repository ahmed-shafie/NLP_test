"""Tests for API-key auth, error envelope, security headers, and body limits."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture
def auth_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "api_keys", "pub-secret")
    monkeypatch.setattr(settings, "admin_api_keys", "admin-secret")
    yield


def test_public_endpoint_open_when_auth_disabled():
    # Default settings: auth disabled -> no key required.
    resp = client.post("/nlu/parse", json={"text": "send 10 dollars to Ahmed"})
    assert resp.status_code == 200


def test_public_endpoint_requires_key_when_auth_on(auth_on):
    resp = client.post("/nlu/parse", json={"text": "send 10 dollars to Ahmed"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_public_endpoint_rejects_wrong_key(auth_on):
    resp = client.post(
        "/nlu/parse",
        json={"text": "send 10 dollars to Ahmed"},
        headers={"X-API-Key": "nope"},
    )
    assert resp.status_code == 401


def test_public_endpoint_accepts_valid_key(auth_on):
    resp = client.post(
        "/nlu/parse",
        json={"text": "send 10 dollars to Ahmed"},
        headers={"X-API-Key": "pub-secret"},
    )
    assert resp.status_code == 200


def test_admin_requires_admin_key(auth_on):
    # A valid public key must NOT grant admin access.
    resp = client.get("/admin/api/connections", headers={"X-API-Key": "pub-secret"})
    assert resp.status_code == 401
    resp_ok = client.get(
        "/admin/api/connections", headers={"X-Admin-Key": "admin-secret"}
    )
    assert resp_ok.status_code == 200


def test_auth_fails_closed_without_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "api_keys", "")
    resp = client.post("/nlu/parse", json={"text": "hi"})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "service_unavailable"


def test_error_envelope_on_not_found():
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == "not_found"


def test_security_headers_present():
    resp = client.get("/health")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"


def test_body_size_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "max_request_bytes", 50)
    big = "x" * 200
    resp = client.post("/nlu/parse", json={"text": big})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"
