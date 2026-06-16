"""API endpoint tests using the FastAPI test client."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_parse_endpoint():
    resp = client.post("/nlu/parse", json={"text": "send 200 dollars to Ahmed"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "transfer_money"
    assert data["entities"]["amount"] == "200"
    assert data["entities"]["currency"] == "USD"


def test_parse_arabic_endpoint():
    resp = client.post("/nlu/parse", json={"text": "حوّل ٥٠٠ جنيه إلى أحمد"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "transfer_money"
    assert data["language"] == "ar"


def test_validate_endpoint_success():
    resp = client.post(
        "/transfer/validate",
        json={"amount": 100, "currency": "EGP", "recipient": "Ali"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["transfer"]["recipient"] == "Ali"


def test_validate_endpoint_missing_fields():
    resp = client.post(
        "/transfer/validate", json={"amount": None, "currency": None, "recipient": None}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0 or len(data["missing"]) > 0
