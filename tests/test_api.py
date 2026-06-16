"""API endpoint tests using the FastAPI test client."""

from fastapi.testclient import TestClient

from app.embeddings import get_embedder
from app.main import app

client = TestClient(app)

_HAS_EMBEDDER = get_embedder() is not None


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


def test_contacts_resolve_endpoint():
    resp = client.post(
        "/contacts/resolve",
        json={
            "name": "Ahmed",
            "contacts": [
                {"id": "1", "name": "Ahmed Hassan", "account": "A1"},
                {"id": "2", "name": "Sara Adel", "account": "A2"},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["candidates"]) >= 1
    if _HAS_EMBEDDER:
        assert data["matched"] is not None
        assert data["matched"]["contact"]["name"] == "Ahmed Hassan"


def test_similar_endpoint():
    resp = client.get("/nlu/similar", params={"text": "send 100 to John", "k": 3})
    assert resp.status_code == 200
    data = resp.json()
    if _HAS_EMBEDDER:
        assert len(data) == 3
        assert data[0]["score"] >= data[-1]["score"]


def test_parse_includes_intent_source():
    resp = client.post("/nlu/parse", json={"text": "send 200 dollars to Ahmed"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent_source"] in {"semantic", "keyword"}
