import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_readiness(client):
    response = client.get("/health/readiness")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_predict_endpoint(client):
    payload = {
        "transaction_amount": 100.0,
        "velocity_score": 1.5,
        "account_age_months": 24.0,
        "auth_attempts": 1.0,
        "device_trust_score": 0.9,
        "risk_score": 0.1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "confidence" in data