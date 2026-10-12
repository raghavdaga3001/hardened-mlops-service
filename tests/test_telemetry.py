import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.telemetry import DriftMonitor

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_drift_monitor_detects_distribution_shift():
    """Verify statistical drift flags when features diverge from golden baseline."""
    monitor = DriftMonitor(baseline_path="baseline/baseline_data.parquet")
    
    # Generate 25 records with heavy synthetic drift on transaction_amount and velocity_score
    shifted_batch = [
        {
            "transaction_amount": float(np.random.normal(loc=15.0, scale=1.5)),
            "velocity_score": float(np.random.normal(loc=10.0, scale=1.0)),
            "account_age_months": 12.0,
            "auth_attempts": 3.0,
            "device_trust_score": 0.2,
            "risk_score": 0.9
        }
        for _ in range(25)
    ]
    
    results = monitor.evaluate_batch(shifted_batch, html_report_path=None)
    
    assert results["summary"]["drift_detected"] is True
    # KS-test null hypothesis rejected (p < 0.05)
    assert results["features"]["transaction_amount"]["drift_flag"] is True
    assert results["features"]["transaction_amount"]["p_value"] < 0.05
    # Wasserstein distance reflects wide distribution divergence
    assert results["features"]["transaction_amount"]["wasserstein_distance"] > 5.0

def test_telemetry_endpoint_batch_size_guard(client):
    """Enforce minimum batch size constraint (n >= 15) for distribution checks."""
    small_batch = [
        {
            "transaction_amount": 10.0,
            "velocity_score": 1.0,
            "account_age_months": 12.0,
            "auth_attempts": 1.0,
            "device_trust_score": 0.9,
            "risk_score": 0.1
        }
        for _ in range(5)
    ]
    response = client.post("/telemetry/drift-report", json=small_batch)
    assert response.status_code == 400
    assert "Minimum batch size of 15" in response.json()["detail"]

def test_telemetry_endpoint_success(client):
    """Verify endpoint successfully parses a valid batch and returns telemetry schema."""
    valid_batch = [
        {
            "transaction_amount": float(np.random.normal(loc=0.0, scale=1.0)),
            "velocity_score": float(np.random.normal(loc=0.0, scale=1.0)),
            "account_age_months": 24.0,
            "auth_attempts": 1.0,
            "device_trust_score": 0.85,
            "risk_score": 0.15
        }
        for _ in range(20)
    ]
    response = client.post("/telemetry/drift-report", json=valid_batch)
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["batch_size"] == 20
    assert "transaction_amount" in payload["features"]
    assert "ks_statistic" in payload["features"]["transaction_amount"]
    assert "wasserstein_distance" in payload["features"]["transaction_amount"]