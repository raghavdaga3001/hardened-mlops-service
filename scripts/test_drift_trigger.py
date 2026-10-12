import requests
import numpy as np

# Simulate an adversarial distribution shift
drifted_batch = []
for _ in range(30):
    drifted_batch.append({
        "transaction_amount": float(np.random.normal(loc=15.0, scale=2.0)),
        "velocity_score": float(np.random.normal(loc=9.0, scale=1.5)),
        "account_age_months": 12.0,
        "auth_attempts": 4.0,
        "device_trust_score": 0.1,
        "risk_score": 0.95
    })

res = requests.post("http://127.0.0.1:8000/telemetry/drift-report", json=drifted_batch)
print(f"Status Code: {res.status_code}")

if res.status_code == 200:
    data = res.json()
    print("Summary:", data["summary"])
    print("Transaction Amount Metrics:", data["features"]["transaction_amount"])
else:
    print("Server Error Detail:")
    print(res.text)