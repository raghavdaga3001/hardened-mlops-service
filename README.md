<div align="center">

# 🛡️ Hardened MLOps Service & Real-Time Drift Telemetry

**Enterprise-Grade Model Serving Pipeline with Supply-Chain Hardening & Statistical Drift Observability**

[![CI & Security](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Optimized-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Security](https://img.shields.io/badge/Security-Bandit%20%7C%20Safetensors-brightgreen)](https://github.com/huggingface/safetensors)
[![Compliance](https://img.shields.io/badge/EU%20AI%20Act-Articles%2012%20%26%2015-orange)](#-regulatory-alignment--governance)

<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-statistical-telemetry-engine">Telemetry Math</a> •
  <a href="#-container-security">Container Hardening</a> •
  <a href="#-verification--testing">Verification</a>
</p>

</div>

---

## 📌 Executive Overview

In production environments, machine learning deployments face dual threats: **supply-chain vulnerabilities** (arbitrary code execution through unsafe weight serialization) and **silent model degradation** (statistical feature and representation drift).

This repository implements a production-grade inference microservice engineered with:
1. **Supply-chain-safe serialization** (`.safetensors` format exclusively) to eliminate Python `pickle` deserialization hazards (OWASP LLM04 / Remote Code Execution).
2. **Asynchronous lifespan resource management** with decoupled Kubernetes readiness and liveness health probes.
3. **Automated post-market drift telemetry** running two-sample Kolmogorov-Smirnov (KS) tests and Wasserstein distances against golden reference baselines.
4. **Least-privilege containerization** via multi-stage builds executed under an unprivileged user (`UID 10001:10001`).

---

## 🏛️ System Architecture

```text
[ Client Traffic / Ingestion ]
              │
              ▼
   ┌──────────────────────────────────────────────┐
   │             FastAPI Gateway                  │
   │  • Strict Pydantic Schema Validation         │
   │  • Decoupled /health/readiness & /liveness   │
   └──────────────┬───────────────────────────────┘
                  │
        ┌─────────┴────────────────────────┐
        ▼                                  ▼
┌───────────────────────────────┐ ┌────────────────────────────────────────┐
│     Inference Engine          │ │       Drift Telemetry Worker           │
│ • PyTorch via .safetensors    │ │ • Baseline Parquet Store               │
│ • Zero Bytecode Execution     │ │ • Two-Sample Kolmogorov-Smirnov Test   │
│ • Real-time Decision Logits   │ │ • Wasserstein Metric (Earth Mover's)   │
│ • Strict Memory Allocation    │ │ • Automated Distribution Alerting      │
└───────────────────────────────┘ └────────────────────────────────────────┘
```

---

## 🚀 Key Features

| Capability | Implementation | Technical Advantage |
| :--- | :--- | :--- |
| **Model Serialization** | Hugging Face `.safetensors` | Bypasses `pickle.__reduce__()` vulnerabilities; pure binary tensor storage with structured JSON headers. |
| **Lifecycle Serving** | FastAPI `@asynccontextmanager` | Pre-allocates and memory-maps weights *prior* to network binding, preventing traffic routing during cold starts. |
| **Drift Monitoring** | SciPy + Evidently AI | Calculates non-parametric statistical divergence on batches ($N \ge 15$) against reference Parquet baselines. |
| **Container Hardening** | Multi-Stage `python:3.11-slim` | Drops build tools (`gcc`, headers); runs strictly as non-root user `10001:10001`. |
| **CI/CD Quality Gates** | GitHub Actions + Bandit | Automated test execution on push, paired with static Abstract Syntax Tree (AST) security analysis. |

---

## ⚖️ Regulatory Alignment & Governance

This architecture operationalizes key technical mandates for **High-Risk AI Systems (Annex III)** under the **EU Artificial Intelligence Act**:

* **Article 12 (Record-Keeping & Traceability):** Structured input validation and batch telemetry payloads enable continuous, tamper-evident audit logging of system decisions and operational inputs.
* **Article 15 (Accuracy, Robustness & Cybersecurity):** Continuous post-market monitoring using non-parametric statistical metrics flags input distribution divergence before catastrophic model failure occurs.

---

## 📊 Statistical Telemetry Engine

The monitoring module benchmarks incoming inference batches against a curated reference dataset (`baseline/baseline_data.parquet`):

### 1. Two-Sample Kolmogorov-Smirnov (KS) Test
Used for individual continuous numerical feature streams. Evaluates the empirical cumulative distribution function (eCDF) of reference data $F_{ref}(x)$ versus production data $F_{prod}(x)$:

$$D = \sup_x \vert{}F_{ref}(x) - F_{prod}(x)\vert{}$$

* **Null Hypothesis ($H_0$):** Inbound inference batch comes from the training baseline distribution.
* **Rejection Boundary:** Flagged as drifted when $p < 0.05$.

### 2. Wasserstein Distance (Earth Mover's Distance)
Quantifies the minimal transportation cost required to transform the current distribution into the baseline reference distribution:

$$W_1(P_{ref}, P_{prod}) = \int_{-\infty}^{+\infty} \vert{}F_{ref}(x) - F_{prod}(x)\vert{} \, dx$$

---

## 🛠️ Quickstart

### Local Environment Setup (WSL2 / Linux)

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)raghavdaga3001/hardened-mlops-service.git
cd hardened-mlops-service

# 2. Initialize virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install locked dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate reference baseline data & .safetensors model weights
python scripts/train_and_export.py

# 5. Launch the FastAPI server
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## 🐳 Container Security & Docker Deployment

The application utilizes a hardened multi-stage build pattern that separates compilation tooling from runtime execution:

```bash
# 1. Build the production image
docker build -t hardened-mlops-service:v1.0.0 .

# 2. Verify non-root execution (must return uid=10001)
docker run --rm hardened-mlops-service:v1.0.0 id
# Output: uid=10001(appuser) gid=10001(appgroup) groups=10001(appgroup)

# 3. Run the containerized service
docker run -d --name mlops-service -p 8000:8000 hardened-mlops-service:v1.0.0
```

---

## 🧪 Verification & Testing

### 1. Automated Test Suite & Static Security Scans
Run local unit tests and static AST vulnerability checks:

```bash
# Execute Pytest test cases
pytest -v tests/

# Execute static security analysis via Bandit
bandit -r app/ -ll -ii
```

### 2. Live API Testing

#### Health & Readiness Probes
```bash
curl -i [http://127.0.0.1:8000/health/readiness](http://127.0.0.1:8000/health/readiness)
```

#### Synchronous Inference
```bash
curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 120.50,
    "velocity_score": 1.2,
    "account_age_months": 24.0,
    "auth_attempts": 1.0,
    "device_trust_score": 0.95,
    "risk_score": 0.15
  }'
```

#### Trigger Distribution Shift & Telemetry Report
```bash
python scripts/test_drift_trigger.py
```

---

## 📁 Repository Structure

```text
hardened-mlops-service/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI runner & Bandit AST scan
├── app/
│   ├── __init__.py
│   ├── main.py                  # Lifespan gateway & Pydantic schemas
│   ├── model.py                 # Pure tensor loading via safetensors
│   └── telemetry.py             # KS-test & Wasserstein drift calculations
├── baseline/
│   ├── baseline_data.parquet    # Columnar golden reference distribution
│   └── model.safetensors        # Supply-chain-safe weights
├── scripts/
│   ├── train_and_export.py      # Artifact generator script
│   └── test_drift_trigger.py    # Synthetic distribution shift harness
├── tests/
│   ├── __init__.py
│   ├── test_api.py              # API contract & health check tests
│   └── test_telemetry.py        # Distribution shift validation tests
├── .dockerignore
├── .gitignore
├── Dockerfile                   # Multi-Stage non-root container manifest
├── requirements.txt             # Production dependency manifest
└── README.md
```

---

<div align="center">
  <sub>Engineered with rigorous software hygiene, reproducible evaluation, and defense-in-depth principles.</sub>
</div>