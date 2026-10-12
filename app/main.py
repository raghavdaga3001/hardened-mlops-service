from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.model import InferenceEngine
from app.telemetry import DriftMonitor

# Global lifecycle state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("[*] Starting up: Loading model weights and baseline data...")
    
    # 1. Initialize PyTorch Inference Engine via safetensors
    engine = InferenceEngine(weights_path="baseline/model.safetensors")
    engine.load()
    app_state["engine"] = engine

    # 2. Initialize Drift Monitor with reference Parquet baseline
    monitor = DriftMonitor(baseline_path="baseline/baseline_data.parquet")
    app_state["monitor"] = monitor

    app_state["ready"] = True
    print("[✓] Service fully initialized and ready.")
    
    yield
    
    print("[*] Tearing down service...")
    app_state.clear()

app = FastAPI(
    title="Hardened AI Inference Service",
    version="1.0.0",
    description="Secure model serving pipeline with automated drift telemetry.",
    lifespan=lifespan
)

# Schema definitions
class RiskPredictionRequest(BaseModel):
    transaction_amount: float = Field(..., description="Normalized transaction amount")
    velocity_score: float = Field(..., description="Transaction frequency/velocity metric")
    account_age_months: float = Field(..., ge=0, description="Age of account in months")
    auth_attempts: float = Field(..., ge=0, description="Recent authentication attempts")
    device_trust_score: float = Field(..., description="Device reputation score")
    risk_score: float = Field(..., description="Pre-computed identity risk metric")

class RiskPredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    risk_probability: float

# Health endpoints
@app.get("/health/liveness", status_code=status.HTTP_200_OK)
def liveness() -> dict:
    return {"status": "alive"}

@app.get("/health/readiness", status_code=status.HTTP_200_OK)
def readiness() -> dict:
    if not app_state.get("ready"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model weights or telemetry baseline loading"
        )
    return {"status": "ready"}

# Inference route
@app.post("/predict", response_model=RiskPredictionResponse, status_code=status.HTTP_200_OK)
def predict(payload: RiskPredictionRequest) -> dict:
    engine: InferenceEngine = app_state.get("engine")
    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine uninitialized")
    
    features = [
        payload.transaction_amount,
        payload.velocity_score,
        payload.account_age_months,
        payload.auth_attempts,
        payload.device_trust_score,
        payload.risk_score
    ]
    return engine.predict(features)

# Drift telemetry route
@app.post("/telemetry/drift-report", status_code=status.HTTP_200_OK)
def run_drift_analysis(batch: list[RiskPredictionRequest]) -> dict:
    monitor: DriftMonitor = app_state.get("monitor")
    if not monitor:
        raise HTTPException(status_code=503, detail="Telemetry monitor uninitialized")
    if len(batch) < 15:
        raise HTTPException(status_code=400, detail="Minimum batch size of 15 records required for distribution tests")

    payloads = [item.model_dump() for item in batch]
    return monitor.evaluate_batch(payloads)