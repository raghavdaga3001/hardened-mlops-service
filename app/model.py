import os
import torch
import torch.nn as nn
from safetensors.torch import load_file

class RiskClassifier(nn.Module):
    """6-feature risk classification neural network."""
    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Logits: [0: legitimate, 1: suspicious]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class InferenceEngine:
    def __init__(self, weights_path: str = "baseline/model.safetensors"):
        self.weights_path = weights_path
        self.model: RiskClassifier | None = None
        self.device = torch.device("cpu")

    def load(self) -> None:
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.weights_path}")
        
        # Instantiate architecture
        self.model = RiskClassifier(input_dim=6)
        
        # Load weights strictly using safetensors (zero pickle execution risk)
        state_dict = load_file(self.weights_path, device="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, features: list[float]) -> dict:
        if self.model is None:
            raise RuntimeError("Inference engine invoked before weights were loaded.")
        
        tensor_input = torch.tensor([features], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=-1)
            pred_class = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, pred_class].item())

        return {
            "prediction": pred_class,
            "label": "suspicious" if pred_class == 1 else "legitimate",
            "confidence": round(confidence, 4),
            "risk_probability": round(float(probs[0, 1].item()), 4)
        }