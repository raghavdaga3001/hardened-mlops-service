import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from safetensors.torch import save_file

# --- 1. Define Architecture ---
class RiskClassifier(nn.Module):
    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Binary classification logits [legit, suspicious]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def main():
    print("[*] Generating synthetic high-assurance tabular dataset...")
    # 6 continuous numerical features
    feature_names = [
        "transaction_amount",
        "velocity_score",
        "account_age_months",
        "auth_attempts",
        "device_trust_score",
        "risk_score"
    ]
    
    X, y = make_classification(
        n_samples=5000,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 2. Train PyTorch Model ---
    print("[*] Training PyTorch baseline classifier...")
    model = RiskClassifier(input_dim=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for epoch in range(25):
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    print(f"    Final Training Loss: {loss.item():.4f}")

    # --- 3. Save Baseline Data as Parquet ---
    # Parquet preserves columnar types and serves as the reference distribution for Evidently AI
    os.makedirs("baseline", exist_ok=True)
    baseline_df = pd.DataFrame(X_train, columns=feature_names)
    baseline_df["target"] = y_train
    baseline_path = "baseline/baseline_data.parquet"
    baseline_df.to_parquet(baseline_path, index=False)
    print(f"[✓] Baseline reference distribution saved to: {baseline_path}")

    # --- 4. Export Model Weights in .safetensors Format ---
    # Bypasses pickle.__reduce__() arbitrary execution risks completely
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    weights_path = "baseline/model.safetensors"
    save_file(tensors, weights_path)
    print(f"[✓] Hardened model weights saved to: {weights_path}")

if __name__ == "__main__":
    main()