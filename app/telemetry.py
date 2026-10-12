import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

FEATURE_COLUMNS = [
    "transaction_amount",
    "velocity_score",
    "account_age_months",
    "auth_attempts",
    "device_trust_score",
    "risk_score"
]

class DriftMonitor:
    def __init__(self, baseline_path: str = "baseline/baseline_data.parquet"):
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Baseline file missing at: {baseline_path}")
        self.baseline_df = pd.read_parquet(baseline_path)[FEATURE_COLUMNS]

    def evaluate_batch(self, current_records: list[dict], html_report_path: str | None = "baseline/drift_report.html") -> dict:
        current_df = pd.DataFrame(current_records)[FEATURE_COLUMNS]
        
        # 1. Direct statistical computation (KS-test and Wasserstein distance)
        metric_results = {}
        drift_detected = False

        for col in FEATURE_COLUMNS:
            ref_col = self.baseline_df[col].to_numpy(dtype=float)
            curr_col = current_df[col].to_numpy(dtype=float)

            # Two-sample KS test (null hypothesis: distributions are identical)
            ks_stat, p_value = ks_2samp(ref_col, curr_col)
            # Wasserstein / Earth Mover's Distance
            w_dist = float(wasserstein_distance(ref_col, curr_col))
            
            # Reject null hypothesis if p-value < 0.05
            is_col_drifted = bool(p_value < 0.05)
            if is_col_drifted:
                drift_detected = True

            metric_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "wasserstein_distance": round(w_dist, 4),
                "drift_flag": is_col_drifted
            }

        # 2. Evidently AI Dashboard Export (Isolated with fallback)
        report_saved = False
        if html_report_path:
            try:
                from evidently import Report
                from evidently.presets import DataDriftPreset
                
                os.makedirs(os.path.dirname(html_report_path), exist_ok=True)
                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=self.baseline_df, current_data=current_df)
                report.save_html(html_report_path)
                report_saved = True
            except Exception as exc:
                print(f"[!] Evidently visual report generation deferred: {exc}")

        return {
            "summary": {
                "batch_size": int(len(current_records)),
                "drift_detected": drift_detected,
                "report_generated": report_saved
            },
            "features": metric_results
        }