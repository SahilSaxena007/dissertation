"""
run_full_analysis.py
---------------------------------------------------------
Unified driver script to:
  1. Run the ModelsFinal pipeline (model training, tuning)
  2. Automatically perform advanced performance analysis
     using performance_analysis.py.

Outputs are saved in ../reports/ (figures + tables)
---------------------------------------------------------
Author: Sahil Saxena
Date: 2025-10-16
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure import paths (add current dir and eval/)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'eval')))

# Import the training + model evaluation pipeline (Benjamin's ModelsFinal)
from ModelsFinal import (
    X_test_scaled, y_test_enc,
    best_model_catboost, rf_model, nn_model,
    cat_proba, rf_proba, nn_proba
)

# Import your dissertation performance analysis extension
from eval.orchestrator import analyze_model_performance


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "reports", f"run_{timestamp}"))
os.makedirs(REPORT_DIR, exist_ok=True)

CLASS_NAMES = ["SCD", "MCI", "AD"]

# ─────────────────────────────────────────────────────────────
# Evaluate All Models
# ─────────────────────────────────────────────────────────────
print("\n🚀 Running Full Dissertation Evaluation Pipeline")

models_for_analysis = {
    "CatBoost": (
        best_model_catboost.predict(X_test_scaled),
        best_model_catboost.predict_proba(X_test_scaled)
    ),
    "RandomForest": (
        rf_model.predict(X_test_scaled),
        rf_model.predict_proba(X_test_scaled)
    ),
    "NeuralNetwork": (
        nn_model.predict(X_test_scaled),
        nn_model.predict_proba(X_test_scaled)
    ),
    "VotingEnsemble": (
        np.argmax((cat_proba + rf_proba + nn_proba) / 3.0, axis=1),
        (cat_proba + rf_proba + nn_proba) / 3.0
    )
}

summaries = []

for model_name, (y_pred, y_prob) in models_for_analysis.items():
    print(f"\n=== Analyzing {model_name} ===")
    summary = analyze_model_performance(
        y_true=y_test_enc,
        y_pred=y_pred,
        y_prob=y_prob,
        model_name=model_name,
        class_names=CLASS_NAMES,
        output_dir=REPORT_DIR
    )
    summaries.append(summary)


# ─────────────────────────────────────────────────────────────
# Save Summary Report
# ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summaries)
summary_path = os.path.join(REPORT_DIR, "tables", "summary_overall.csv")
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
summary_df.to_csv(summary_path, index=False)

print("\n✅ All analyses completed successfully!")
print(f"📁 Reports saved to: {REPORT_DIR}")
