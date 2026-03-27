"""
Lightweight smoke test for Step 2.

Usage (from src/):
    python .\escalation\test_step2_escalation.py
"""

import sys
import os
import joblib
import numpy as np

# stub for joblib
from .model_stub import create_model as _create_model  # noqa: F401

sys.modules["__main__"].create_model = _create_model

from engine import run_batch_escalation, run_single_escalation
from config import CLASS_NAMES, ESCALATION_CONFIG


def main():
    artifacts_dir = os.path.join("..", "artifacts")

    print("Loading models...")
    models = joblib.load(os.path.join(artifacts_dir, "voting_ensemble.pkl"))

    print("Loading X_test / y_test...")
    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    print("Running batch escalation on first 20 samples...")
    df = run_batch_escalation(
        X=X_test[:20],
        y_true=y_test[:20],
        models=models,
        class_names=CLASS_NAMES,
        config=ESCALATION_CONFIG,
    )
    print(df[["true_label_name", "pred_ens_name", "escalation_level"]])

    print("\nEscalation distribution (first 20):")
    print(df["escalation_level"].value_counts())

    print("\nTesting single-case inference for sample_id=0...")
    result = run_single_escalation(
        x_row=X_test[0],
        models=models,
        class_names=CLASS_NAMES,
        config=ESCALATION_CONFIG,
        y_true=int(y_test[0]),
    )
    print(result)


if __name__ == "__main__":
    main()
