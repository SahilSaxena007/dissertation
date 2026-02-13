"""
Demo: Apply Step 2 meta-model + Step 3 threshold to a single patient.

Usage (from src/):
    python .\escalation\step2_new_patient_meta.py          # default sample_id = 0
    python .\escalation\step2_new_patient_meta.py 42       # specific sample_id

Requires:
    - voting_ensemble.pkl
    - X_test.npy, y_test.npy (strict locked holdout from ModelsFinal.py)
    - step2_meta_model.pkl
    - step3_best_threshold.json
"""

import json
import os
import sys
import joblib
import numpy as np
from pprint import pprint

# for SciKeras unpickling
from model_stub import create_model as _create_model  # noqa: F401

sys.modules["__main__"].create_model = _create_model

from escalation_engine import run_single_escalation
from escalation_config import CLASS_NAMES, ESCALATION_CONFIG


def build_feature_vector_from_row(row: dict) -> np.ndarray:
    """
    Build the same risk feature vector used in training the meta-model.

    Features:
        risk_1_inv_conf           = 1 - ens_max_prob
        risk_2_margin             = ens_margin
        risk_3_entropy            = ens_entropy
        risk_4_disagreement       = disagreement_score
        risk_5_missing_fraction   = missing_fraction
        risk_6_critical_missing   = critical_missing
        risk_7_multimodal_mismatch = multimodal_mismatch
    """
    return np.array(
        [
            1.0 - float(row["ens_max_prob"]),
            float(row["ens_margin"]),
            float(row["ens_entropy"]),
            float(row["disagreement_score"]),
            float(row["missing_fraction"]),
            float(row["critical_missing"]),
            float(row["multimodal_mismatch"]),
        ],
        dtype=float,
    )


def main():
    # choose sample_id
    if len(sys.argv) > 1:
        sample_id = int(sys.argv[1])
    else:
        sample_id = 0

    artifacts_dir = os.path.join("..", "artifacts")

    # 1) Load base ensemble models
    print("📦 Loading base ensemble models (voting_ensemble.pkl) …")
    models = joblib.load(os.path.join(artifacts_dir, "voting_ensemble.pkl"))

    # 2) Load meta-model and best threshold
    meta_model_path = os.path.join(artifacts_dir, "step2_meta_model.pkl")
    if not os.path.exists(meta_model_path):
        raise FileNotFoundError(
            f"{meta_model_path} not found. "
            "Run `python .\\escalation\\step2_train_meta_model.py` first."
        )
    meta_model = joblib.load(meta_model_path)
    print("✅ Loaded Step 2 meta-model.")

    best_thresh_path = os.path.join(artifacts_dir, "step3_best_threshold.json")
    if not os.path.exists(best_thresh_path):
        raise FileNotFoundError(
            f"{best_thresh_path} not found. "
            "Run `python .\\escalation\\run_step3_thresholds.py` first."
        )

    with open(best_thresh_path, "r", encoding="utf-8") as f:
        best_info = json.load(f)
    tau_star = float(best_info["best_threshold"])
    print(f"✅ Loaded Step 3 best threshold τ* = {tau_star:.3f}")

    # 3) Load test data
    print("\n📂 Loading X_test / y_test …")
    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    if sample_id < 0 or sample_id >= len(X_test):
        raise ValueError(f"sample_id {sample_id} out of range (0..{len(X_test)-1})")

    x = X_test[sample_id]
    y_true = int(y_test[sample_id])
    print(f"\n🧪 Evaluating sample_id={sample_id} (true label = {CLASS_NAMES[y_true]})")
    print("   Raw features:", x)

    # 4) Run existing Step 2 escalation engine to compute reliability signals
    row = run_single_escalation(
        x_row=x,
        models=models,
        class_names=CLASS_NAMES,
        config=ESCALATION_CONFIG,
        y_true=y_true,
    )

    print("\n📌 Step 2 (rule-based) result:")
    pprint(row)

    # 5) Build feature vector for meta-model and compute review_risk_score
    feat = build_feature_vector_from_row(row).reshape(1, -1)
    risk_score = float(meta_model.predict_proba(feat)[0, 1])

    # 6) Apply Step 3 policy: compare to τ*
    escalate = risk_score >= tau_star
    policy_decision = "ESCALATE to clinician" if escalate else "AI-AUTONOMOUS"

    # 7) Print combined decision summary
    print("\n🧠 HITL Policy Decision (Step 2 + Step 3):")
    print(f"  • Ensemble prediction : {row['pred_ens_name']} (index {row['pred_ens']})")
    print(f"  • True label          : {row['true_label_name']} (index {row['true_label']})")
    print(f"  • review_risk_score   : {risk_score:.3f}")
    print(f"  • τ* (best threshold) : {tau_star:.3f}")
    print(f"  • Escalate?           : {escalate}")
    print(f"  • Final policy        : {policy_decision}")
    print("\n  (If escalated, clinician reviews the case; otherwise AI decision is used.)")


if __name__ == "__main__":
    main()
