"""
Step 3 – Threshold Optimisation for HITL Policy

Reads:
    ../reports/tables/threshold_data.csv
(with columns: sample_id, true_label, pred_ens, ai_correct, review_risk_score, ...)

Computes:
    For many thresholds τ:
        - review_rate(τ)
        - policy_accuracy(τ)
        - ai_only_accuracy (baseline)
        - accuracy_gain(τ)

Saves:
    ../reports/tables/threshold_analysis.csv
    ../artifacts/step3_best_threshold.json
"""

import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd


def optimize_thresholds(
    review_budget: float = 0.4,
    human_accuracy: float = 0.95,
    num_thresholds: int = 201,
) -> Dict[str, Any]:
    """
    Optimise the HITL threshold τ on the meta-model's review_risk_score.

    Parameters
    ----------
    review_budget : float
        Max fraction of cases allowed to be escalated to human review
        (e.g. 0.2 = 20% of cases can be reviewed).
    human_accuracy : float
        Assumed average accuracy of clinicians on escalated cases.
        Must be in (0, 1]. Using 1.0 assumes perfect humans; 0.9–0.99
        is more realistic.
    num_thresholds : int
        Number of evenly spaced τ values in [0, 1] to evaluate.

    Returns
    -------
    Dict[str, Any]
        Information about the best threshold under the given budget.
    """
    if not (0.0 < human_accuracy <= 1.0):
        raise ValueError("human_accuracy must be in the interval (0, 1].")

    tables_dir = os.path.join("..", "reports", "tables")
    artifacts_dir = os.path.join("..", "artifacts")

    thresh_path = os.path.join(tables_dir, "threshold_data.csv")
    if not os.path.exists(thresh_path):
        raise FileNotFoundError(
            f"Could not find {thresh_path}. "
            "Run `python .\\escalation\\step2_train_meta_model.py` first."
        )

    print(f"📂 Loading threshold dataset from: {thresh_path}")
    df = pd.read_csv(thresh_path)

    if "review_risk_score" not in df.columns or "ai_correct" not in df.columns:
        raise ValueError(
            "threshold_data.csv must contain 'review_risk_score' and 'ai_correct' columns."
        )

    risk = df["review_risk_score"].values.astype(float)
    ai_correct = df["ai_correct"].values.astype(int)

    # Baseline AI-only accuracy (no human review anywhere)
    ai_only_acc = ai_correct.mean()
    print(f"\n🎯 Baseline AI-only accuracy: {ai_only_acc:.3f}")

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    rows = []

    for tau in thresholds:
        # Escalation decision: True = send to clinician
        escalate = risk >= tau
        review_rate = escalate.mean()

        # Expected correctness under the HITL policy:
        # - Non-escalated cases: correctness = ai_correct (0 or 1)
        # - Escalated cases: expected correctness = human_accuracy (e.g. 0.95)
        # This avoids assuming humans are perfect (1.0).
        final_correct = np.where(escalate, human_accuracy, ai_correct.astype(float))

        policy_acc = final_correct.mean()
        accuracy_gain = policy_acc - ai_only_acc

        rows.append(
            {
                "threshold": tau,
                "review_rate": review_rate,
                "policy_accuracy": policy_acc,
                "ai_only_accuracy": ai_only_acc,
                "accuracy_gain": accuracy_gain,
                "human_accuracy_assumed": human_accuracy,
            }
        )

    df_tau = pd.DataFrame(rows)

    # Save full curve for plots / analysis
    out_analysis = os.path.join(tables_dir, "threshold_analysis.csv")
    df_tau.to_csv(out_analysis, index=False)
    print(f"\n💾 Saved threshold analysis to: {out_analysis}")

    # Budget-constrained optimum (Method A)
    feasible = df_tau[df_tau["review_rate"] <= review_budget]
    if feasible.empty:
        # If nothing fits the budget, just pick the globally best policy
        print(
            "\n⚠️ No threshold satisfies the review budget; "
            "using global best policy instead."
        )
        best_row = df_tau.loc[df_tau["policy_accuracy"].idxmax()]
    else:
        best_row = feasible.loc[feasible["policy_accuracy"].idxmax()]

    best_info = {
        "best_threshold": float(best_row["threshold"]),
        "review_budget": float(review_budget),
        "review_rate": float(best_row["review_rate"]),
        "policy_accuracy": float(best_row["policy_accuracy"]),
        "ai_only_accuracy": float(best_row["ai_only_accuracy"]),
        "accuracy_gain": float(best_row["accuracy_gain"]),
        "human_accuracy_assumed": float(best_row["human_accuracy_assumed"]),
    }

    print("\n🏆 Best policy under review budget:")
    for k, v in best_info.items():
        print(f"   {k}: {v}")

    # Save JSON with chosen τ* and assumptions
    os.makedirs(artifacts_dir, exist_ok=True)
    best_path = os.path.join(artifacts_dir, "step3_best_threshold.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    print(f"\n💾 Saved best threshold info to: {best_path}")

    return best_info


if __name__ == "__main__":
    optimize_thresholds()
