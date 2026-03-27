# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Run full Phase 2 HITL pipeline end-to-end.

Usage (from src/):
    python .\\escalation\\run_phase2_hitl.py
    python .\\escalation\\run_phase2_hitl.py --review-budget 0.2 --run-testset-demo
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np

from inference_batch import run_oof_mode, run_testset_mode
from threshold_runner import build_human_accuracy_grid
from meta_feature_builder import build_meta_dataframe_from_escalation
from meta_model_trainer import train_meta_model
from threshold_optimizer import optimize_thresholds_with_sensitivity

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


def _safe_save_csv(df, out_path: str) -> str:
    try:
        df.to_csv(out_path, index=False)
        return out_path
    except PermissionError:
        base_dir, filename = os.path.split(out_path)
        stem, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = os.path.join(base_dir, f"{stem}_{timestamp}{ext}")
        print(f"Permission denied for {filename}; saving to fallback: {os.path.basename(alt_path)}")
        df.to_csv(alt_path, index=False)
        return alt_path


def _evaluate_testset_policy(tau_star: float, human_accuracy_assumed: float) -> dict:
    artifacts_dir = ARTIFACTS_DIR
    tables_dir = TABLES_DIR
    os.makedirs(tables_dir, exist_ok=True)

    df_test, filename = run_testset_mode(artifacts_dir)
    out_path = _safe_save_csv(df_test.reset_index(), os.path.join(tables_dir, filename))
    df_test_reset = df_test.reset_index()
    df_meta = build_meta_dataframe_from_escalation(df_test_reset)

    meta_model_path = os.path.join(ARTIFACTS_DIR, "step2_meta_model.pkl")
    if not os.path.exists(meta_model_path):
        raise FileNotFoundError(
            f"Could not find {meta_model_path}. "
            "Run Step 2 meta-model training before testset policy evaluation."
        )

    model = joblib.load(meta_model_path)
    risk_feature_cols = [c for c in df_meta.columns if c.startswith("risk_")]
    risk_score = model.predict_proba(df_meta[risk_feature_cols].values.astype(float))[:, 1]

    escalate = risk_score >= float(tau_star)
    ai_correct = df_meta["ai_correct"].values.astype(float)
    expected_final_correct = np.where(escalate, float(human_accuracy_assumed), ai_correct)

    result = {
        "testset_escalation_table_path": out_path,
        "testset_size": int(df_meta.shape[0]),
        "testset_ai_only_accuracy": float(ai_correct.mean()),
        "testset_review_rate": float(escalate.mean()),
        "testset_expected_policy_accuracy": float(expected_final_correct.mean()),
        "testset_expected_accuracy_gain": float(expected_final_correct.mean() - ai_correct.mean()),
    }
    return result


def run_phase2_hitl(
    review_budget: float = 0.40,
    human_accuracy_min: float = 0.85,
    human_accuracy_max: float = 0.95,
    human_accuracy_step: float = 0.01,
    run_testset_demo: bool = False,
) -> dict:
    if human_accuracy_step <= 0:
        raise ValueError("human_accuracy_step must be > 0.")
    if human_accuracy_min > human_accuracy_max:
        raise ValueError("human_accuracy_min must be <= human_accuracy_max.")

    print("Phase 2 pipeline: Step 1/4 - Build OOF escalation table")
    artifacts_dir = ARTIFACTS_DIR
    tables_dir = TABLES_DIR
    os.makedirs(tables_dir, exist_ok=True)

    df_oof, oof_filename = run_oof_mode(artifacts_dir)
    oof_escalation_path = _safe_save_csv(df_oof.reset_index(), os.path.join(tables_dir, oof_filename))

    print("Phase 2 pipeline: Step 2/4 - Build Step 2 meta dataset")
    df_meta_oof = build_meta_dataframe_from_escalation(df_oof.reset_index())
    meta_dataset_path = _safe_save_csv(df_meta_oof, os.path.join(tables_dir, "step2_meta_dataset.csv"))
    print(f"Saved meta-dataset to: {meta_dataset_path}")

    print("Phase 2 pipeline: Step 3/4 - Train Step 2 reliability meta-model")
    step2_info = train_meta_model(meta_path=meta_dataset_path)

    print("Phase 2 pipeline: Step 4/4 - Optimize Step 3 threshold with sensitivity analysis")
    human_accuracy_values = build_human_accuracy_grid(
        min_value=human_accuracy_min,
        max_value=human_accuracy_max,
        step=human_accuracy_step,
    )
    best = optimize_thresholds_with_sensitivity(
        review_budget=review_budget,
        human_accuracy_values=human_accuracy_values,
        num_thresholds=201,
    )

    summary = {
        "phase2_mode": "oof_train_testset_demo_optional",
        "dataset_used_for_phase2_training": "OOF train-split artifacts",
        "dataset_rationale": (
            "OOF predictions are leak-free for each sample, supporting reliable Step 2 risk-model "
            "training and Step 3 threshold optimization."
        ),
        "oof_escalation_table_path": oof_escalation_path,
        "meta_dataset_path": meta_dataset_path,
        "step2": step2_info,
        "step3": best,
        "run_testset_demo": bool(run_testset_demo),
    }

    if run_testset_demo:
        summary["testset_demo"] = _evaluate_testset_policy(
            tau_star=float(best["best_threshold"]),
            human_accuracy_assumed=float(best["human_accuracy_assumed"]),
        )

    artifacts_dir = ARTIFACTS_DIR
    os.makedirs(artifacts_dir, exist_ok=True)
    summary_path = os.path.join(artifacts_dir, "phase2_hitl_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPhase 2 completed.")
    print(f"Summary saved to: {summary_path}")
    print(
        "Chosen tau*: "
        f"{summary['step3']['best_threshold']:.3f}, "
        f"review_rate={summary['step3']['review_rate']:.3f}, "
        f"policy_accuracy={summary['step3']['policy_accuracy']:.3f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-budget", type=float, default=0.08, help="Max fraction for clinician review.")
    parser.add_argument(
        "--human-accuracy-min",
        type=float,
        default=0.85,
        help="Lower bound of human accuracy sensitivity range.",
    )
    parser.add_argument(
        "--human-accuracy-max",
        type=float,
        default=0.95,
        help="Upper bound of human accuracy sensitivity range.",
    )
    parser.add_argument(
        "--human-accuracy-step",
        type=float,
        default=0.01,
        help="Step size of human accuracy sensitivity grid.",
    )
    parser.add_argument(
        "--run-testset-demo",
        action="store_true",
        help="Also run unseen holdout demo after OOF-based policy training.",
    )
    args = parser.parse_args()

    run_phase2_hitl(
        review_budget=args.review_budget,
        human_accuracy_min=args.human_accuracy_min,
        human_accuracy_max=args.human_accuracy_max,
        human_accuracy_step=args.human_accuracy_step,
        run_testset_demo=args.run_testset_demo,
    )


if __name__ == "__main__":
    main()
