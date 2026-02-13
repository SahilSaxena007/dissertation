"""
Step 2 - Feature Builder for Reliability Meta-Model

Reads (preferred):
    ../reports/tables/escalation_table_oof.csv
Fallback:
    ../reports/tables/escalation_table_testset.csv

Creates:
    ../reports/tables/step2_meta_dataset.csv
"""

import os
import pandas as pd


def build_step2_meta_dataset():
    tables_dir = os.path.join("..", "reports", "tables")
    preferred_in_path = os.path.join(tables_dir, "escalation_table_oof.csv")
    fallback_in_path = os.path.join(tables_dir, "escalation_table_testset.csv")
    out_path = os.path.join(tables_dir, "step2_meta_dataset.csv")

    if os.path.exists(preferred_in_path):
        in_path = preferred_in_path
    elif os.path.exists(fallback_in_path):
        in_path = fallback_in_path
    else:
        raise FileNotFoundError(
            "Could not find escalation input table. "
            "Run `python .\\escalation\\run_inference_batch.py --source oof` first."
        )

    print(f"Loading escalation table from: {in_path}")
    df = pd.read_csv(in_path)

    required_cols = [
        "sample_id",
        "true_label",
        "pred_ens",
        "ens_max_prob",
        "ens_margin",
        "ens_entropy",
        "disagreement_score",
        "missing_fraction",
        "critical_missing",
        "multimodal_mismatch",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in escalation table: {missing}")

    df["ai_correct"] = (df["true_label"] == df["pred_ens"]).astype(int)
    df["ai_error"] = 1 - df["ai_correct"]

    df["risk_1_inv_conf"] = 1.0 - df["ens_max_prob"]
    df["risk_2_margin"] = df["ens_margin"]
    df["risk_3_entropy"] = df["ens_entropy"]
    df["risk_4_disagreement"] = df["disagreement_score"]
    df["risk_5_missing_fraction"] = df["missing_fraction"]
    df["risk_6_critical_missing"] = df["critical_missing"].astype(int)
    df["risk_7_multimodal_mismatch"] = df["multimodal_mismatch"].astype(int)

    feature_cols = [
        "risk_1_inv_conf",
        "risk_2_margin",
        "risk_3_entropy",
        "risk_4_disagreement",
        "risk_5_missing_fraction",
        "risk_6_critical_missing",
        "risk_7_multimodal_mismatch",
    ]

    df_out = df[["sample_id", "true_label", "pred_ens", "ai_correct", "ai_error"] + feature_cols].copy()
    os.makedirs(tables_dir, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print("Built Step 2 meta-dataset:")
    print(f"   Samples: {len(df_out)}")
    print(f"   Target: ai_error (1=AI wrong, 0=AI correct)")
    print(f"Saved meta-dataset to: {out_path}")


if __name__ == "__main__":
    build_step2_meta_dataset()
