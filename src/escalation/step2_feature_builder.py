"""
Step 2 – Feature Builder for Reliability Meta-Model

Reads:
    ../reports/tables/escalation_table_testset.csv
(which is produced by: python .\escalation\run_inference_batch.py)

Creates:
    ../reports/tables/step2_meta_dataset.csv

Each row = one patient with:
    - error label: ai_error (1 if AI wrong, 0 if AI correct)
    - key reliability features:
        * risk_1 = 1 - ens_max_prob
        * risk_2 = ens_margin
        * risk_3 = ens_entropy
        * risk_4 = disagreement_score
        * risk_5 = missing_fraction
        * risk_6 = critical_missing (0/1)
        * risk_7 = multimodal_mismatch (0/1)
"""

import os
import pandas as pd
import numpy as np


def build_step2_meta_dataset():
    tables_dir = os.path.join("..", "reports", "tables")
    in_path = os.path.join(tables_dir, "escalation_table_testset.csv")
    out_path = os.path.join(tables_dir, "step2_meta_dataset.csv")

    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Could not find {in_path}. "
            "Run `python .\\escalation\\run_inference_batch.py` first."
        )

    print(f"📂 Loading escalation table from: {in_path}")
    df = pd.read_csv(in_path)

    # basic sanity checks
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

    # 1) AI correct / error
    df["ai_correct"] = (df["true_label"] == df["pred_ens"]).astype(int)
    df["ai_error"] = 1 - df["ai_correct"]  # target for meta-model

    # 2) Build feature columns for meta-model
    #    (you can extend this later with extra signals)
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

    print("\n✅ Built Step 2 meta-dataset:")
    print(f"   Samples: {len(df)}")
    print(f"   Feature columns: {feature_cols}")
    print("   Target: ai_error (1 = AI wrong, 0 = AI correct)")

    df_out = df[
        ["sample_id", "true_label", "pred_ens", "ai_correct", "ai_error"]
        + feature_cols
    ].copy()

    os.makedirs(tables_dir, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\n💾 Saved meta-dataset to: {out_path}")


if __name__ == "__main__":
    build_step2_meta_dataset()
