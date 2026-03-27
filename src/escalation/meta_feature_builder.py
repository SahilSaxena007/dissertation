# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


RISK_FEATURE_MAP = {
    "risk_1_inv_conf": lambda df: 1.0 - df["ens_max_prob"],
    "risk_2_margin": lambda df: df["ens_margin"],
    "risk_3_entropy": lambda df: df["ens_entropy"],
    "risk_4_disagreement": lambda df: df["disagreement_score"],
    "risk_5_missing_fraction": lambda df: df["missing_fraction"],
    "risk_6_critical_missing": lambda df: df["critical_missing"].astype(int),
    "risk_7_multimodal_mismatch": lambda df: df["multimodal_mismatch"].astype(int),
}


def build_meta_dataframe_from_escalation(df: pd.DataFrame) -> pd.DataFrame:
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

    df_local = df.copy()
    df_local["ai_correct"] = (df_local["true_label"] == df_local["pred_ens"]).astype(int)
    df_local["ai_error"] = 1 - df_local["ai_correct"]

    for feature_name, builder in RISK_FEATURE_MAP.items():
        df_local[feature_name] = builder(df_local)

    feature_cols = list(RISK_FEATURE_MAP.keys())
    return df_local[
        ["sample_id", "true_label", "pred_ens", "ai_correct", "ai_error"] + feature_cols
    ].copy()


def build_step2_meta_dataset():
    tables_dir = TABLES_DIR
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
    df_out = build_meta_dataframe_from_escalation(df)
    os.makedirs(tables_dir, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print("Built Step 2 meta-dataset:")
    print(f"   Samples: {len(df_out)}")
    print(f"   Target: ai_error (1=AI wrong, 0=AI correct)")
    print(f"Saved meta-dataset to: {out_path}")
    return df_out, out_path


if __name__ == "__main__":
    build_step2_meta_dataset()
