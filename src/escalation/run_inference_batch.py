"""
Run Step 2 escalation.

Usage (from src/):
    python .\\\\escalation\\\\run_inference_batch.py --source oof
    python .\\\\escalation\\\\run_inference_batch.py --source testset
"""

import argparse
import os
import __main__

import joblib
import numpy as np

from model_stub import create_model as _create_model  # noqa: F401
from escalation_engine import run_batch_escalation, run_batch_escalation_from_probabilities
from escalation_config import CLASS_NAMES, ESCALATION_CONFIG

__main__.create_model = _create_model


def run_oof_mode(artifacts_dir: str):
    print("Loading OOF artifacts...")
    X = np.load(os.path.join(artifacts_dir, "X_oof.npy"))
    y_true = np.load(os.path.join(artifacts_dir, "y_oof.npy"))
    probs_cat = np.load(os.path.join(artifacts_dir, "oof_cat_proba.npy"))
    probs_rf = np.load(os.path.join(artifacts_dir, "oof_rf_proba.npy"))
    probs_nn = np.load(os.path.join(artifacts_dir, "oof_nn_proba.npy"))
    probs_ens_selected_path = os.path.join(artifacts_dir, "oof_ens_proba_selected.npy")
    probs_ens_calibrated_path = os.path.join(artifacts_dir, "oof_ens_proba_calibrated.npy")
    probs_ens_selected = None
    if os.path.exists(probs_ens_selected_path):
        probs_ens_selected = np.load(probs_ens_selected_path)
        print("   Using selected OOF ensemble probabilities for uncertainty signals.")
    elif os.path.exists(probs_ens_calibrated_path):
        probs_ens_selected = np.load(probs_ens_calibrated_path)
        print("   Using calibrated OOF ensemble probabilities for uncertainty signals.")
    else:
        print("   Selected OOF ensemble probabilities not found; using raw ensemble average.")

    print(f"   X_oof shape: {X.shape}")
    print(f"   y_oof shape: {y_true.shape}")

    df = run_batch_escalation_from_probabilities(
        X=X,
        y_true=y_true,
        probs_cat=probs_cat,
        probs_rf=probs_rf,
        probs_nn=probs_nn,
        probs_ens_override=probs_ens_selected,
        class_names=CLASS_NAMES,
        config=ESCALATION_CONFIG,
    )
    return df, "escalation_table_oof.csv"


def run_testset_mode(artifacts_dir: str):
    print("Loading ensemble models from voting_ensemble.pkl...")
    models = joblib.load(os.path.join(artifacts_dir, "voting_ensemble.pkl"))

    print("Loading legacy X_test / y_test...")
    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape}")

    df = run_batch_escalation(
        X=X_test,
        y_true=y_test,
        models=models,
        class_names=CLASS_NAMES,
        config=ESCALATION_CONFIG,
    )
    return df, "escalation_table_testset.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["oof", "testset"],
        default="oof",
        help="Use OOF artifacts (recommended) or legacy test split.",
    )
    args = parser.parse_args()

    artifacts_dir = os.path.join("..", "artifacts")
    out_dir = os.path.join("..", "reports", "tables")
    os.makedirs(out_dir, exist_ok=True)

    if args.source == "oof":
        df, filename = run_oof_mode(artifacts_dir)
    else:
        df, filename = run_testset_mode(artifacts_dir)

    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path)

    print(f"\nSaved escalation table to: {out_path}")
    print("\nPreview:")
    print(df.head(10))
    print("\nEscalation counts:")
    print(df["escalation_level"].value_counts())


if __name__ == "__main__":
    main()
