"""
Run Step 2 escalation.

Usage (from src/):
    python .\\escalation\\run_inference_batch.py --source oof
    python .\\escalation\\run_inference_batch.py --source testset
"""

import argparse
import os
from datetime import datetime
import __main__

import joblib
import numpy as np

from model_stub import create_model as _create_model  # noqa: F401
from escalation_engine import run_batch_escalation, run_batch_escalation_from_probabilities
from escalation_config import CLASS_NAMES, build_escalation_config
from shap_utils import load_or_compute_shap

__main__.create_model = _create_model
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


def _load_nan_mask(artifacts_dir: str, source: str):
    """Load pre-imputation NaN mask if available."""
    if source == "oof":
        path = os.path.join(artifacts_dir, "nan_mask_oof.npy")
    else:
        path = os.path.join(artifacts_dir, "nan_mask_holdout.npy")

    if os.path.exists(path):
        mask = np.load(path)
        print(f"   Loaded pre-imputation NaN mask from: {path}")
        return mask
    print(f"   Pre-imputation NaN mask not found at {path}; using post-imputation check.")
    return None


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

    nan_mask = _load_nan_mask(artifacts_dir, "oof")

    # Load models for SHAP computation
    models = joblib.load(os.path.join(artifacts_dir, "voting_ensemble.pkl"))
    if "rf" in models and hasattr(models["rf"], "n_jobs"):
        models["rf"].n_jobs = 1
    shap_values = load_or_compute_shap(models, X, source="oof")

    # Build config with populated multimodal groups
    config = build_escalation_config()

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
        config=config,
        shap_values_by_model=shap_values,
        nan_mask=nan_mask,
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

    # Keep inference single-process for sandboxed environments.
    if "rf" in models and hasattr(models["rf"], "n_jobs"):
        models["rf"].n_jobs = 1

    nan_mask = _load_nan_mask(artifacts_dir, "holdout")

    # Apply calibration to testset ensemble probs (Issue #9)
    calibrator_path = os.path.join(artifacts_dir, "calibrator_best.pkl")
    calibrated_probs = None
    if os.path.exists(calibrator_path):
        calibrator = joblib.load(calibrator_path)
        if calibrator is not None:
            all_probs = [m.predict_proba(X_test) for m in models.values()]
            raw_ens = np.mean(all_probs, axis=0)
            calibrated_probs = calibrator.predict_proba(raw_ens)
            print("   Applied calibration to testset ensemble probabilities.")

    shap_values = load_or_compute_shap(models, X_test, source="holdout")
    config = build_escalation_config()

    if calibrated_probs is not None:
        # Use from_probabilities path with calibrated override
        all_probs = {name: m.predict_proba(X_test) for name, m in models.items()}
        probs_cat = all_probs.get("catboost", list(all_probs.values())[0])
        probs_rf = all_probs.get("rf", list(all_probs.values())[min(1, len(all_probs) - 1)])
        probs_nn = all_probs.get("nn", list(all_probs.values())[min(2, len(all_probs) - 1)])
        df = run_batch_escalation_from_probabilities(
            X=X_test,
            y_true=y_test,
            probs_cat=probs_cat,
            probs_rf=probs_rf,
            probs_nn=probs_nn,
            probs_ens_override=calibrated_probs,
            class_names=CLASS_NAMES,
            config=config,
            shap_values_by_model=shap_values,
            nan_mask=nan_mask,
        )
    else:
        df = run_batch_escalation(
            X=X_test,
            y_true=y_test,
            models=models,
            class_names=CLASS_NAMES,
            config=config,
            shap_values_by_model=shap_values,
            nan_mask=nan_mask,
        )
    return df, "escalation_table_testset.csv"


def run_and_save(source: str = "oof"):
    artifacts_dir = ARTIFACTS_DIR
    out_dir = TABLES_DIR
    os.makedirs(out_dir, exist_ok=True)

    if source == "oof":
        df, filename = run_oof_mode(artifacts_dir)
    elif source == "testset":
        df, filename = run_testset_mode(artifacts_dir)
    else:
        raise ValueError("source must be one of {'oof', 'testset'}")

    out_path = os.path.join(out_dir, filename)
    try:
        df.to_csv(out_path)
    except PermissionError:
        stem, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_filename = f"{stem}_{timestamp}{ext}"
        out_path = os.path.join(out_dir, alt_filename)
        print(f"Permission denied writing {filename}; saving to fallback file: {alt_filename}")
        df.to_csv(out_path)
    return df, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["oof", "testset"],
        default="oof",
        help="Use OOF artifacts (recommended) or legacy test split.",
    )
    args = parser.parse_args()

    df, out_path = run_and_save(source=args.source)

    print(f"\nSaved escalation table to: {out_path}")
    print("\nPreview:")
    print(df.head(10))
    print("\nEscalation counts:")
    print(df["escalation_level"].value_counts())


if __name__ == "__main__":
    main()
