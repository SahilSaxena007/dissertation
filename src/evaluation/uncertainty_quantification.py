"""
Issue #14: Conformal Prediction + Uncertainty Decomposition

Adaptive Prediction Sets with coverage guarantee.
Epistemic vs aleatoric decomposition.
Saves: reports/tables/conformal_prediction.csv, reports/tables/uncertainty_decomposition.csv

Usage (from src/):
    python evaluation/uncertainty_quantification.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row."""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


def adaptive_prediction_sets(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """
    Conformal prediction using Adaptive Prediction Sets (APS).

    Parameters
    ----------
    cal_probs : (N_cal, C) calibration set probabilities
    cal_labels : (N_cal,) true labels
    test_probs : (N_test, C) test set probabilities
    alpha : desired miscoverage rate (1 - coverage)

    Returns
    -------
    dict with prediction sets, coverage, and average set size
    """
    n_cal = len(cal_labels)
    n_classes = cal_probs.shape[1]

    # Compute nonconformity scores on calibration set
    # Score = 1 - cumulative probability up to and including the true class
    # when classes sorted by descending probability
    scores = np.zeros(n_cal)
    for i in range(n_cal):
        sorted_idx = np.argsort(-cal_probs[i])
        cumsum = 0.0
        for j, cls_idx in enumerate(sorted_idx):
            cumsum += cal_probs[i, cls_idx]
            if cls_idx == cal_labels[i]:
                scores[i] = cumsum
                break

    # Quantile with finite-sample correction
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    qhat = float(np.quantile(scores, q_level))

    # Build prediction sets for test samples
    n_test = test_probs.shape[0]
    set_sizes = np.zeros(n_test, dtype=int)
    prediction_sets = []

    for i in range(n_test):
        sorted_idx = np.argsort(-test_probs[i])
        cumsum = 0.0
        pset = []
        for cls_idx in sorted_idx:
            cumsum += test_probs[i, cls_idx]
            pset.append(int(cls_idx))
            if cumsum >= qhat:
                break
        prediction_sets.append(pset)
        set_sizes[i] = len(pset)

    return {
        "qhat": qhat,
        "set_sizes": set_sizes,
        "prediction_sets": prediction_sets,
        "avg_set_size": float(set_sizes.mean()),
        "singleton_fraction": float((set_sizes == 1).mean()),
    }


def uncertainty_decomposition(
    probs_per_model: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Decompose total uncertainty into epistemic and aleatoric components.

    Total = H(mean_probs)
    Aleatoric = mean(H(probs_per_model))
    Epistemic = Total - Aleatoric
    """
    model_names = list(probs_per_model.keys())
    all_probs = np.stack([probs_per_model[m] for m in model_names], axis=0)  # (M, N, C)
    mean_probs = all_probs.mean(axis=0)  # (N, C)

    total = _entropy(mean_probs)
    aleatoric = np.mean([_entropy(all_probs[m]) for m in range(len(model_names))], axis=0)
    epistemic = total - aleatoric

    return pd.DataFrame({
        "total_uncertainty": total,
        "aleatoric_uncertainty": aleatoric,
        "epistemic_uncertainty": epistemic,
        "epistemic_fraction": epistemic / (total + 1e-12),
    })


def run_uncertainty_quantification():
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Load calibrated OOF probs and per-model probs
    probs_selected = np.load(os.path.join(ARTIFACTS_DIR, "oof_ens_proba_selected.npy"))
    y_oof = np.load(os.path.join(ARTIFACTS_DIR, "y_oof.npy"))
    probs_cat = np.load(os.path.join(ARTIFACTS_DIR, "oof_cat_proba.npy"))
    probs_rf = np.load(os.path.join(ARTIFACTS_DIR, "oof_rf_proba.npy"))
    probs_nn = np.load(os.path.join(ARTIFACTS_DIR, "oof_nn_proba.npy"))

    # Holdout for conformal test set
    X_test = np.load(os.path.join(ARTIFACTS_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))

    # Load models for holdout inference
    import joblib
    import __main__
    import sys
    src_dir = os.path.join(PROJECT_ROOT, "src")
    esc_dir = os.path.join(src_dir, "escalation")
    if esc_dir not in sys.path:
        sys.path.insert(0, esc_dir)
    from model_stub import create_model as _create_model
    __main__.create_model = _create_model

    models = joblib.load(os.path.join(ARTIFACTS_DIR, "voting_ensemble.pkl"))
    if "rf" in models and hasattr(models["rf"], "n_jobs"):
        models["rf"].n_jobs = 1
    test_cat = models["catboost"].predict_proba(X_test)
    test_rf = models["rf"].predict_proba(X_test)
    test_nn = models["nn"].predict_proba(X_test)
    test_ens = (test_cat + test_rf + test_nn) / 3.0

    # Apply calibration if available
    calibrator = joblib.load(os.path.join(ARTIFACTS_DIR, "calibrator_best.pkl"))
    if calibrator is not None:
        test_ens_cal = calibrator.predict_proba(test_ens)
    else:
        test_ens_cal = test_ens

    # 1. Conformal prediction
    alphas = [0.05, 0.10, 0.15, 0.20]
    conf_rows = []
    for alpha in alphas:
        result = adaptive_prediction_sets(
            cal_probs=probs_selected,
            cal_labels=y_oof,
            test_probs=test_ens_cal,
            alpha=alpha,
        )

        # Compute empirical coverage
        coverage = 0.0
        for i in range(len(y_test)):
            if int(y_test[i]) in result["prediction_sets"][i]:
                coverage += 1.0
        coverage /= len(y_test)

        conf_rows.append({
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "empirical_coverage": coverage,
            "avg_set_size": result["avg_set_size"],
            "singleton_fraction": result["singleton_fraction"],
            "qhat": result["qhat"],
        })

    df_conf = pd.DataFrame(conf_rows)
    conf_path = os.path.join(TABLES_DIR, "conformal_prediction.csv")
    df_conf.to_csv(conf_path, index=False)
    print(f"Saved conformal prediction results to: {conf_path}")
    print(df_conf.to_string(index=False))

    # 2. Uncertainty decomposition (on OOF data)
    df_unc = uncertainty_decomposition({
        "catboost": probs_cat,
        "rf": probs_rf,
        "nn": probs_nn,
    })

    # Summary statistics
    summary_rows = []
    for col in ["total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty", "epistemic_fraction"]:
        summary_rows.append({
            "metric": col,
            "mean": float(df_unc[col].mean()),
            "std": float(df_unc[col].std()),
            "median": float(df_unc[col].median()),
            "q25": float(df_unc[col].quantile(0.25)),
            "q75": float(df_unc[col].quantile(0.75)),
        })
    df_unc_summary = pd.DataFrame(summary_rows)
    unc_path = os.path.join(TABLES_DIR, "uncertainty_decomposition.csv")
    df_unc_summary.to_csv(unc_path, index=False)
    print(f"\nSaved uncertainty decomposition to: {unc_path}")
    print(df_unc_summary.to_string(index=False))

    return df_conf, df_unc_summary


if __name__ == "__main__":
    run_uncertainty_quantification()
