"""
High-level Step 2 engine.

Takes:
  - models (catboost, rf, nn)
  - X (test data or single row)
  - optional y_true
and returns:
  - per-patient escalation labels + reasons
  - numeric signals used by the rules
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from escalation_config import ESCALATION_CONFIG, CLASS_NAMES, EscalationConfig
from escalation_rules import (
    compute_uncertainty_metrics,
    compute_model_disagreement,
    compute_missingness_flags,
    compute_multimodal_mismatch,
    compute_shap_instability,
)


# ---------------------------------------------------------------------------
# Core helper for a SINGLE patient
# ---------------------------------------------------------------------------

def _evaluate_single_case(
    x_row: np.ndarray,
    y_true: Optional[int],
    probs_cat: np.ndarray,
    probs_rf: np.ndarray,
    probs_nn: np.ndarray,
    config: EscalationConfig,
    class_names: List[str] = CLASS_NAMES,
    shap_by_model: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    # Ensemble probabilities
    probs_ens = (probs_cat + probs_rf + probs_nn) / 3.0

    pred_cat = int(np.argmax(probs_cat))
    pred_rf = int(np.argmax(probs_rf))
    pred_nn = int(np.argmax(probs_nn))
    pred_ens = int(np.argmax(probs_ens))

    unc = compute_uncertainty_metrics(probs_ens, config)
    disagreement = compute_model_disagreement({"catboost": pred_cat, "rf": pred_rf, "nn": pred_nn})
    missing = compute_missingness_flags(x_row, config)
    multimodal = compute_multimodal_mismatch(x_row, config)
    shap = compute_shap_instability(shap_by_model=shap_by_model)

    reasons: List[str] = []

    if missing["critical_missing"]:
        escalation = "Clinician-Mandatory"
        reasons.append("critical_missing_data")
    elif multimodal["mismatch"]:
        escalation = "Clinician-Mandatory"
        reasons.append("multimodal_mismatch")
    else:
        if disagreement["disagreement"]:
            reasons.append("model_disagreement")
        if unc["low_confidence"]:
            reasons.append("low_confidence")
        if unc["small_margin"]:
            reasons.append("small_margin")
        if unc["high_entropy"]:
            reasons.append("high_entropy")
        if shap["instability"]:
            reasons.append("shap_instability")

        escalation = "AI-Assisted" if reasons else "AI-Autonomous"

    row: Dict[str, Any] = {
        "true_label": int(y_true) if y_true is not None else None,
        "true_label_name": class_names[int(y_true)] if y_true is not None else None,
        "pred_cat": pred_cat,
        "pred_rf": pred_rf,
        "pred_nn": pred_nn,
        "pred_ens": pred_ens,
        "pred_ens_name": class_names[pred_ens],
        "ens_max_prob": unc["max_prob"],
        "ens_margin": unc["margin"],
        "ens_entropy": unc["entropy"],
        "low_confidence": unc["low_confidence"],
        "small_margin": unc["small_margin"],
        "high_entropy": unc["high_entropy"],
        "model_disagreement": disagreement["disagreement"],
        "disagreement_score": disagreement["disagreement_score"],
        "missing_fraction": missing["missing_fraction"],
        "has_missing": missing["has_missing"],
        "critical_missing": missing["critical_missing"],
        "multimodal_mismatch": multimodal["mismatch"],
        "shap_instability": shap["instability"],
        "shap_variance": shap["variance"],
        "escalation_level": escalation,
        "escalation_reasons": ";".join(reasons) if reasons else "",
    }
    return row


def run_batch_escalation(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    models: Dict[str, Any],
    class_names: List[str] = CLASS_NAMES,
    config: EscalationConfig = ESCALATION_CONFIG,
    shap_values_by_model: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    n_samples = X.shape[0]

    cat = models["catboost"]
    rf = models["rf"]
    nn = models["nn"]

    probs_cat = cat.predict_proba(X)
    probs_rf = rf.predict_proba(X)
    probs_nn = nn.predict_proba(X)

    return run_batch_escalation_from_probabilities(
        X=X,
        y_true=y_true,
        probs_cat=probs_cat,
        probs_rf=probs_rf,
        probs_nn=probs_nn,
        class_names=class_names,
        config=config,
        shap_values_by_model=shap_values_by_model,
    )


def run_batch_escalation_from_probabilities(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    probs_cat: np.ndarray,
    probs_rf: np.ndarray,
    probs_nn: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    config: EscalationConfig = ESCALATION_CONFIG,
    shap_values_by_model: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    n_samples = X.shape[0]
    if y_true is None:
        y_arr = np.array([None] * n_samples)
    else:
        y_arr = np.asarray(y_true)

    rows: List[Dict[str, Any]] = []
    for i in range(n_samples):
        shap_row = None
        if shap_values_by_model:
            shap_row = {}
            for model_name, shap_values in shap_values_by_model.items():
                arr = np.asarray(shap_values)
                if arr.ndim >= 2 and arr.shape[0] > i:
                    shap_row[model_name] = arr[i]

        row = _evaluate_single_case(
            x_row=X[i],
            y_true=y_arr[i] if y_true is not None else None,
            probs_cat=probs_cat[i],
            probs_rf=probs_rf[i],
            probs_nn=probs_nn[i],
            config=config,
            class_names=class_names,
            shap_by_model=shap_row,
        )
        row["sample_id"] = int(i)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.set_index("sample_id")


def run_single_escalation(
    x_row: np.ndarray,
    models: Dict[str, Any],
    class_names: List[str] = CLASS_NAMES,
    config: EscalationConfig = ESCALATION_CONFIG,
    y_true: Optional[int] = None,
) -> Dict[str, Any]:
    x_row = np.asarray(x_row, dtype=float)
    if x_row.ndim == 2:
        x_row = x_row[0]

    X_one = x_row.reshape(1, -1)
    probs_cat = models["catboost"].predict_proba(X_one)[0]
    probs_rf = models["rf"].predict_proba(X_one)[0]
    probs_nn = models["nn"].model_.predict(X_one, verbose=0)[0]

    return _evaluate_single_case(
        x_row=x_row,
        y_true=y_true,
        probs_cat=probs_cat,
        probs_rf=probs_rf,
        probs_nn=probs_nn,
        config=config,
        class_names=class_names,
    )
