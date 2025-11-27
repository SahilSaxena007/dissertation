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
) -> Dict[str, Any]:
    """
    Evaluate one patient and produce all escalation signals + label.

    Parameters
    ----------
    x_row : (n_features,)
    probs_* : (n_classes,) probabilities from each model
    """
    # ------------------------------------------------------------------
    # 1) Ensemble probabilities (simple mean of the 3 models)
    # ------------------------------------------------------------------
    probs_ens = (probs_cat + probs_rf + probs_nn) / 3.0

    # Predicted labels from each model + ensemble
    pred_cat = int(np.argmax(probs_cat))
    pred_rf = int(np.argmax(probs_rf))
    pred_nn = int(np.argmax(probs_nn))
    pred_ens = int(np.argmax(probs_ens))

    # ------------------------------------------------------------------
    # 2) Compute all escalation signals
    # ------------------------------------------------------------------
    # Uncertainty: max prob, margin, entropy, low_confidence, etc.
    unc = compute_uncertainty_metrics(probs_ens, config)

    # Model disagreement (class-level disagreement score)
    disagreement = compute_model_disagreement(
        {"catboost": pred_cat, "rf": pred_rf, "nn": pred_nn}
    )

    # Missing data flags
    missing = compute_missingness_flags(x_row, config)

    # Multimodal mismatch (biomarkers vs cognitive vs prediction)
    multimodal = compute_multimodal_mismatch(x_row, config)

    # SHAP instability (currently a placeholder)
    shap = compute_shap_instability()

    # ------------------------------------------------------------------
    # 3) Decide escalation level from signals
    # ------------------------------------------------------------------
    reasons: List[str] = []

    # Hard rules first → Clinician-Mandatory
    if missing["critical_missing"]:
        escalation = "Clinician-Mandatory"
        reasons.append("critical_missing_data")
    elif multimodal["mismatch"]:
        escalation = "Clinician-Mandatory"
        reasons.append("multimodal_mismatch")
    else:
        # Softer reasons → AI-Assisted
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

        if reasons:
            escalation = "AI-Assisted"
        else:
            escalation = "AI-Autonomous"

    # ------------------------------------------------------------------
    # 4) Build row dict (everything we want in the CSV / single result)
    # ------------------------------------------------------------------
    row: Dict[str, Any] = {
        "true_label": int(y_true) if y_true is not None else None,
        "true_label_name": (
            class_names[int(y_true)] if y_true is not None else None
        ),
        "pred_cat": pred_cat,
        "pred_rf": pred_rf,
        "pred_nn": pred_nn,
        "pred_ens": pred_ens,
        "pred_ens_name": class_names[pred_ens],

        # Uncertainty
        "ens_max_prob": unc["max_prob"],
        "ens_margin": unc["margin"],
        "ens_entropy": unc["entropy"],
        "low_confidence": unc["low_confidence"],
        "small_margin": unc["small_margin"],
        "high_entropy": unc["high_entropy"],

        # Disagreement
        "model_disagreement": disagreement["disagreement"],
        "disagreement_score": disagreement["disagreement_score"],

        # Missingness
        "missing_fraction": missing["missing_fraction"],
        "has_missing": missing["has_missing"],
        "critical_missing": missing["critical_missing"],

        # Multimodal + SHAP
        "multimodal_mismatch": multimodal["mismatch"],
        "shap_instability": shap["instability"],
        "shap_variance": shap["variance"],

        # Final decision
        "escalation_level": escalation,
        "escalation_reasons": ";".join(reasons) if reasons else "",
    }
    return row


# ---------------------------------------------------------------------------
# Public API – batch + single
# ---------------------------------------------------------------------------

def run_batch_escalation(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    models: Dict[str, Any],
    class_names: List[str] = CLASS_NAMES,
    config: EscalationConfig = ESCALATION_CONFIG,
) -> pd.DataFrame:
    """
    Run Step 2 on a whole batch of patients (e.g., your test set).

    Parameters
    ----------
    X : (n_samples, n_features)
    y_true : (n_samples,) or None
    models : dict with keys {"catboost", "rf", "nn"}
    """
    n_samples = X.shape[0]

    cat = models["catboost"]
    rf = models["rf"]
    nn = models["nn"]

    # NOTE: we ALWAYS use predict_proba for all 3 models,
    # and we always use the SAME X (here: X_test.npy).
    probs_cat = cat.predict_proba(X)
    probs_rf = rf.predict_proba(X)
    probs_nn = nn.predict_proba(X)

    if y_true is None:
        y_arr = np.array([None] * n_samples)
    else:
        y_arr = np.asarray(y_true)

    rows: List[Dict[str, Any]] = []
    for i in range(n_samples):
        row = _evaluate_single_case(
            x_row=X[i],
            y_true=y_arr[i] if y_true is not None else None,
            probs_cat=probs_cat[i],
            probs_rf=probs_rf[i],
            probs_nn=probs_nn[i],
            config=config,
            class_names=class_names,
        )
        row["sample_id"] = int(i)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Keep sample_id as a normal column (not index),
    # so CSV matches exactly and is easier to inspect.
    df = df.set_index("sample_id")
    return df


def run_single_escalation(
    x_row: np.ndarray,
    models: Dict[str, Any],
    class_names: List[str] = CLASS_NAMES,
    config: EscalationConfig = ESCALATION_CONFIG,
    y_true: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for a single NEW patient.

    x_row : (n_features,) or (1, n_features)
    """
    x_row = np.asarray(x_row, dtype=float)
    if x_row.ndim == 2:
        x_row = x_row[0]

    cat = models["catboost"]
    rf = models["rf"]
    nn = models["nn"]

    # IMPORTANT: use predict_proba for ALL three models,
    # exactly like in run_batch_escalation.
    # create a fake batch of size N=1
    X_one = x_row.reshape(1, -1)

    probs_cat = models["catboost"].predict_proba(X_one)[0]
    probs_rf  = models["rf"].predict_proba(X_one)[0]

    # IMPORTANT: do NOT call nn alone
    # instead force batch-mode by using underlying Keras model directly
    keras_model = models["nn"].model_

    probs_nn = keras_model.predict(X_one, verbose=0)[0]


    result = _evaluate_single_case(
        x_row=x_row,
        y_true=y_true,
        probs_cat=probs_cat,
        probs_rf=probs_rf,
        probs_nn=probs_nn,
        config=config,
        class_names=class_names,
    )
    return result
