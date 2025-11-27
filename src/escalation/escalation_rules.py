"""
Low-level rule functions for Step 2.

Each function:
  - takes simple inputs (probabilities, predictions, feature row)
  - returns numeric metrics + boolean flags
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from escalation_config import EscalationConfig


# ---------------------------------------------------------------------------
# 1. Uncertainty metrics (max prob, margin, entropy)
# ---------------------------------------------------------------------------

def compute_uncertainty_metrics(
    probs: np.ndarray, config: EscalationConfig
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    probs : (n_classes,) array of ensemble probabilities.
    config : EscalationConfig

    Returns
    -------
    dict with:
        max_prob, margin, entropy,
        low_confidence, small_margin, high_entropy
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError(f"Expected 1-D probs, got shape {probs.shape}")

    max_prob = float(probs.max())
    # top-2 margin
    sorted_probs = np.sort(probs)[::-1]
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = float(top1 - top2)

    # Shannon entropy
    eps = 1e-12
    entropy = float(-np.sum(probs * np.log(probs + eps)))

    th = config.uncertainty
    return {
        "max_prob": max_prob,
        "margin": margin,
        "entropy": entropy,
        "low_confidence": max_prob < th.low_confidence,
        "small_margin": margin < th.small_margin,
        "high_entropy": entropy > th.high_entropy,
    }


# ---------------------------------------------------------------------------
# 2. Model disagreement
# ---------------------------------------------------------------------------

def compute_model_disagreement(preds: Dict[str, int]) -> Dict[str, Any]:
    """
    Parameters
    ----------
    preds : mapping model_name -> predicted_label_index

    Returns
    -------
    dict with:
        unique_labels, disagreement, disagreement_score
    """
    labels = list(preds.values())
    unique = set(labels)
    disagreement = len(unique) > 1

    # disagreement score = 1 - majority_fraction
    values, counts = np.unique(labels, return_counts=True)
    majority_fraction = counts.max() / len(labels)
    disagreement_score = float(1.0 - majority_fraction)

    return {
        "unique_labels": unique,
        "disagreement": disagreement,
        "disagreement_score": disagreement_score,
    }


# ---------------------------------------------------------------------------
# 3. Missing-data check
# ---------------------------------------------------------------------------

def compute_missingness_flags(
    x_row: np.ndarray, config: EscalationConfig
) -> Dict[str, Any]:
    """
    Simple NaN-based missingness logic.

    Parameters
    ----------
    x_row : (n_features,) array
    """
    x_row = np.asarray(x_row, dtype=float)
    mask = np.isnan(x_row)
    missing_fraction = float(mask.mean())
    has_missing = bool(mask.any())
    critical_missing = missing_fraction >= config.missing.critical_missing_fraction

    return {
        "missing_fraction": missing_fraction,
        "has_missing": has_missing,
        "critical_missing": critical_missing,
    }


# ---------------------------------------------------------------------------
# 4. Multimodal mismatch (placeholder, safe default = False)
# ---------------------------------------------------------------------------

def compute_multimodal_mismatch(
    x_row: np.ndarray, config: EscalationConfig
) -> Dict[str, Any]:
    """
    Very light-weight placeholder.

    If you later specify feature groups in config.multimodal_feature_groups,
    you can compare simple averages between groups.

    For now, if no groups are defined → always returns mismatch=False.
    """
    groups = config.multimodal_feature_groups
    if not groups or len(groups) < 2:
        return {"mismatch": False, "group_scores": {}}

    x_row = np.asarray(x_row, dtype=float)
    group_scores = {}
    for name, idxs in groups.items():
        idxs = [i for i in idxs if i < len(x_row)]
        if not idxs:
            continue
        vals = x_row[idxs]
        group_scores[name] = float(np.nanmean(vals))

    if len(group_scores) < 2:
        return {"mismatch": False, "group_scores": group_scores}

    # crude rule: if any pair differs by more than 1 std of all scores → mismatch
    scores = np.array(list(group_scores.values()))
    diff_matrix = np.abs(scores[:, None] - scores[None, :])
    max_diff = float(diff_matrix.max())
    std_all = float(scores.std()) if scores.size > 1 else 0.0
    threshold = std_all if std_all > 0 else 0.0

    mismatch = bool(max_diff > threshold and threshold > 0)
    return {"mismatch": mismatch, "group_scores": group_scores}


# ---------------------------------------------------------------------------
# 5. SHAP instability placeholder (for future extension)
# ---------------------------------------------------------------------------

def compute_shap_instability() -> Dict[str, Any]:
    """
    Placeholder – in Step 2 we don't recompute SHAP.
    You can plug real numbers in later.

    For now:
        instability = False
        variance = NaN
    """
    return {"instability": False, "variance": float("nan")}
