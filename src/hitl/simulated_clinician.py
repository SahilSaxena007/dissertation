"""
Simulated clinician utilities for HITL experiments.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["SCD", "MCI", "AD"]


def simulate_corrected_labels(
    y_true: np.ndarray,
    y_ai_pred: np.ndarray,
    clinician_accuracy: float,
    seed: int = 42,
) -> np.ndarray:
    if clinician_accuracy <= 0.0 or clinician_accuracy > 1.0:
        raise ValueError("clinician_accuracy must be in (0, 1].")

    y_true = np.asarray(y_true, dtype=int)
    y_ai_pred = np.asarray(y_ai_pred, dtype=int)
    if y_true.shape != y_ai_pred.shape:
        raise ValueError("y_true and y_ai_pred must have the same shape.")

    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(np.concatenate([y_true, y_ai_pred])))
    corrected = np.copy(y_true)

    is_correct = rng.random(y_true.shape[0]) < clinician_accuracy
    wrong_idx = np.where(~is_correct)[0]
    for i in wrong_idx:
        choices = [c for c in range(n_classes) if c != y_true[i]]
        corrected[i] = int(rng.choice(choices))
    return corrected


def confidence_to_sample_weight(confidence_1_to_5: np.ndarray) -> np.ndarray:
    conf = np.asarray(confidence_1_to_5, dtype=float)
    conf = np.clip(conf, 1.0, 5.0)
    return 0.5 + (conf / 5.0)

