"""
Sanity checks for new HITL backend modules.
"""

import os
import sqlite3
import sys
import tempfile

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hitl.interaction_logger import ensure_db, log_clinician_feedback
from hitl.feedback_loop import run_active_learning_simulation
from hitl.simulated_clinician import simulate_corrected_labels


def test_feedback_logger_validation_and_write():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "test_feedback.db")
        ensure_db(db)
        log_clinician_feedback(
            source="oof",
            sample_id=1,
            predicted_label="MCI",
            agree=False,
            corrected_label="AD",
            clinician_confidence=4,
            true_label="AD",
            notes="test",
            db_path=db,
        )
        conn = sqlite3.connect(db)
        try:
            n = conn.execute("SELECT COUNT(*) FROM clinician_feedback").fetchone()[0]
        finally:
            conn.close()
        assert n == 1


def test_simulated_clinician_outputs_valid_classes():
    y_true = np.array([0, 1, 2, 1, 0], dtype=int)
    y_ai = np.array([0, 0, 2, 2, 1], dtype=int)
    y_corr = simulate_corrected_labels(y_true, y_ai, clinician_accuracy=0.9, seed=1)
    assert y_corr.shape == y_true.shape
    assert set(np.unique(y_corr)).issubset({0, 1, 2})


def test_active_learning_curve_smoke():
    curve = run_active_learning_simulation(
        clinician_accuracy=0.9,
        batch_size=20,
        max_rounds=1,
        use_confidence_weighting=True,
        seed=42,
    )
    assert {"round", "feedback_pool_size", "test_accuracy", "test_macro_f1"}.issubset(curve.columns)
    assert len(curve) >= 1


if __name__ == "__main__":
    test_feedback_logger_validation_and_write()
    test_simulated_clinician_outputs_valid_classes()
    test_active_learning_curve_smoke()
    print("test_hitl_backend.py passed.")
