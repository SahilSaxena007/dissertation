"""
Basic deterministic checks for Phase 2 HITL helpers.

Run from repo root:
    python tests/test_hitl_phase2.py
"""

import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESCALATION_DIR = os.path.join(ROOT, "src", "escalation")
if ESCALATION_DIR not in sys.path:
    sys.path.insert(0, ESCALATION_DIR)

from run_step3_thresholds import build_human_accuracy_grid  # noqa: E402
from step2_feature_builder import build_meta_dataframe_from_escalation  # noqa: E402


def test_build_meta_dataframe_from_escalation():
    df = pd.DataFrame(
        {
            "sample_id": [0, 1],
            "true_label": [1, 2],
            "pred_ens": [1, 0],
            "ens_max_prob": [0.9, 0.6],
            "ens_margin": [0.5, 0.1],
            "ens_entropy": [0.3, 0.95],
            "disagreement_score": [0.0, 1.0 / 3.0],
            "missing_fraction": [0.0, 0.25],
            "critical_missing": [False, True],
            "multimodal_mismatch": [False, True],
        }
    )

    out = build_meta_dataframe_from_escalation(df)
    assert out.shape[0] == 2
    assert out["ai_correct"].tolist() == [1, 0]
    assert out["ai_error"].tolist() == [0, 1]
    assert np.allclose(out["risk_1_inv_conf"].values, np.array([0.1, 0.4]))
    assert out["risk_6_critical_missing"].tolist() == [0, 1]
    assert out["risk_7_multimodal_mismatch"].tolist() == [0, 1]


def test_build_human_accuracy_grid():
    grid = build_human_accuracy_grid(min_value=0.85, max_value=0.95, step=0.01)
    assert grid[0] == 0.85
    assert grid[-1] == 0.95
    assert len(grid) == 11


if __name__ == "__main__":
    test_build_meta_dataframe_from_escalation()
    test_build_human_accuracy_grid()
    print("test_hitl_phase2.py passed.")
