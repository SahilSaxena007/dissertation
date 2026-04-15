# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Run all evaluation analysis scripts in sequence.

Usage (from src/):
    python evaluation/run_analysis.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = SRC_DIR / "evaluation"


def _run(script: str) -> None:
    cmd = [sys.executable, str(EVAL_DIR / script)]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(SRC_DIR), check=True)


def main() -> int:
    _run("statistical_tests.py")
    _run("cost_analysis.py")
    _run("ablation_study.py")
    _run("fairness_analysis.py")
    _run("uncertainty_quantification.py")
    _run("efficiency_analysis.py")
    _run("agreement_analysis.py")
    _run("threshold_tradeoff_plot.py")
    _run("hitl_perclass_comparison.py")
    _run("clinical_cost_plot.py")
    _run("escalation_quality.py")
    _run("active_learning_plot.py")
    print("Evaluation analysis suite completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
