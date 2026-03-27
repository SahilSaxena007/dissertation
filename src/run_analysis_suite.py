"""
Non-interactive implementation validation pipeline.

Usage (from src/):
  python run_analysis_suite.py
  python run_analysis_suite.py --full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils.project_config import SEED

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
TABLES_DIR = ROOT / "reports" / "tables"

CORE_OUTPUTS = [
    "hitl_simulated_clinician_experiment.csv",
    "threshold_analysis.csv",
    "threshold_analysis_by_human_accuracy.csv",
    "threshold_sensitivity_summary.csv",
    "statistical_tests.csv",
    "cost_analysis.csv",
    "ablation_study.csv",
    "fairness_analysis.csv",
    "conformal_prediction.csv",
    "uncertainty_decomposition.csv",
    "efficiency_analysis.csv",
    "agreement_analysis.csv",
]


def _run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(SRC_DIR), check=True)


def _assert_outputs() -> None:
    missing = []
    empty = []
    for name in CORE_OUTPUTS:
        p = TABLES_DIR / name
        if not p.exists():
            missing.append(name)
            continue
        if p.stat().st_size == 0:
            empty.append(name)

    if missing or empty:
        if missing:
            print("Missing outputs:")
            for n in missing:
                print(f"- {n}")
        if empty:
            print("Empty outputs:")
            for n in empty:
                print(f"- {n}")
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run full training + Phase 2 build before evaluation scripts.",
    )
    args = parser.parse_args()

    py = sys.executable

    if args.full:
        _run([py, "ModelsFinal.py"])
        _run([py, "run_evaluation.py"])
        _run([py, "escalation/run_escalation.py", "--run-testset-demo"])

    _run(
        [
            py,
            "hitl/run_experiment.py",
            "--accuracies",
            "0.85",
            "0.9",
            "0.95",
            "--seed",
            str(SEED),
            "--batch-size",
            "10",
            "--max-rounds",
            "1",
        ]
    )
    _run([py, "escalation/threshold_runner.py"])
    _run([py, "evaluation/run_analysis.py"])

    _assert_outputs()
    print("Analysis suite completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
