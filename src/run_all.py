# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Deterministic artifact regeneration script (fixed seed config).

Usage (from src/):
  python run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from utils.project_config import SEED

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(SRC_DIR), env=env, check=True)


def main() -> int:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(SEED)
    env["PYTHONIOENCODING"] = "utf-8"

    py = sys.executable

    _run([py, "merge_files.py"], env)
    _run([py, "EDA.py"], env)
    _run([py, "Preprocess.py"], env)
    _run([py, "ModelsFinal.py"], env)
    _run([py, "run_evaluation.py"], env)
    _run([py, "escalation/run_escalation.py", "--run-testset-demo"], env)
    _run([py, "run_analysis_suite.py"], env)

    print("Deterministic pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
