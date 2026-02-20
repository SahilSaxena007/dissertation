"""
Run the local test suite from one command with UTF-8 output.

Usage (repo root):
  python scripts/run_tests.py
  python scripts/run_tests.py --venv src/.venv/Scripts/python.exe
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS = [
    "test_imports.py",
    "test_metrics.py",
    "test_uncertainty.py",
    "test_statistical.py",
    "test_explainability.py",
    "test_taxonomy.py",
    "test_bias.py",
    "test_visualization.py",
    "test_hitl_phase2.py",
    "test_hitl_backend.py",
]


def _default_python() -> str:
    venv_py = ROOT / "src" / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default=_default_python(), help="Python executable to use")
    parser.add_argument("--tests", nargs="+", default=DEFAULT_TESTS)
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    failures = []
    for test_name in args.tests:
        test_path = ROOT / "tests" / test_name
        if not test_path.exists():
            failures.append((test_name, "missing file"))
            print(f"[FAIL] {test_name} (missing)")
            continue

        print(f"\n=== {test_name} ===")
        proc = subprocess.run([args.venv, str(test_path)], cwd=str(ROOT), env=env)
        if proc.returncode != 0:
            failures.append((test_name, f"exit={proc.returncode}"))
            print(f"[FAIL] {test_name} exit={proc.returncode}")
        else:
            print(f"[OK] {test_name}")

    if failures:
        print("\nFailed tests:")
        for name, reason in failures:
            print(f"- {name}: {reason}")
        return 1

    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
