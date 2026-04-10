"""
Quick import smoke test for core modules.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("Testing imports...")

targets = [
    ("eval.orchestrator", "from eval import orchestrator"),
    ("eval.metrics", "from eval import metrics"),
    ("utils.constants", "from utils import constants"),
    ("escalation.escalation_engine", "from escalation import escalation_engine"),
    ("hitl.feedback_loop", "from hitl import feedback_loop"),
    ("dashboard.app", "from dashboard import app"),
]

failures = []
for name, stmt in targets:
    try:
        exec(stmt, {})
        print(f"[OK] {name}")
    except Exception as e:
        failures.append((name, str(e)))
        print(f"[FAIL] {name}: {e}")

if failures:
    print("\nImport failures:")
    for name, err in failures:
        print(f"- {name}: {err}")
    raise SystemExit(1)

print("\nAll imports successful.")
