"""Central project config for reproducibility and orchestration."""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42
UTF8_ENV = {"PYTHONIOENCODING": "utf-8"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"


def ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
