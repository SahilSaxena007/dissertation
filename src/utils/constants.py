"""
Global constants for the Alzheimer's biomarker pipeline.
"""

import os

# Class and feature definitions
CLASS_NAMES = ["SCD", "MCI", "AD"]  # Diagnostic stages
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}
INT_TO_CLASS = {i: name for name, i in CLASS_TO_INT.items()}

# Random seed for reproducibility
RANDOM_SEED = 42

# Path constants (relative to src/)
DATA_DIR = "../data"
REPORTS_DIR = "../reports"
FIGURES_DIR = f"{REPORTS_DIR}/figures"
TABLES_DIR = f"{REPORTS_DIR}/tables"
NOTES_DIR = f"{REPORTS_DIR}/notes"

# Feature list (update after Preprocess.py runs)
FEATURE_LIST = [
    'ABETA', 'TAU', 'PTAU', 'AB4240',
    'PLASMATAU', 'PLASMA_NFL',
    'AGE', 'PTEDUCAT', 'MMSE'
]

# Ensure output directories exist
for dir_path in [FIGURES_DIR, TABLES_DIR, NOTES_DIR]:
    os.makedirs(dir_path, exist_ok=True)