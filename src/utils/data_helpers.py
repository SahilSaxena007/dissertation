# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Data loading and preprocessing helpers.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .constants import CLASS_NAMES


def _coerce_dx_to_int(dx: pd.Series) -> pd.Series:
    """
    Normalize diagnosis labels to fixed mapping:
      SCD=0, MCI=1, AD=2
    """
    if pd.api.types.is_numeric_dtype(dx):
        out = pd.to_numeric(dx, errors="coerce")
        valid = out.dropna().isin([0, 1, 2]).all()
        if not valid:
            raise ValueError("Numeric DX contains labels outside {0,1,2}.")
        return out.astype(int)

    mapping = {"SCD": 0, "MCI": 1, "AD": 2}
    normalized = dx.astype(str).str.strip().str.upper().map(mapping)
    if normalized.isna().any():
        bad_vals = sorted(set(dx[normalized.isna()].astype(str)))
        raise ValueError(f"Unrecognized DX labels: {bad_vals}")
    return normalized.astype(int)


def load_preprocessed_data(
    data_path: str = "../data/preprocessed_data.csv",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load preprocessed data and return X, y split.

    Parameters
    ----------
    data_path : str
        Path to preprocessed_data.csv

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix.
    y : pd.Series
        Target labels, integer-encoded.
    class_names : list[str]
        Fixed class order: ["SCD", "MCI", "AD"].
    """
    df = pd.read_csv(data_path)
    if "DX" not in df.columns:
        raise ValueError(f"Missing required target column 'DX' in {data_path}")

    df = df.dropna(subset=["DX"]).copy()
    y = _coerce_dx_to_int(df["DX"])

    X = df.drop(columns=["DX"]).copy()
    X = X.select_dtypes(include=["number"]).copy()

    if X.shape[1] == 0:
        raise ValueError("No numeric features available after preprocessing.")
    if len(X) != len(y):
        raise ValueError("Feature/target length mismatch after preprocessing.")

    return X, y, CLASS_NAMES.copy()
