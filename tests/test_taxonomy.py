"""
Test error taxonomy export and analysis.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import error_taxonomy
from utils import constants


def _synthetic_probs(y_true: np.ndarray, n_classes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_prob = np.zeros((len(y_true), n_classes), dtype=float)
    for i in range(len(y_true)):
        y_prob[i, y_true[i]] = rng.uniform(0.5, 1.0)
        remain = 1.0 - y_prob[i, y_true[i]]
        other = np.delete(np.arange(n_classes), y_true[i])
        y_prob[i, other] = rng.dirichlet(np.ones(n_classes - 1)) * remain
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    return y_prob


def test_export_error_taxonomy():
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3

    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(y_true)

    y_pred = y_true.copy()
    error_idx = np.random.choice(len(y_true), size=int(0.2 * len(y_true)), replace=False)
    y_pred[error_idx] = np.random.randint(0, n_classes, size=len(error_idx))

    y_prob = _synthetic_probs(y_true, n_classes=n_classes, seed=constants.RANDOM_SEED)

    sample_ids = np.arange(1000, 1000 + n_samples)
    metadata_df = pd.DataFrame(
        {
            "age": np.random.uniform(50, 90, n_samples),
            "sex": np.random.choice(["M", "F"], n_samples),
            "scanner": np.random.choice(["Siemens", "GE", "Philips"], n_samples),
        }
    )

    os.makedirs("./reports/tables", exist_ok=True)
    save_path = "./reports/tables/test_error_taxonomy_catboost.csv"
    error_df = error_taxonomy.export_error_taxonomy(
        y_true,
        y_pred,
        y_prob,
        constants.CLASS_NAMES,
        model_name="CatBoost",
        save_path=save_path,
        sample_ids=sample_ids,
        metadata_df=metadata_df,
    )

    assert os.path.exists(save_path)
    assert isinstance(error_df, pd.DataFrame)
    assert len(error_df) == n_samples

    expected_cols = {
        "sample_id",
        "sample_index",
        "true_label",
        "predicted_label",
        "entropy",
        "margin",
        "max_prob",
        "is_correct",
        "is_error",
        "model_name",
        "timestamp",
    }
    assert expected_cols.issubset(error_df.columns)

    prob_cols = [f"prob_{cn}" for cn in constants.CLASS_NAMES]
    assert all(col in error_df.columns for col in prob_cols)
    assert "age" in error_df.columns and "sex" in error_df.columns

    return error_df


def test_error_taxonomy_summary(error_df: pd.DataFrame):
    summary = error_taxonomy.error_taxonomy_summary(error_df, constants.CLASS_NAMES)
    assert isinstance(summary, dict)
    assert abs((summary["accuracy"] + summary["error_rate"]) - 1.0) < 1e-9



def test_error_cases_by_uncertainty(error_df: pd.DataFrame):
    top_uncertain = error_taxonomy.error_cases_by_uncertainty(error_df, n_top=5)
    assert isinstance(top_uncertain, pd.DataFrame)
    assert len(top_uncertain) <= 5
    assert (top_uncertain["is_error"] == 1).all()



def test_export_meta_model_training_data():
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 99
    n_classes = 3

    error_dfs = []
    for model_name in ["CatBoost", "RandomForest", "NeuralNetwork"]:
        y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
        n_local = len(y_true)
        np.random.shuffle(y_true)

        y_pred = y_true.copy()
        error_idx = np.random.choice(len(y_true), size=int(0.15 * len(y_true)), replace=False)
        y_pred[error_idx] = np.random.randint(0, n_classes, size=len(error_idx))

        y_prob = _synthetic_probs(y_true, n_classes=n_classes, seed=constants.RANDOM_SEED)

        error_df = error_taxonomy.export_error_taxonomy(
            y_true,
            y_pred,
            y_prob,
            constants.CLASS_NAMES,
            model_name=model_name,
            save_path=f"./reports/tables/test_error_taxonomy_{model_name.lower()}.csv",
        )
        error_dfs.append(error_df)

    save_path = "./reports/tables/test_meta_model_training_data.csv"
    meta_training_df = error_taxonomy.export_meta_model_training_data(
        error_dfs, constants.CLASS_NAMES, save_path
    )

    assert os.path.exists(save_path)
    assert isinstance(meta_training_df, pd.DataFrame)
    assert len(meta_training_df) == len(error_dfs) * n_local


if __name__ == "__main__":
    df = test_export_error_taxonomy()
    test_error_taxonomy_summary(df)
    test_error_cases_by_uncertainty(df)
    test_export_meta_model_training_data()
    print("test_taxonomy.py passed.")
