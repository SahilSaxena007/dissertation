"""
Active-learning style feedback loop over escalation risk ordering.

Retrains the actual CatBoost + RF + NN ensemble (not a surrogate model).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from .simulated_clinician import confidence_to_sample_weight, simulate_corrected_labels
except ImportError:
    from simulated_clinician import confidence_to_sample_weight, simulate_corrected_labels

SEED = 42


def _paths():
    root = Path(__file__).resolve().parents[2]
    return {
        "root": root,
        "artifacts": root / "artifacts",
        "tables": root / "reports" / "tables",
    }


def _build_ensemble_predict_from_models(models_dict, X):
    """Get ensemble predictions from a dict of models."""
    all_probs = [m.predict_proba(X) for m in models_dict.values()]
    probs_ens = np.mean(all_probs, axis=0)
    return np.argmax(probs_ens, axis=1)


def _build_ensemble_predict(cat_model, rf_model, nn_model, X):
    """Get ensemble predictions from the 3 core models."""
    probs_cat = cat_model.predict_proba(X)
    probs_rf = rf_model.predict_proba(X)
    probs_nn = nn_model.predict_proba(X)
    probs_ens = (probs_cat + probs_rf + probs_nn) / 3.0
    return np.argmax(probs_ens, axis=1)


def _train_ensemble(X_train, y_train, sample_weight=None, fast=False):
    """Train CatBoost + RF + NN on provided data."""
    iterations = 100 if fast else 500
    n_estimators = 100 if fast else 400
    epochs = 30 if fast else 120

    cat_model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.01,
        depth=6,
        random_seed=SEED,
        verbose=0,
        loss_function="MultiClass",
    )
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    )

    # CatBoost and RF support sample_weight natively
    if sample_weight is not None:
        cat_model.fit(X_train, y_train, sample_weight=sample_weight)
        rf_model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        cat_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

    # For NN: duplicate high-weight samples instead of sample_weight
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Lazy import to avoid TF overhead when not needed
    nn_model = _train_nn(X_train, y_train, sample_weight, epochs, fast)

    return cat_model, rf_model, nn_model


def _train_nn(X_train, y_train, sample_weight, epochs, fast):
    """Train NN with optional sample weighting via duplication."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    if sample_weight is not None:
        # Duplicate samples proportional to weight for NN training
        weights_int = np.round(sample_weight).astype(int).clip(min=1)
        X_nn = np.repeat(X_train, weights_int, axis=0)
        y_nn = np.repeat(y_train, weights_int, axis=0)
    else:
        X_nn = X_train
        y_nn = y_train

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="loss", patience=5 if fast else 10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3 if fast else 5, min_lr=1e-6),
    ]

    model.fit(X_nn, y_nn, epochs=epochs, batch_size=32, verbose=0, callbacks=callbacks)
    return model


def run_active_learning_simulation(
    clinician_accuracy: float = 0.9,
    batch_size: int = 20,
    max_rounds: int = 6,
    use_confidence_weighting: bool = True,
    seed: int = 42,
    fast: bool = False,
) -> pd.DataFrame:
    paths = _paths()
    tables = paths["tables"]
    artifacts = paths["artifacts"]

    df_thresh = pd.read_csv(tables / "threshold_data.csv")
    X_train = np.load(artifacts / "X_oof.npy")
    y_train = np.load(artifacts / "y_oof.npy")
    X_test = np.load(artifacts / "X_test.npy")
    y_test = np.load(artifacts / "y_test.npy")

    if X_train.shape[0] != df_thresh.shape[0]:
        n = min(X_train.shape[0], df_thresh.shape[0])
        X_train = X_train[:n]
        y_train = y_train[:n]
        df_thresh = df_thresh.iloc[:n].copy()

    risk = df_thresh["review_risk_score"].to_numpy(dtype=float)
    y_ai = df_thresh["pred_ens"].to_numpy(dtype=int)
    y_true = df_thresh["true_label"].to_numpy(dtype=int)
    ranked = np.argsort(-risk)

    # Round 0 baseline: load saved ensemble models
    ensemble = joblib.load(artifacts / "voting_ensemble.pkl")
    for name, model in ensemble.items():
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

    base_pred = _build_ensemble_predict_from_models(ensemble, X_test)
    base_acc = accuracy_score(y_test, base_pred)
    base_f1 = f1_score(y_test, base_pred, average="macro")

    rng = np.random.default_rng(seed)
    rows = [
        {
            "round": 0,
            "feedback_pool_size": 0,
            "test_accuracy": float(base_acc),
            "test_macro_f1": float(base_f1),
        }
    ]

    feedback_X = []
    feedback_y = []
    feedback_w = []
    start = 0
    for r in range(1, max_rounds + 1):
        end = min(start + batch_size, ranked.size)
        idx = ranked[start:end]
        if idx.size == 0:
            break

        corrected = simulate_corrected_labels(
            y_true=y_true[idx],
            y_ai_pred=y_ai[idx],
            clinician_accuracy=clinician_accuracy,
            seed=int(rng.integers(1, 1_000_000)),
        )
        conf = rng.integers(3, 6, size=idx.size)
        w = confidence_to_sample_weight(conf) if use_confidence_weighting else np.ones(idx.size)

        feedback_X.append(X_train[idx])
        feedback_y.append(corrected)
        feedback_w.append(w)

        X_fb = np.vstack(feedback_X)
        y_fb = np.concatenate(feedback_y)
        w_fb = np.concatenate(feedback_w)

        X_fit = np.vstack([X_train, X_fb])
        y_fit = np.concatenate([y_train, y_fb])
        sample_weight = np.concatenate([np.ones(X_train.shape[0]), w_fb])

        cat, rf, nn_model = _train_ensemble(
            X_fit, y_fit, sample_weight=sample_weight, fast=fast
        )
        pred = _build_ensemble_predict(cat, rf, nn_model, X_test)

        rows.append(
            {
                "round": r,
                "feedback_pool_size": int(X_fb.shape[0]),
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
            }
        )
        start = end

    return pd.DataFrame(rows)


def run_and_save_experiment(
    clinician_accuracy: float = 0.9,
    batch_size: int = 20,
    max_rounds: int = 6,
    use_confidence_weighting: bool = True,
    seed: int = 42,
    fast: bool = False,
) -> Path:
    df = run_active_learning_simulation(
        clinician_accuracy=clinician_accuracy,
        batch_size=batch_size,
        max_rounds=max_rounds,
        use_confidence_weighting=use_confidence_weighting,
        seed=seed,
        fast=fast,
    )
    paths = _paths()
    out_dir = paths["tables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hitl_active_learning_curve.csv"
    df.to_csv(out_path, index=False)

    summary = {
        "clinician_accuracy": clinician_accuracy,
        "batch_size": batch_size,
        "max_rounds": max_rounds,
        "use_confidence_weighting": use_confidence_weighting,
        "seed": seed,
        "best_test_accuracy": float(df["test_accuracy"].max()),
        "best_round": int(df.loc[df["test_accuracy"].idxmax(), "round"]),
        "output_curve": str(out_path),
    }
    with open(paths["artifacts"] / "hitl_active_learning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path
