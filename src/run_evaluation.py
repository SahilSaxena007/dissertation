import os

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from eval.orchestrator import analyze_model_performance


class_names = ["SCD", "MCI", "AD"]
artifacts_dir = "../artifacts"


def oof_artifacts_exist():
    required = [
        "X_oof.npy",
        "y_oof.npy",
        "oof_cat_proba.npy",
        "oof_rf_proba.npy",
        "oof_nn_proba.npy",
        "oof_ens_proba.npy",
    ]
    return all(os.path.exists(os.path.join(artifacts_dir, name)) for name in required)


def evaluate_oof():
    print("=== Evaluating with leak-free OOF artifacts ===")

    X_eval = np.load(os.path.join(artifacts_dir, "X_oof.npy"))
    y_eval = np.load(os.path.join(artifacts_dir, "y_oof.npy"))

    y_prob_cat = np.load(os.path.join(artifacts_dir, "oof_cat_proba.npy"))
    y_prob_rf = np.load(os.path.join(artifacts_dir, "oof_rf_proba.npy"))
    y_prob_nn = np.load(os.path.join(artifacts_dir, "oof_nn_proba.npy"))
    y_prob_ens = np.load(os.path.join(artifacts_dir, "oof_ens_proba.npy"))
    selected_path = os.path.join(artifacts_dir, "oof_ens_proba_selected.npy")
    calibrated_path = os.path.join(artifacts_dir, "oof_ens_proba_calibrated.npy")
    if os.path.exists(selected_path):
        y_prob_ens_selected = np.load(selected_path)
    elif os.path.exists(calibrated_path):
        y_prob_ens_selected = np.load(calibrated_path)
    else:
        y_prob_ens_selected = None

    y_pred_cat = np.argmax(y_prob_cat, axis=1)
    y_pred_rf = np.argmax(y_prob_rf, axis=1)
    y_pred_nn = np.argmax(y_prob_nn, axis=1)
    y_pred_ens = np.argmax(y_prob_ens, axis=1)
    y_pred_ens_selected = (
        np.argmax(y_prob_ens_selected, axis=1) if y_prob_ens_selected is not None else None
    )

    # Load deployment models for compatibility and optional SHAP on CatBoost
    cat_model = joblib.load(os.path.join(artifacts_dir, "best_model_catboost.pkl"))

    print("\n=== Evaluating Random Forest (OOF) ===")
    analyze_model_performance(
        y_true=y_eval,
        y_pred=y_pred_rf,
        y_prob=y_prob_rf,
        model=None,
        X_test=X_eval,
        class_names=class_names,
        model_name="RandomForest",
    )

    print("\n=== Evaluating Neural Network (OOF) ===")
    analyze_model_performance(
        y_true=y_eval,
        y_pred=y_pred_nn,
        y_prob=y_prob_nn,
        model=None,
        X_test=X_eval,
        class_names=class_names,
        model_name="NeuralNet",
    )

    print("\n=== Evaluating Soft Voting Ensemble (OOF) ===")
    analyze_model_performance(
        y_true=y_eval,
        y_pred=y_pred_ens,
        y_prob=y_prob_ens,
        model=cat_model,
        X_test=X_eval,
        class_names=class_names,
        model_name="Ensemble",
    )

    if y_prob_ens_selected is not None:
        print("\n=== Evaluating Selected Soft Voting Ensemble (OOF) ===")
        analyze_model_performance(
            y_true=y_eval,
            y_pred=y_pred_ens_selected,
            y_prob=y_prob_ens_selected,
            model=cat_model,
            X_test=X_eval,
            class_names=class_names,
            model_name="EnsembleSelected",
        )

    print("\n=== Evaluating CatBoost (OOF) ===")
    analyze_model_performance(
        y_true=y_eval,
        y_pred=y_pred_cat,
        y_prob=y_prob_cat,
        model=cat_model,
        X_test=X_eval,
        class_names=class_names,
        model_name="CatBoost",
    )


def evaluate_legacy_testset():
    print("=== Evaluating with legacy test split artifacts ===")

    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    models = joblib.load(os.path.join(artifacts_dir, "voting_ensemble.pkl"))
    cat_model = models["catboost"]
    rf_model = models["rf"]
    nn_model = load_model(os.path.join(artifacts_dir, "neural_network.h5"))

    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)

    y_prob_nn = nn_model.predict(X_test, verbose=0)
    y_pred_nn = np.argmax(y_prob_nn, axis=1)

    y_pred_cat = cat_model.predict(X_test)
    y_prob_cat = cat_model.predict_proba(X_test)

    y_prob_ensemble = (y_prob_cat + y_prob_rf + y_prob_nn) / 3
    y_pred_ensemble = np.argmax(y_prob_ensemble, axis=1)

    print("\n=== Evaluating Random Forest ===")
    analyze_model_performance(y_test, y_pred_rf, y_prob_rf, rf_model, X_test, class_names, "RandomForest")

    print("\n=== Evaluating Neural Network ===")
    analyze_model_performance(y_test, y_pred_nn, y_prob_nn, nn_model, X_test, class_names, "NeuralNet")

    print("\n=== Evaluating Soft Voting Ensemble ===")
    analyze_model_performance(y_test, y_pred_ensemble, y_prob_ensemble, cat_model, X_test, class_names, "Ensemble")

    print("\n=== Evaluating CatBoost ===")
    analyze_model_performance(y_test, y_pred_cat, y_prob_cat, cat_model, X_test, class_names, "CatBoost")


if __name__ == "__main__":
    if oof_artifacts_exist():
        evaluate_oof()
    else:
        evaluate_legacy_testset()

    print("\n=== ALL MODELS EVALUATED SUCCESSFULLY ===")
    print("Reports saved to /reports/")
