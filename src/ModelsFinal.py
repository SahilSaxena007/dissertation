import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

os.makedirs("../artifacts", exist_ok=True)
os.makedirs("../Outputs", exist_ok=True)

# Quiet TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
CV_FOLDS = 5
CV_REPEATS = 2
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def multiclass_ece(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    correctness = (y_pred == y_true).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        if np.any(in_bin):
            bin_acc = correctness[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (np.sum(in_bin) / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


class MulticlassProbabilityCalibrator:
    def __init__(self, method="sigmoid"):
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be one of {'sigmoid', 'isotonic'}")
        self.method = method
        self.models = []
        self.n_classes = None

    def fit(self, probs, y):
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes = probs.shape[1]
        self.models = []

        for class_idx in range(self.n_classes):
            targets = (y == class_idx).astype(int)
            class_probs = probs[:, class_idx]

            if self.method == "sigmoid":
                clf = LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=SEED,
                )
                clf.fit(class_probs.reshape(-1, 1), targets)
                self.models.append(clf)
            else:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(class_probs, targets)
                self.models.append(iso)
        return self

    def predict_proba(self, probs):
        probs = np.asarray(probs, dtype=float)
        if self.n_classes is None or len(self.models) != self.n_classes:
            raise RuntimeError("Calibrator must be fitted before predict_proba.")

        calibrated = np.zeros_like(probs, dtype=float)
        for class_idx, model in enumerate(self.models):
            class_probs = probs[:, class_idx]
            if self.method == "sigmoid":
                calibrated[:, class_idx] = model.predict_proba(class_probs.reshape(-1, 1))[:, 1]
            else:
                calibrated[:, class_idx] = model.predict(class_probs)

        calibrated = np.clip(calibrated, 1e-12, 1.0)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        return calibrated


def generate_cv_calibrated_probs(raw_probs, y_true, method, n_splits=5):
    raw_probs = np.asarray(raw_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    calibrated = np.zeros_like(raw_probs, dtype=float)

    for train_idx, val_idx in cv.split(raw_probs, y_true):
        calibrator = MulticlassProbabilityCalibrator(method=method).fit(raw_probs[train_idx], y_true[train_idx])
        calibrated[val_idx] = calibrator.predict_proba(raw_probs[val_idx])

    return calibrated


def create_model(input_dim=12, units_1=64, units_2=32, units_3=16, dropout_1=0.3, dropout_2=0.2, learning_rate=0.001):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(units_1, activation="relu"),
            BatchNormalization(),
            Dropout(dropout_1),
            Dense(units_2, activation="relu"),
            BatchNormalization(),
            Dropout(dropout_2),
            Dense(units_3, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def multiclass_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob, multi_class="ovr")


def summarize_metrics(name, y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "Model": name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "MacroF1": float(f1_score(y_true, y_pred, average="macro")),
        "AUC": float(multiclass_auc(y_true, y_prob)),
    }


def summarize_cv_metrics(cv_metrics_df):
    summary = (
        cv_metrics_df.groupby("Model")[["Accuracy", "MacroF1", "AUC"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "Model",
        "Accuracy_Mean",
        "Accuracy_Std",
        "MacroF1_Mean",
        "MacroF1_Std",
        "AUC_Mean",
        "AUC_Std",
    ]
    return summary


# ---------------------------------------------------------------------
# Load and prepare dataset
# ---------------------------------------------------------------------
data = pd.read_csv("../data/preprocessed_data.csv")
data = data.dropna(subset=["DX"]).copy()

y = data["DX"].astype(int).to_numpy()
X_df = data.drop(columns=["DX"])
X_df = X_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).copy()
X_all = X_df.to_numpy(dtype=float)
feature_names = X_df.columns.tolist()

# ---------------------------------------------------------------------
# OOF generation for leak-free evaluation
# ---------------------------------------------------------------------
n_samples = X_all.shape[0]
n_classes = len(np.unique(y))
cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=SEED)

oof_cat_sum = np.zeros((n_samples, n_classes), dtype=float)
oof_rf_sum = np.zeros((n_samples, n_classes), dtype=float)
oof_nn_sum = np.zeros((n_samples, n_classes), dtype=float)
oof_selected_X_sum = np.zeros((n_samples, 12), dtype=float)
oof_counts = np.zeros(n_samples, dtype=int)
oof_sample_ids = np.arange(n_samples, dtype=int)
cv_metric_rows = []

total_folds = CV_FOLDS * CV_REPEATS
print(
    f"Generating OOF predictions with RepeatedStratifiedKFold "
    f"(k={CV_FOLDS}, repeats={CV_REPEATS}, total folds={total_folds})..."
)

for split_idx, (train_idx, val_idx) in enumerate(cv.split(X_all, y), start=1):
    repeat_id = (split_idx - 1) // CV_FOLDS + 1
    fold_id = (split_idx - 1) % CV_FOLDS + 1
    print(f"Repeat {repeat_id}/{CV_REPEATS}, Fold {fold_id}/{CV_FOLDS}")

    X_train_raw = X_all[train_idx]
    y_train = y[train_idx]
    X_val_raw = X_all[val_idx]

    imputer_fold = KNNImputer(n_neighbors=9, weights="distance")
    X_train_imp = imputer_fold.fit_transform(X_train_raw)
    X_val_imp = imputer_fold.transform(X_val_raw)

    selector_fold = SelectKBest(mutual_info_classif, k=12)
    X_train_sel = selector_fold.fit_transform(X_train_imp, y_train)
    X_val_sel = selector_fold.transform(X_val_imp)

    scaler_fold = RobustScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train_sel)
    X_val_scaled = scaler_fold.transform(X_val_sel)

    cat_model_fold = CatBoostClassifier(
        iterations=500,
        learning_rate=0.01,
        depth=6,
        random_seed=SEED,
        verbose=0,
        loss_function="MultiClass",
    )
    rf_model_fold = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    nn_model_fold = KerasClassifier(
        model=create_model,
        model__input_dim=X_train_scaled.shape[1],
        random_state=SEED,
        epochs=120,
        batch_size=32,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6),
        ],
    )

    cat_model_fold.fit(X_train_scaled, y_train)
    rf_model_fold.fit(X_train_scaled, y_train)
    nn_model_fold.fit(X_train_scaled, y_train)

    y_prob_cat_fold = cat_model_fold.predict_proba(X_val_scaled)
    y_prob_rf_fold = rf_model_fold.predict_proba(X_val_scaled)
    y_prob_nn_fold = nn_model_fold.predict_proba(X_val_scaled)
    y_prob_ens_fold = (y_prob_cat_fold + y_prob_rf_fold + y_prob_nn_fold) / 3.0

    oof_cat_sum[val_idx] += y_prob_cat_fold
    oof_rf_sum[val_idx] += y_prob_rf_fold
    oof_nn_sum[val_idx] += y_prob_nn_fold
    oof_selected_X_sum[val_idx] += X_val_scaled
    oof_counts[val_idx] += 1

    y_val = y[val_idx]
    for model_name, y_prob_fold in [
        ("CatBoost_OOF", y_prob_cat_fold),
        ("RandomForest_OOF", y_prob_rf_fold),
        ("NeuralNetwork_OOF", y_prob_nn_fold),
        ("VotingEnsemble_OOF", y_prob_ens_fold),
    ]:
        fold_metrics = summarize_metrics(model_name, y_val, y_prob_fold)
        cv_metric_rows.append(
            {
                "Repeat": repeat_id,
                "Fold": fold_id,
                "Model": fold_metrics["Model"],
                "Accuracy": fold_metrics["Accuracy"],
                "MacroF1": fold_metrics["MacroF1"],
                "AUC": fold_metrics["AUC"],
            }
        )

if np.any(oof_counts == 0):
    raise RuntimeError("Each sample must appear in at least one validation fold.")

oof_cat = oof_cat_sum / oof_counts[:, None]
oof_rf = oof_rf_sum / oof_counts[:, None]
oof_nn = oof_nn_sum / oof_counts[:, None]
oof_selected_X = oof_selected_X_sum / oof_counts[:, None]
oof_ens = (oof_cat + oof_rf + oof_nn) / 3.0

# ---------------------------------------------------------------------
# Probability calibration (Phase 1.1)
# ---------------------------------------------------------------------
print("Calibrating OOF ensemble probabilities (sigmoid + isotonic)...")
oof_ens_sigmoid = generate_cv_calibrated_probs(oof_ens, y, method="sigmoid", n_splits=5)
oof_ens_isotonic = generate_cv_calibrated_probs(oof_ens, y, method="isotonic", n_splits=5)

raw_logloss = float(log_loss(y, oof_ens))
sigmoid_logloss = float(log_loss(y, oof_ens_sigmoid))
isotonic_logloss = float(log_loss(y, oof_ens_isotonic))
raw_ece = multiclass_ece(y, oof_ens, n_bins=15)
sigmoid_ece = multiclass_ece(y, oof_ens_sigmoid, n_bins=15)
isotonic_ece = multiclass_ece(y, oof_ens_isotonic, n_bins=15)

selection_candidates = {
    "raw": {"proba": oof_ens, "logloss": raw_logloss, "ece": raw_ece},
    "sigmoid": {"proba": oof_ens_sigmoid, "logloss": sigmoid_logloss, "ece": sigmoid_ece},
    "isotonic": {"proba": oof_ens_isotonic, "logloss": isotonic_logloss, "ece": isotonic_ece},
}
best_calibration_method = min(
    selection_candidates.keys(),
    key=lambda name: (selection_candidates[name]["logloss"], selection_candidates[name]["ece"]),
)
oof_ens_selected = selection_candidates[best_calibration_method]["proba"]

print(
    "Calibration summary (OOF CV): "
    f"raw logloss={raw_logloss:.5f}, sigmoid={sigmoid_logloss:.5f}, isotonic={isotonic_logloss:.5f}; "
    f"raw ECE={raw_ece:.5f}, sigmoid={sigmoid_ece:.5f}, isotonic={isotonic_ece:.5f}; "
    f"selected={best_calibration_method}"
)

cv_metrics_table = pd.DataFrame(cv_metric_rows)
cv_metrics_table.to_csv("../Outputs/cv_fold_metrics.csv", index=False)

metrics_table = summarize_cv_metrics(cv_metrics_table)
metrics_table.to_csv("../Outputs/overall_metrics.csv", index=False)
print("Saved fold metrics to ../Outputs/cv_fold_metrics.csv")
print("Saved CV mean +/- std metrics to ../Outputs/overall_metrics.csv")

# Save OOF artifacts for evaluation and escalation
np.save("../artifacts/oof_cat_proba.npy", oof_cat)
np.save("../artifacts/oof_rf_proba.npy", oof_rf)
np.save("../artifacts/oof_nn_proba.npy", oof_nn)
np.save("../artifacts/oof_ens_proba.npy", oof_ens)
np.save("../artifacts/oof_ens_proba_calibrated_sigmoid.npy", oof_ens_sigmoid)
np.save("../artifacts/oof_ens_proba_calibrated_isotonic.npy", oof_ens_isotonic)
np.save("../artifacts/oof_ens_proba_calibrated.npy", oof_ens_selected)
np.save("../artifacts/oof_ens_proba_selected.npy", oof_ens_selected)
np.save("../artifacts/oof_pred_ens.npy", np.argmax(oof_ens, axis=1))
np.save("../artifacts/y_oof.npy", y)
np.save("../artifacts/X_oof.npy", oof_selected_X)
np.save("../artifacts/oof_sample_ids.npy", oof_sample_ids)

calibration_summary = {
    "selected_method": best_calibration_method,
    "selected_logloss": selection_candidates[best_calibration_method]["logloss"],
    "selected_ece": selection_candidates[best_calibration_method]["ece"],
    "logloss": {
        "raw": raw_logloss,
        "sigmoid": sigmoid_logloss,
        "isotonic": isotonic_logloss,
    },
    "ece": {
        "raw": raw_ece,
        "sigmoid": sigmoid_ece,
        "isotonic": isotonic_ece,
    },
}
with open("../artifacts/calibration_summary.json", "w", encoding="utf-8") as f:
    json.dump(calibration_summary, f, indent=2)

sigmoid_calibrator = MulticlassProbabilityCalibrator(method="sigmoid").fit(oof_ens, y)
isotonic_calibrator = MulticlassProbabilityCalibrator(method="isotonic").fit(oof_ens, y)
joblib.dump(sigmoid_calibrator, "../artifacts/calibrator_sigmoid.pkl")
joblib.dump(isotonic_calibrator, "../artifacts/calibrator_isotonic.pkl")
if best_calibration_method == "sigmoid":
    joblib.dump(sigmoid_calibrator, "../artifacts/calibrator_best.pkl")
elif best_calibration_method == "isotonic":
    joblib.dump(isotonic_calibrator, "../artifacts/calibrator_best.pkl")
else:
    # Raw selected: no probability transform beyond base ensemble.
    joblib.dump(None, "../artifacts/calibrator_best.pkl")

# ---------------------------------------------------------------------
# Fit preprocessing and final models on full data for deployment artifacts
# ---------------------------------------------------------------------
imputer = KNNImputer(n_neighbors=9, weights="distance")
X_imputed = imputer.fit_transform(X_all)

selector = SelectKBest(mutual_info_classif, k=12)
X_selected = selector.fit_transform(X_imputed, y)
selected_features = X_df.columns[selector.get_support()].tolist()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)

joblib.dump(imputer, "../artifacts/imputer.pkl")
joblib.dump(selector, "../artifacts/select_k_best.pkl")
joblib.dump(scaler, "../artifacts/scaler.pkl")
with open("../artifacts/selected_features.json", "w", encoding="utf-8") as f:
    json.dump(selected_features, f, indent=2)

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.01,
    depth=6,
    random_seed=SEED,
    verbose=0,
    loss_function="MultiClass",
)
rf_model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
nn_model = KerasClassifier(
    model=create_model,
    model__input_dim=X_scaled.shape[1],
    random_state=SEED,
    epochs=120,
    batch_size=32,
    verbose=0,
    callbacks=[
        EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6),
    ],
)

cat_model.fit(X_scaled, y)
rf_model.fit(X_scaled, y)
nn_model.fit(X_scaled, y)

joblib.dump(cat_model, "../artifacts/best_model_catboost.pkl")
joblib.dump(rf_model, "../artifacts/random_forest.pkl")
nn_model.model_.save("../artifacts/neural_network.h5")

joblib.dump({"catboost": cat_model, "rf": rf_model, "nn": nn_model}, "../artifacts/voting_ensemble.pkl")

# ---------------------------------------------------------------------
# Legacy compatibility split (not for reported performance)
# ---------------------------------------------------------------------
X_train_legacy, X_test_legacy, y_train_legacy, y_test_legacy = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)

np.save("../artifacts/X_train.npy", X_train_legacy)
np.save("../artifacts/X_test.npy", X_test_legacy)
np.save("../artifacts/y_train.npy", y_train_legacy)
np.save("../artifacts/y_test.npy", y_test_legacy)

print("Saved models, preprocessors, OOF artifacts, calibration artifacts, and legacy split artifacts.")
