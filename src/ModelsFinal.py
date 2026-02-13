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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_cat = np.zeros((n_samples, n_classes), dtype=float)
oof_rf = np.zeros((n_samples, n_classes), dtype=float)
oof_nn = np.zeros((n_samples, n_classes), dtype=float)
oof_selected_X = np.zeros((n_samples, 12), dtype=float)
oof_sample_ids = np.arange(n_samples, dtype=int)

print("Generating out-of-fold (OOF) predictions with StratifiedKFold (k=5)...")

for fold_id, (train_idx, val_idx) in enumerate(cv.split(X_all, y), start=1):
    print(f"Fold {fold_id}/5")

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

    oof_cat[val_idx] = cat_model_fold.predict_proba(X_val_scaled)
    oof_rf[val_idx] = rf_model_fold.predict_proba(X_val_scaled)
    oof_nn[val_idx] = nn_model_fold.predict_proba(X_val_scaled)
    oof_selected_X[val_idx] = X_val_scaled

oof_ens = (oof_cat + oof_rf + oof_nn) / 3.0

metrics_table = pd.DataFrame(
    [
        summarize_metrics("CatBoost_OOF", y, oof_cat),
        summarize_metrics("RandomForest_OOF", y, oof_rf),
        summarize_metrics("NeuralNetwork_OOF", y, oof_nn),
        summarize_metrics("VotingEnsemble_OOF", y, oof_ens),
    ]
)
metrics_table.to_csv("../Outputs/overall_metrics.csv", index=False)
print("Saved leak-free OOF metrics to ../Outputs/overall_metrics.csv")

# Save OOF artifacts for evaluation and escalation
np.save("../artifacts/oof_cat_proba.npy", oof_cat)
np.save("../artifacts/oof_rf_proba.npy", oof_rf)
np.save("../artifacts/oof_nn_proba.npy", oof_nn)
np.save("../artifacts/oof_ens_proba.npy", oof_ens)
np.save("../artifacts/oof_pred_ens.npy", np.argmax(oof_ens, axis=1))
np.save("../artifacts/y_oof.npy", y)
np.save("../artifacts/X_oof.npy", oof_selected_X)
np.save("../artifacts/oof_sample_ids.npy", oof_sample_ids)

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

print("Saved models, preprocessors, OOF artifacts, and legacy split artifacts.")
