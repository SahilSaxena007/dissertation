"""
Fixed Extended k-sensitivity experiment — publication-ready version.
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier

np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────
# Data
# ─────────────────────────────
DATA_PATH = "../data/preprocessed_data.csv"
data = pd.read_csv(DATA_PATH)
if "PTGENDER" in data.columns:
    data["PTGENDER"] = LabelEncoder().fit_transform(data["PTGENDER"].astype(str))
data.dropna(subset=["DX"], inplace=True)
data.drop(columns=[c for c in ["RID","VISCODE"] if c in data.columns], inplace=True)
X, y = data.drop(columns=["DX"]), data["DX"].astype(int)

# ─────────────────────────────
# Models
# ─────────────────────────────
def build_baseline_nn(learning_rate=0.001):
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(32, activation="relu"), BatchNormalization(), Dropout(0.2),
        Dense(16, activation="relu"), BatchNormalization(), Dropout(0.2),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Custom scorer for AUC (fixes 'needs_proba' bug)
def multiclass_auc(y_true, y_pred_proba):
    try:
        return roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
    except ValueError:
        return np.nan

def auc_scorer(estimator, X, y):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return multiclass_auc(y, proba)
    return np.nan

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1_macro": make_scorer(f1_score, average="macro"),
    "roc_auc_ovr": auc_scorer
}

# ─────────────────────────────
# Cross-validation loop
# ─────────────────────────────
K_VALUES = [3, 5, 7, 9, 11, 13]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_avg = []

for k in K_VALUES:
    print(f"\n=== K = {k} ===")
    imputer = KNNImputer(n_neighbors=k, weights="distance")
    scaler  = RobustScaler()

    base_models = {
        "CatBoost": CatBoostClassifier(random_seed=42, verbose=0),
        "RandomForest": RandomForestClassifier(random_state=42),
        "NeuralNetwork": KerasClassifier(
            model=build_baseline_nn,
            learning_rate=0.001,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6)
            ]
        )
    }

    all_acc, all_f1, all_auc = [], [], []

    for name, model in base_models.items():
        pipe = Pipeline([
            ("imputer", imputer),
            ("scaler", scaler),
            ("model", model)
        ])
        cv_res = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        mean_acc = np.nanmean(cv_res["test_accuracy"])
        std_acc  = np.nanstd(cv_res["test_accuracy"])
        mean_f1  = np.nanmean(cv_res["test_f1_macro"])
        mean_auc = np.nanmean(cv_res["test_roc_auc_ovr"])

        all_acc.extend(cv_res["test_accuracy"])
        all_f1.extend(cv_res["test_f1_macro"])
        all_auc.extend(cv_res["test_roc_auc_ovr"])

        print(f"{name:15}| Acc: {mean_acc:.3f} ± {std_acc:.3f} | "
              f"F1: {mean_f1:.3f} | AUC: {mean_auc:.3f}")

    avg_acc = np.nanmean(all_acc)
    std_acc = np.nanstd(all_acc)
    avg_f1  = np.nanmean(all_f1)
    avg_auc = np.nanmean(all_auc)
    results_avg.append({
        "k": k,
        "avg_accuracy": avg_acc,
        "std_accuracy": std_acc,
        "avg_f1_macro": avg_f1,
        "avg_roc_auc_ovr": avg_auc
    })
    print(f"OVERALL     | Acc: {avg_acc:.3f} ± {std_acc:.3f} | "
          f"F1: {avg_f1:.3f} | AUC: {avg_auc:.3f}")

# ─────────────────────────────
# Plot
# ─────────────────────────────
os.makedirs("../Outputs", exist_ok=True)
avg_df = pd.DataFrame(results_avg)
best_idx = avg_df["avg_accuracy"].idxmax()
best_k   = avg_df.loc[best_idx, "k"]
best_acc = avg_df.loc[best_idx, "avg_accuracy"]

plt.figure(figsize=(10,8))
plt.errorbar(avg_df["k"], avg_df["avg_accuracy"],
             yerr=avg_df["std_accuracy"], fmt="o-", capsize=5, lw=2)
plt.annotate(f"Best k={best_k}",
             xy=(best_k, best_acc),
             xytext=(best_k, best_acc+0.01),
             arrowprops=dict(arrowstyle="->", lw=2))
plt.xlabel("k (KNN Imputer)")
plt.ylabel("Average Accuracy (across models)")
plt.title("Effect of Imputation Parameter k on Model Performance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../Outputs/avg_accuracy_vs_k.png", dpi=300)
plt.close()
print("✅ Figure saved to ./Outputs/avg_accuracy_vs_k.png")
