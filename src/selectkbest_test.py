"""
Extended k-sensitivity experiment that evaluates the impact of feature selection
using SelectKBest with different k values.
For every value of *k* in SelectKBest we now
  • train CatBoost, Random Forest, Neural Network and a soft-voting ensemble of all three
  • report per-model Accuracy, Macro-F1 and multi-class AUC
  • compute the overall mean ± std-dev of Accuracy across all models & folds for the current *k*
Finally, the script produces a publication-ready figure illustrating how the
average accuracy (across models) varies with *k*, including error bars and annotation.
"""

# ────────────────────────────────────────────────────────────────────────────
#  IMPORTS & GLOBALS
# ────────────────────────────────────────────────────────────────────────────
import os
# Suppress TF info/warning logs about oneDNN and CPU optimisations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # only errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # disable oneDNN logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from catboost import CatBoostClassifier

# Now import TensorFlow and silence its logger
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier

# Global seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ───────────────────────────────────────────────────────────────────────────
#  PATHS & DATA LOADING
# ───────────────────────────────────────────────────────────────────────────
DATA_PATH = "../data/preprocessed_data.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Cannot locate {DATA_PATH}. Check the path.")
data = pd.read_csv(DATA_PATH)

if "PTGENDER" in data.columns:
    data["PTGENDER"] = LabelEncoder().fit_transform(data["PTGENDER"].astype(str))

initial_rows = len(data)
data = data.dropna(subset=["DX"])
print(f"Dropped {initial_rows - len(data)} rows with missing DX.")

for id_col in ["RID", "VISCODE"]:
    if id_col in data.columns:
        data.drop(columns=[id_col], inplace=True)

# ───────────────────────────────────────────────────────────────────────────
#  PREPARE FEATURES & TARGET
# ───────────────────────────────────────────────────────────────────────────
X = data.drop(columns=["DX"])
y = data["DX"].astype(int)

# ───────────────────────────────────────────────────────────────────────────
#  BASELINE MODEL DEFINITIONS
# ───────────────────────────────────────────────────────────────────────────
def build_baseline_nn(n_features, learning_rate=0.001):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(32, activation="relu"), BatchNormalization(), Dropout(0.2),
        Dense(16, activation="relu"), BatchNormalization(), Dropout(0.2),
        Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy"
    )
    return model

# Custom scorers for cross_validate
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

# ───────────────────────────────────────────────────────────────────────────
#  K LOOP – FEATURE SELECTION SENSITIVITY WITH CV
# ───────────────────────────────────────────────────────────────────────────
# Define range of k values to test (number of features to select)
K_VALUES = list(range(2, 16, 2))  # Test even numbers from 2 to 16 features
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_avg = []

for k in K_VALUES:
    print(f"\n=== K = {k} ===")
    all_acc = []
    all_f1  = []
    all_auc = []
    
    # Preprocessing steps with fixed KNN imputation (k=9 as found optimal)
    imputer = KNNImputer(n_neighbors=9, weights="distance")
    scaler  = RobustScaler()
    selector = SelectKBest(f_classif, k=k)

    # Baseline models
    base_models = {
        "CatBoost": CatBoostClassifier(random_seed=42, verbose=0),
        "RandomForest": RandomForestClassifier(random_state=42),
        "NeuralNetwork": KerasClassifier(
            model=build_baseline_nn,
            n_features=k,
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

    # Cross-validated evaluation
    for name, model in base_models.items():
        pipe = Pipeline([
            ("imputer", imputer),
            ("scaler", scaler),
            ("selector", selector),
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

    # Aggregate overall
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
    print(f"\nOVERALL     | Acc: {avg_acc:.3f} ± {std_acc:.3f} | "
          f"F1: {avg_f1:.3f} | AUC: {avg_auc:.3f}")

# ───────────────────────────────────────────────────────────────────────────
#  RESULTS — PUBLICATION-QUALITY FIGURE
# ───────────────────────────────────────────────────────────────────────────
avg_df = pd.DataFrame(results_avg)

# Best k
best_idx = avg_df["avg_accuracy"].idxmax()
best_k   = avg_df.loc[best_idx, "k"]
best_acc = avg_df.loc[best_idx, "avg_accuracy"]

os.makedirs("../Outputs", exist_ok=True)

plt.figure(figsize=(10, 8))
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

# Plot mean ± std-dev as error bars
plt.errorbar(
    avg_df["k"],
    avg_df["avg_accuracy"],
    yerr=avg_df["std_accuracy"],
    marker="o",
    linewidth=2,
    capsize=5
)

plt.xlabel("Number of Selected Features (k)", fontsize=20)
plt.ylabel("Average Accuracy across Models", fontsize=20)
plt.title("Effect of Feature Selection on Model Performance", fontsize=22)
plt.xticks(K_VALUES)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
# Note: stratified 5-fold CV; metric = accuracy (± std-dev)
plt.tight_layout()
plt.savefig("../Outputs/avg_accuracy_vs_features.png", dpi=300, bbox_inches='tight')
plt.close()
print("Publication-quality figure saved to ./Outputs/avg_accuracy_vs_features.png")
