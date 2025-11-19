import numpy as np
import joblib
from tensorflow.keras.models import load_model

def create_model():
    pass


# =============================
# LOAD DATA
# =============================
X_test = np.load("../artifacts/X_test.npy")
y_test = np.load("../artifacts/y_test.npy")

# Consistent class names (fixed order)
class_names = ["SCD", "MCI", "AD"]


# =============================
# LOAD MODELS
# =============================

# Load ensemble container (CatBoost + RF only)
models = joblib.load("../artifacts/voting_ensemble.pkl")

cat_model = models["catboost"]
rf_model = models["rf"]

# Load Neural Network separately (Keras)
nn_model = load_model("../artifacts/neural_network.h5")


# =============================
# HELPER: Run evaluation
# =============================
from eval.orchestrator import analyze_model_performance





# =============================
# 1️⃣ RANDOM FOREST
# =============================
print("\n=== Evaluating Random Forest ===")

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)

summary_rf = analyze_model_performance(
    y_true=y_test,
    y_pred=y_pred_rf,
    y_prob=y_prob_rf,
    model=rf_model,
    X_test=X_test,
    class_names=class_names,
    model_name="RandomForest",
)


# =============================
# 2️⃣ NEURAL NETWORK (Keras)
# =============================
print("\n=== Evaluating Neural Network ===")

y_prob_nn = nn_model.predict(X_test, verbose=0)
y_pred_nn = np.argmax(y_prob_nn, axis=1)

summary_nn = analyze_model_performance(
    y_true=y_test,
    y_pred=y_pred_nn,
    y_prob=y_prob_nn,
    model=nn_model,
    X_test=X_test,
    class_names=class_names,
    model_name="NeuralNet",
)



y_pred_cat = cat_model.predict(X_test)
y_prob_cat = cat_model.predict_proba(X_test)




# =============================
# 4️⃣ SOFT VOTING ENSEMBLE
# =============================
print("\n=== Evaluating Soft Voting Ensemble ===")

y_prob_ensemble = (y_prob_cat + y_prob_rf + y_prob_nn) / 3
y_pred_ensemble = np.argmax(y_prob_ensemble, axis=1)

summary_ensemble = analyze_model_performance(
    y_true=y_test,
    y_pred=y_pred_ensemble,
    y_prob=y_prob_ensemble,
    model=cat_model,   # any model works for SHAP
    X_test=X_test,
    class_names=class_names,
    model_name="Ensemble",
)

# =============================
# 3️⃣ CATBOOST
# =============================
print("\n=== Evaluating CatBoost ===")

summary_cat = analyze_model_performance(
    y_true=y_test,
    y_pred=y_pred_cat,
    y_prob=y_prob_cat,
    model=cat_model,
    X_test=X_test,
    class_names=class_names,
    model_name="CatBoost",
)

print("\n=== ALL MODELS EVALUATED SUCCESSFULLY ===")
print("Reports saved to /reports/")
