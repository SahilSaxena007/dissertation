import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix, precision_score, roc_auc_score
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
import os
from scikeras.wrappers import KerasClassifier
import warnings
from sklearn.metrics import make_scorer
import joblib
import json

os.makedirs("../artifacts", exist_ok=True)
os.makedirs("../Outputs", exist_ok=True)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

os.environ['PYTHONHASHSEED'] = '42'
tf.random.set_seed(42)
np.random.seed(42)


# Load the dataset
data = pd.read_csv('../data/preprocessed_data.csv')

# Quick sanity checks
data.columns.tolist()
data.head()

# How many missing?
n_missing = data['DX'].isna().sum()
print(f"Found {n_missing} rows with missing DX; dropping these.")

# Drop rows with missing DX
data = data.dropna(subset=['DX'])

# Define features and labels
X = data.drop(columns=['DX'])
y = data['DX']

# Select numeric columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X = X[numeric_cols]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit imputer
imputer = KNNImputer(n_neighbors=9, weights='distance')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Feature Selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=12)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()]

# === SAVE PREPROCESSING ARTIFACTS ===
joblib.dump(imputer, "../artifacts/imputer.pkl")
joblib.dump(selector, "../artifacts/select_k_best.pkl")

# Scaler will be saved later (after fitting)
with open("../artifacts/selected_features.json", "w") as f:
    json.dump(list(selected_features), f, indent=4)

print("✔ Saved preprocessing feature selector artifacts.")

# Train Random Forest for feature importance
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_selected, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_
importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(12), palette='viridis')
plt.title("Feature Importance (Random Forest)", pad=20)
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('../Outputs/feature_importance.png', dpi=600, bbox_inches='tight')
plt.close()

# Model Evaluation on initial split
y_pred = rf_classifier.predict(X_test_selected)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

selected_features = X_train.columns[selector.get_support()].tolist()
print("\n=== Selected Features ===")
print(selected_features)

# === Preprocessing Pipeline ===
X_train_balanced, y_train_balanced = X_train_selected, y_train

# Fit scaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test_selected)

# SAVE SCALER
joblib.dump(scaler, "../artifacts/scaler.pkl")
print("✔ Saved scaler.")

# === SAVE TRAIN/TEST SPLIT ===
np.save("../artifacts/X_train.npy", X_train_scaled)
np.save("../artifacts/X_test.npy", X_test_scaled)
np.save("../artifacts/y_train.npy", y_train_balanced)
np.save("../artifacts/y_test.npy", y_test)

print("✔ Saved train/test split.")

# === CatBoost Hyperparameter Search ===
catboost_param_grid = {
    'iterations': [300, 500, 700],
    'learning_rate': [0.001, 0.005, 0.01],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [2, 3, 4],
    'class_weights': [[1.0,1.0,1.0], [1.0,2.0,1.0], [1.1,2.0,1.0], [1.0,3.0,1.0]]
}

def macro_f1_mci_recall(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    recall_mci = recall_score(y_true, y_pred, average=None)[1]
    return 0.9 * macro_f1 + 0.1 * recall_mci

custom_scorer = make_scorer(macro_f1_mci_recall, greater_is_better=True)

catboost_search = RandomizedSearchCV(
    CatBoostClassifier(
        random_seed=42,
        eval_metric='TotalF1',
        loss_function='MultiClass',
        verbose=0,
        bootstrap_type='Bayesian',
        thread_count=1,
        devices='CPU'
    ),
    param_distributions=catboost_param_grid,
    n_iter=20,
    cv=5,
    scoring=custom_scorer,
    random_state=42,
    n_jobs=-1
)

catboost_search.fit(X_train_scaled, y_train_balanced)
best_model_catboost = catboost_search.best_estimator_

print("\nBest CatBoost Parameters:")
print(catboost_search.best_params_)

# === RANDOM FOREST GRID SEARCH ===
rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, "sqrt", "log2"],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=20,
    cv=5,
    scoring=custom_scorer,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train_scaled, y_train_balanced)
rf_model = rf_random.best_estimator_

print("\nBest Random Forest Parameters:")
print(rf_random.best_params_)

# === NEURAL NETWORK GRID SEARCH ===
def create_model(units_1=64, units_2=32, units_3=16,
                 dropout_1=0.4, dropout_2=0.2,
                 learning_rate=0.001):
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(units_1, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_1),
        Dense(units_2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_2),
        Dense(units_3, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

nn_model = KerasClassifier(
    model=create_model, random_state=42,
    units_1=64, units_2=32, units_3=16,
    dropout_1=0.3, dropout_2=0.2,
    learning_rate=0.001,
    epochs=200, batch_size=32, verbose=0,
    callbacks=[
        EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5, min_lr=1e-6)
    ]
)

nn_grid = {
    'model__units_1': [32, 64, 128],
    'model__units_2': [16, 32, 64],
    'model__units_3': [16, 32],
    'model__dropout_1': [0.1, 0.3, 0.5],
    'model__dropout_2': [0.1, 0.3, 0.5],
    'model__learning_rate': [0.001, 0.01, 0.05]
}

nn_random = RandomizedSearchCV(
    nn_model,
    param_distributions=nn_grid,
    n_iter=20,
    cv=3,
    scoring=custom_scorer,
    random_state=42,
    n_jobs=-1
)

nn_random.fit(X_train_scaled, y_train_balanced)
nn_model = nn_random.best_estimator_

print("\nBest NN Params:")
print(nn_random.best_params_)

# === SAVE TRAINED MODELS ===
joblib.dump(best_model_catboost, "../artifacts/best_model_catboost.pkl")
joblib.dump(rf_model, "../artifacts/random_forest.pkl")
nn_model.model_.save("../artifacts/neural_network.h5")

joblib.dump({
    "catboost": best_model_catboost,
    "rf": rf_model,
    "nn": nn_model
}, "../artifacts/voting_ensemble.pkl")

print("✔ Saved all models.")

# === VOTING ENSEMBLE ===
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_balanced)
y_test_enc = le.transform(y_test)

cat_proba = best_model_catboost.predict_proba(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)
nn_proba = nn_model.predict_proba(X_test_scaled)

avg_proba = (cat_proba + rf_proba + nn_proba) / 3.0
y_pred = np.argmax(avg_proba, axis=1)

# Evaluation function preserved, outputs CSVs later
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_accuracy = []
    class_specificity = []
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        mask = np.ones(conf_matrix.shape, dtype=bool)
        mask[i, :] = False
        mask[:, i] = False
        TN = conf_matrix[mask].sum()
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        accuracy_i = (TP + TN) / (TP + TN + FP + FN)
        specificity_i = TN / (TN + FP)
        class_accuracy.append(accuracy_i)
        class_specificity.append(specificity_i)

    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_recall = np.mean(recall)
    overall_specificity = np.mean(class_specificity)
    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')

    print(f"\n=== {model_name} Results ===")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"AUC: {auc:.3f}")

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'class_accuracy': class_accuracy,
        'class_specificity': class_specificity,
        'accuracy': overall_accuracy,
        'overall_recall': overall_recall,
        'overall_specificity': overall_specificity,
        'auc': auc,
        'conf_matrix': conf_matrix
    }

# Evaluate models
voting_metrics = evaluate_model(y_test_enc, y_pred, avg_proba, "Voting Ensemble")
catmetrics = evaluate_model(
    y_test_enc,
    best_model_catboost.predict(X_test_scaled),
    best_model_catboost.predict_proba(X_test_scaled),
    "CatBoost"
)
rfmetrics = evaluate_model(
    y_test_enc,
    rf_model.predict(X_test_scaled),
    rf_model.predict_proba(X_test_scaled),
    "Random Forest"
)
nnmetrics = evaluate_model(
    y_test_enc,
    nn_model.predict(X_test_scaled),
    nn_model.predict_proba(X_test_scaled),
    "Neural Network"
)

print("✔ All evaluations complete.")

# Save comparisons CSV
overall_metrics = pd.DataFrame({
    'Model': ['CatBoost', 'Random Forest', 'Neural Network', 'Voting Ensemble'],
    'Accuracy': [catmetrics['accuracy'], rfmetrics['accuracy'], nnmetrics['accuracy'], voting_metrics['accuracy']],
    'AUC': [catmetrics['auc'], rfmetrics['auc'], nnmetrics['auc'], voting_metrics['auc']],
    'F1 Score': [np.mean(catmetrics['f1']), np.mean(rfmetrics['f1']), np.mean(nnmetrics['f1']), np.mean(voting_metrics['f1'])]
})
overall_metrics.to_csv('../Outputs/overall_metrics.csv', index=False)
print("✔ Saved overall metrics.")
