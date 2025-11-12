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

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logging
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

# Drop them
data = data.dropna(subset=['DX'])

# Now define features and labels
X = data.drop(columns=['DX'])
y = data['DX']

# Select only numeric columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X = X[numeric_cols]

# Split data (used for initial training and hyperparameter tuning)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit imputer on training data only
imputer = KNNImputer(n_neighbors=9, weights='distance')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Feature Selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(mutual_info_classif, k=12)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()]

# Train Random Forest Classifier (for feature importance)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_selected, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_
importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print feature importances
print("\n=== Feature Importances ===")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

# Sort and get top 12 features
importance_df_top12 = importance_df.head(12)

# Create barplot with improved styling
sns.barplot(x='Importance', y='Feature', data=importance_df_top12, palette='viridis')

# Customize the plot
plt.title("Feature Importance (Random Forest)", pad=20, fontweight='bold')
plt.xlabel("Relative Importance", labelpad=15)
plt.ylabel("Feature", labelpad=15)

# Add grid
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Adjust layout to prevent cutoff
plt.tight_layout()

# Save with high DPI
os.makedirs('../Outputs', exist_ok=True)
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

# Fit scaler on training data only
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test_selected)

# === CatBoost Model Hyperparameter Tuning using CV ===
catboost_param_grid = {
    'iterations': [300, 500, 700],
    'learning_rate': [0.001, 0.005, 0.01],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [2, 3, 4],
    'class_weights': [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.1, 2.0, 1.0], [1.0, 3.0, 1.0]]
}

def macro_f1_mci_recall(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    recall_mci = recall_score(y_true, y_pred, average=None)[1]  # class 1 = MCI
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
best_params_catboost = catboost_search.best_params_
print("\nBest CatBoost Parameters:")
print(best_params_catboost)
print(f"Best Custom Score: {catboost_search.best_score_:.4f}")

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Calculate per-class metrics
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
        specificity_i = TN / (TN + FP)  # True Negative Rate
        class_accuracy.append(accuracy_i)
        class_specificity.append(specificity_i)
    class_accuracy = np.array(class_accuracy)
    class_specificity = np.array(class_specificity)
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_recall = np.mean(recall)
    overall_specificity = np.mean(class_specificity)
    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    class_counts = np.bincount(y_true)
    print(f"\n=== {model_name} Results ===")
    print("-"*50)
    print(f"Class Distribution:")
    if len(class_counts) > 0:
        print(f"SCD: {class_counts[0]} samples")
    if len(class_counts) > 1:
        print(f"MCI: {class_counts[1]} samples")
    if len(class_counts) > 2:
        print(f"AD: {class_counts[2]} samples")
    print("-"*50)
    print(f"{'':<10}{'SCD':<12}{'MCI':<12}{'AD':<12}")
    print(f"{'Recall':<10}{recall[0]:<12.2f}{recall[1]:<12.2f}{recall[2]:<12.2f}")
    print(f"{'Precision':<10}{precision[0]:<12.2f}{precision[1]:<12.2f}{precision[2]:<12.2f}")
    print(f"{'F1 Score':<10}{f1[0]:<12.2f}{f1[1]:<12.2f}{f1[2]:<12.2f}")
    print(f"{'Per-Class':<10}{class_accuracy[0]:<12.2f}{class_accuracy[1]:<12.2f}{class_accuracy[2]:<12.2f}")
    print(f"{'Specificity':<10}{class_specificity[0]:<12.2f}{class_specificity[1]:<12.2f}{class_specificity[2]:<12.2f}")
    print("-"*50)
    print(f"Overall Accuracy: {overall_accuracy*100:.1f}%")
    print(f"Overall Recall: {overall_recall*100:.1f}%")
    print(f"Overall Specificity: {overall_specificity*100:.1f}%")
    print(f"Overall AUC: {auc:.3f}")
    print("-"*50)
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

# === Random Forest Model Hyperparameter Tuning using CV ===
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
print(f"Best Custom Score: {rf_random.best_score_:.4f}")

# === Neural Network Model Hyperparameter Tuning using CV ===
def create_model(units_1=64, units_2=32, units_3=16, dropout_1=0.4, dropout_2=0.2, learning_rate=0.001):
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

# Create KerasClassifier with fixed model parameters
nn_model = KerasClassifier(
    model=create_model,
    random_state=42,
    units_1=64,
    units_2=32,
    units_3=16,
    dropout_1=0.3,
    dropout_2=0.2,
    learning_rate=0.001,
    epochs=200,
    batch_size=32,
    verbose=0,
    callbacks=[
        EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5, min_lr=1e-6)
    ]
)

# Define parameter grid for Neural Network
param_grid = {
    'model__units_1': [32, 64, 128],
    'model__units_2': [16, 32, 64],
    'model__units_3': [16, 32],
    'model__dropout_1': [0.1, 0.3, 0.5],
    'model__dropout_2': [0.1, 0.3, 0.5],
    'model__learning_rate': [0.001, 0.01, 0.05]
}

nn_random = RandomizedSearchCV(
    nn_model,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring=custom_scorer,
    random_state=42,
    n_jobs=-1
)

nn_random.fit(X_train_scaled, y_train_balanced)
nn_model = nn_random.best_estimator_

print("\nBest Neural Network Parameters:")
print(nn_random.best_params_)
print(f"Best Custom Score: {nn_random.best_score_:.4f}")

# === Voting Ensemble (manual soft voting workaround) ===
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_balanced)
y_test_enc = le.transform(y_test)

# Predict probabilities for each model
cat_proba = best_model_catboost.predict_proba(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)
nn_proba = nn_model.predict_proba(X_test_scaled)

# Average the probabilities (soft voting)
avg_proba = (cat_proba + rf_proba + nn_proba) / 3.0
y_pred = np.argmax(avg_proba, axis=1)

# Evaluate ensemble
voting_metrics = evaluate_model(
    y_test_enc,
    y_pred,
    avg_proba,
    "Voting Ensemble"
)

# Evaluate all models on the held-out test set
print("\n=== Model Evaluation on Held-out Test Set ===")

# Evaluate CatBoost
catboost_metrics = evaluate_model(
    y_test_enc,
    best_model_catboost.predict(X_test_scaled),
    best_model_catboost.predict_proba(X_test_scaled),
    "CatBoost"
)

# Evaluate Random Forest
rf_metrics = evaluate_model(
    y_test_enc,
    rf_model.predict(X_test_scaled),
    rf_model.predict_proba(X_test_scaled),
    "Random Forest"
)

# Evaluate Neural Network
nn_preds = nn_model.predict(X_test_scaled)
nn_metrics = evaluate_model(
    y_test_enc,
    nn_preds,
    nn_model.predict_proba(X_test_scaled),
    "Neural Network"
)

# Print model comparison summary for held-out test set
print("\n=== Model Comparison Summary (Held-out Test Set) ===")
print("-"*120)
print(f"{'Model':<15}{'Accuracy':<10}{'AUC':<10}{'Avg F1':<10}{'Avg Recall':<12}{'Recall by Class (SCD/MCI/AD)':<35}")
print("-"*120)
print(f"{'CatBoost':<15}{catboost_metrics['accuracy']:.3f}{'':>5}{catboost_metrics['auc']:.3f}{'':>5}{np.mean(catboost_metrics['f1']):.3f}{'':>4}{catboost_metrics['overall_recall']:.3f}{'':>6}{catboost_metrics['recall'][0]:.2f}/{catboost_metrics['recall'][1]:.2f}/{catboost_metrics['recall'][2]:.2f}")
print(f"{'Random Forest':<15}{rf_metrics['accuracy']:.3f}{'':>5}{rf_metrics['auc']:.3f}{'':>5}{np.mean(rf_metrics['f1']):.3f}{'':>4}{rf_metrics['overall_recall']:.3f}{'':>6}{rf_metrics['recall'][0]:.2f}/{rf_metrics['recall'][1]:.2f}/{rf_metrics['recall'][2]:.2f}")
print(f"{'Neural Network':<15}{nn_metrics['accuracy']:.3f}{'':>5}{nn_metrics['auc']:.3f}{'':>5}{np.mean(nn_metrics['f1']):.3f}{'':>4}{nn_metrics['overall_recall']:.3f}{'':>6}{nn_metrics['recall'][0]:.2f}/{nn_metrics['recall'][1]:.2f}/{nn_metrics['recall'][2]:.2f}")
print(f"{'Voting Ens.':<15}{voting_metrics['accuracy']:.3f}{'':>5}{voting_metrics['auc']:.3f}{'':>5}{np.mean(voting_metrics['f1']):.3f}{'':>4}{voting_metrics['overall_recall']:.3f}{'':>6}{voting_metrics['recall'][0]:.2f}/{voting_metrics['recall'][1]:.2f}/{voting_metrics['recall'][2]:.2f}")
print("-"*120)

# After all model evaluations
# Save overall metrics table
models = ['CatBoost', 'Random Forest', 'Neural Network', 'Voting Ensemble']
metrics_list = [catboost_metrics, rf_metrics, nn_metrics, voting_metrics]
overall_metrics = pd.DataFrame({
    'Model': models,
    'Accuracy': [m['accuracy'] for m in metrics_list],
    'AUC': [m['auc'] for m in metrics_list],
    'F1 Score': [np.mean(m['f1']) for m in metrics_list],
    'Recall': [m['overall_recall'] for m in metrics_list],
    'Recall by Class (SCD)': [m['recall'][0] for m in metrics_list],
    'Recall by Class (MCI)': [m['recall'][1] for m in metrics_list],
    'Recall by Class (AD)': [m['recall'][2] for m in metrics_list]
})
overall_metrics.to_csv('../Outputs/overall_metrics.csv', index=False)

# Save per-class metrics table
per_class_metrics = pd.DataFrame({
    'Model': np.repeat(models, 3),
    'Class': ['SCD', 'MCI', 'AD'] * 4,
    'Recall': np.concatenate([m['recall'] for m in metrics_list]),
    'Precision': np.concatenate([m['precision'] for m in metrics_list]),
    'F1 Score': np.concatenate([m['f1'] for m in metrics_list])
})
per_class_metrics.to_csv('../Outputs/per_class_metrics.csv', index=False)

# Save confusion matrices
for model, m in zip(models, metrics_list):
    pd.DataFrame(m['conf_matrix'], columns=['SCD', 'MCI', 'AD'], index=['SCD', 'MCI', 'AD']).to_csv(f'../Outputs/confusion_matrix_{model.replace(" ", "_").lower()}.csv')
