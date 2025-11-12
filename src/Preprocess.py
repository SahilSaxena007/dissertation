#!/usr/bin/env python3
"""
Preprocessing pipeline for merged_data.csv:
- Sanity checks on loaded data
- String-to-numeric cleaning of biomarker columns
- Feature engineering (biomarker ratios)
- Categorical encoding
- Outlier detection & capping (IQR method)
- Summary tables (3.1 & 3.2)
- Boxplots before/after outlier treatment
- Export preprocessed_data.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------
# Suppress warnings & set seed
# ------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)

# ------------------------------------------------------
# Academic‐quality plotting settings
# ------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.5)
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# ====================================================================
# 1) LOAD DATA & SANITY CHECKS
# ====================================================================
data = pd.read_csv('../data/merged_data.csv')

# Sanity checks
print("=== Sanity Check ===")
print("Columns:\n", data.columns.tolist())
print("\nData types:\n", data.dtypes)
print("\nFirst five rows:\n", data.head(), "\n")

# ------------------------------------------------------
# 2) Remove non-predictive identifiers
# ------------------------------------------------------
for id_col in ['RID', 'VISCODE']:
    if id_col in data.columns:
        data.drop(columns=[id_col], inplace=True)

# ====================================================================
# 3) STRING-TO-NUMERIC CLEANING OF BIOMARKERS
# ====================================================================
for col in ['TAU', 'AB4240', 'PTAU', 'ABETA', 'PLASMA_NFL', 'PLASMATAU']:
    if col in data.columns:
        # replace censoring symbols with numeric boundaries
        data[col] = data[col].replace({
            '>1300': '1300',
            '>1700': '1700',
            '<80':   '80',
            '<200':  '200'
        })
        data[col] = pd.to_numeric(data[col], errors='coerce')

# ====================================================================
# 4) FEATURE ENGINEERING: BIOMARKER RATIOS
# ====================================================================
ratios = [
    ('TAU',       'PTAU',        'TAU_PTAU_ratio'),
    ('PLASMATAU','PLASMA_NFL',   'PLASMA_ratio'),
    ('AB4240',    'TAU',         'AB4240_TAU_ratio'),
    ('ABETA',     'TAU',         'ABETA_TAU_ratio'),
    ('PTAU',      'ABETA',       'PTAU_ABETA_ratio')
]
for a, b, name in ratios:
    if {a, b}.issubset(data.columns):
        data[name] = data[a] / data[b]

# ====================================================================
# 5) CATEGORICAL ENCODING
# ====================================================================
if 'PTGENDER' in data.columns:
    data['PTGENDER'] = LabelEncoder().fit_transform(data['PTGENDER'])
if 'DX' in data.columns:
    data['DX'] = data['DX'].map({'SCD': 0, 'MCI': 1, 'AD': 2})

# ====================================================================
# Prepare numeric columns list for downstream steps
# ====================================================================
numeric_cols = [
    'AGE', 'PTEDUCAT', 'MMSE',
    'TAU', 'AB4240', 'PTAU', 'ABETA',
    'PLASMA_NFL', 'PLASMATAU',
    'TAU_PTAU_ratio', 'PLASMA_ratio',
    'AB4240_TAU_ratio', 'ABETA_TAU_ratio',
    'PTAU_ABETA_ratio'
]
numeric_cols = [c for c in numeric_cols if c in data.columns]

# ====================================================================
# 6) TABLE 3.1: Filtered Dataset Fields
# ====================================================================
table_3_1 = pd.DataFrame({
    'Field':                data.columns,
    'Data Type':            data.dtypes.astype(str),
    'Percentage Available': 100 * (1 - data.isnull().mean())
})
table_3_1.sort_values('Percentage Available', ascending=False, inplace=True)
table_3_1.to_csv('../Outputs/Table_3_1_filtered_dataset_fields.csv', index=False)

# ====================================================================
# 7) TABLE 3.2: Statistical Analysis of Numeric Features
# ====================================================================
stats = data[numeric_cols].describe().T
stats['skew']     = data[numeric_cols].skew()
stats['kurtosis'] = data[numeric_cols].kurt()
stats.to_csv('../Outputs/Table_3_2_statistical_analysis.csv')

# ====================================================================
# 8) OUTLIER DETECTION & TREATMENT (IQR METHOD)
# ====================================================================
def treat_outliers_iqr(df, cols):
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower,
                           np.where(df[col] > upper, upper, df[col]))

# Boxplots before capping
plt.figure(figsize=(15, 8))
sns.boxplot(data=data[numeric_cols], color='steelblue')
plt.title('Feature Distributions Before Outlier Treatment')
plt.ylabel('Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../Outputs/boxplots_before_outlier_treatment.png', dpi=300, bbox_inches='tight')
plt.close()

# Cap outliers
treat_outliers_iqr(data, numeric_cols)

# Boxplots after capping
plt.figure(figsize=(15, 8))
sns.boxplot(data=data[numeric_cols], color='steelblue')
plt.title('Feature Distributions After Outlier Treatment')
plt.ylabel('Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../Outputs/boxplots_after_outlier_treatment.png', dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# 9) SAVE PREPROCESSED DATA
# ====================================================================
data.to_csv('../data/preprocessed_data.csv', index=False)
print("✅ Preprocessing complete. Tables and figures saved, preprocessed_data.csv exported.")
