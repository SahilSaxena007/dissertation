#!/usr/bin/env python3
"""
Comprehensive data QC and correlation analysis on merged_data.csv.

Outputs:
    - missing_data.png              (missingness bar chart)
    - feature_boxplots.png          (boxplots of standardized features)
    - correlation_bar_chart.pdf     (Pearson correlation bar chart)
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ------------------------------------------------------
# High-quality academic plotting settings
# ------------------------------------------------------
plt.rcParams.update({
    'font.size': 20,
    'figure.dpi': 300,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

# ------------------------------------------------------
# 1) Load data and define features
# ------------------------------------------------------
df = pd.read_csv('../data/merged_data.csv')

features = [
    'ABETA', 'TAU', 'PTAU', 'AB4240',
    'PLASMATAU', 'PLASMA_NFL',
    'AGE', 'PTEDUCAT', 'MMSE'
]

# Ensure numeric dtype for features
for feat in features:
    df[feat] = pd.to_numeric(df[feat], errors='coerce')

# ------------------------------------------------------
# 2) Missing-value analysis
# ------------------------------------------------------
missing_counts = df[features].isnull().sum()
missing_perc = (missing_counts / len(df)) * 100
available_perc = 100 - missing_perc

print("\nTable 4.1: Data Availability Summary")
print("Column\tData Type\tPercentage Available")
for feat in features:
    dtype = df[feat].dtype
    print(f"{feat}\t{dtype}\t{available_perc[feat]:.2f}")
print("\nTable 4.1 displays the filtered dataset with columns, data types, and percentage of available data.\n")

missing_df = pd.DataFrame({
    'n_missing': missing_counts,
    'percent_missing': missing_perc
}).loc[features]

print("Missing Data Summary for Key Features (n_missing and percent_missing):")
print(missing_df)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(missing_df.index, missing_df['percent_missing'],
       color='lightgrey', edgecolor='black')
ax.set_ylim(0, 100)
ax.set_ylabel('Missingness (%)')
ax.set_xlabel('Feature')
plt.xticks(rotation=45, ha='right')
for i, (feat, row) in enumerate(missing_df.iterrows()):
    ax.text(i, row['percent_missing'] + 2,
            f"{row['percent_missing']:.1f}%\n({int(row['n_missing'])})",
            ha='center', va='bottom', fontsize=14)
fig.tight_layout()
fig.savefig('../Outputs/missing_data.png')
plt.close(fig)

# ------------------------------------------------------
# 3) Outlier detection via IQR
# ------------------------------------------------------
stats = []
outliers = []
for feat in features:
    series = df[feat].dropna()
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_lower = (series < lower).sum()
    n_upper = (series > upper).sum()
    stats.append({
        'feature': feat,
        'min': series.min(),
        'Q1': Q1,
        'median': series.median(),
        'Q3': Q3,
        'max': series.max(),
        'IQR': IQR
    })
    outliers.append({
        'feature': feat,
        'n_lower_outliers': n_lower,
        'n_upper_outliers': n_upper,
        'percent_outliers': (n_lower + n_upper) / len(series) * 100
    })

stats_df = pd.DataFrame(stats).set_index('feature')
outliers_df = pd.DataFrame(outliers).set_index('feature')

print("\nDescriptive Statistics (min, Q1, median, Q3, max, IQR):")
print(stats_df)
print("\nOutlier Counts (lower, upper, percent_outliers):")
print(outliers_df)

# Boxplots (z-score standardized)
df_z = df[features].apply(lambda x: (x - x.mean()) / x.std())

fig2, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for ax, feat in zip(axes, features):
    ax.boxplot(
        df_z[feat].dropna(),
        boxprops=dict(color='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black')
    )
    ax.set_title(feat)
    ax.set_ylabel('Z-score')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.tick_params(axis='x', labelrotation=45)
fig2.tight_layout()
fig2.savefig('../Outputs/Outliers.png')
plt.close(fig2)

# ------------------------------------------------------
# 4) Pearson correlation with diagnostic category (DX)
# ------------------------------------------------------
# Ensure DX is present and numeric; map categories if needed
if 'DX' not in df.columns:
    raise KeyError("Column 'DX' not found in merged_data.csv")

# Convert DX to numeric codes if not already numeric
if df['DX'].dtype == object or not np.issubdtype(df['DX'].dtype, np.number):
    mapping = {'SCD': 0, 'MCI': 1, 'AD': 2,}
    df['DX'] = df['DX'].map(mapping)

# Compute Pearson r and p-value for each feature vs DX
results = []
for feat in features:
    subset = df[[feat, 'DX']].dropna()
    if subset.empty:
        continue
    r, pval = pearsonr(subset[feat], subset['DX'])
    results.append({'feature': feat, 'r': r, 'p': pval})

corr_df = pd.DataFrame(results)
corr_df['abs_r'] = corr_df['r'].abs()
corr_df.sort_values('abs_r', ascending=False, inplace=True)

# Plot horizontal bar chart
fig3, ax = plt.subplots(figsize=(12, 8))
ax.invert_yaxis()
bars = ax.barh(corr_df['feature'], corr_df['r'], color='0.3', edgecolor='0')
ax.axvline(0, color='0', linewidth=1)

ax.set_xlabel('Pearson correlation coefficient (r)')
ax.set_ylabel('Feature')
ax.set_title(f'Feature Correlations with Diagnostic Category (n={int(df["DX"].notna().sum())})')

# Adjust margins
ax.margins(x=0.05)
plt.subplots_adjust(left=0.5, right=0.95, top=0.9, bottom=0.1)

# Annotation offsets
axis_min, axis_max = ax.get_xlim()
total_width = axis_max - axis_min
inside_off = 0.025 * total_width
outside_off = 0.025 * total_width
min_inside = 0.12

for bar, row in zip(bars, corr_df.itertuples()):
    x = bar.get_width()
    star = '**' if row.p < 0.01 else '*' if row.p < 0.05 else ''
    # Choose placement
    if abs(x) < min_inside:
        text_x = x + outside_off if x >= 0 else x - outside_off
        ha = 'left' if x >= 0 else 'right'
        color = 'black'
    else:
        text_x = x - inside_off if x >= 0 else x + inside_off
        ha = 'right' if x >= 0 else 'left'
        color = 'white'
    ax.text(
        text_x,
        bar.get_y() + bar.get_height() / 2,
        f'{x:.2f}{star}',
        va='center',
        ha=ha,
        fontsize=16,
        fontweight='bold',
        color=color,
        clip_on=False
    )

fig3.tight_layout()
fig3.savefig('../Outputs/correlation_bar_chart.pdf', bbox_inches='tight')
plt.close(fig3)

print("\nSaved figures: '/Outputs/missing_data.png', '/Outputs/feature_boxplots.png', '/Outputs/correlation_bar_chart.pdf'")
