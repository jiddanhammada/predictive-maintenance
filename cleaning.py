# ============================================================
# cleaning.py
# Predictive Maintenance — Data Preparation & EDA
# Dataset: AI4I 2020 Predictive Maintenance (UCI)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ── Path ─────────────────────────────────────────────────────
RAW_PATH    = 'data/Factory Maintenance Dataset.csv'
OUTPUT_DIR  = 'output'
DATA_DIR    = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PREDICTIVE MAINTENANCE — DATA PREPARATION")
print("=" * 60)


# ============================================================
# SECTION 1: LOAD DATA
# ============================================================
df_raw = pd.read_csv(RAW_PATH)
df     = df_raw.copy()

print(f"\n[1] Dataset loaded")
print(f"    Shape  : {df.shape}")
print(f"    Columns: {df.columns.tolist()}")


# ============================================================
# SECTION 2: BASIC INSPECTION
# ============================================================
print("\n[2] Basic Inspection")
print(f"    Dtypes:\n{df.dtypes}")
print(f"\n    Missing values:\n{df.isnull().sum()}")
print(f"\n    Duplicates: {df.duplicated().sum()}")
print(f"\n    Descriptive stats:\n{df.describe()}")


# ============================================================
# SECTION 3: EDA — TARGET DISTRIBUTION
# ============================================================
print("\n[3] EDA — Target Distribution")

target_counts = df['Machine failure'].value_counts()
failure_rate  = df['Machine failure'].mean()
print(f"    Class distribution: {target_counts.to_dict()}")
print(f"    Failure rate      : {failure_rate:.2%}")
print(f"    Imbalance ratio   : {target_counts[0]/target_counts[1]:.1f}:1")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(['Normal (0)', 'Failure (1)'],
            target_counts.values,
            color=['#2196F3', '#F44336'])
axes[0].set_title('Class Distribution — Machine Failure')
axes[0].set_ylabel('Count')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 30, f'{v} ({v/len(df)*100:.1f}%)', ha='center')

axes[1].pie(target_counts.values,
            labels=['Normal', 'Failure'],
            colors=['#2196F3', '#F44336'],
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Failure Rate Proportion')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_01_target_distribution.png', dpi=150)
plt.close()
print(f"    → Saved: eda_01_target_distribution.png")


# ============================================================
# SECTION 4: EDA — NUMERIC FEATURE DISTRIBUTION
# ============================================================
print("\n[4] EDA — Numeric Feature Distribution")

numeric_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, col in enumerate(numeric_cols):
    # Histogram
    axes[0, i].hist(df[col], bins=40, color='#2196F3', edgecolor='white', linewidth=0.3)
    axes[0, i].set_title(col, fontsize=9)
    axes[0, i].set_ylabel('Count')

    # Boxplot per class
    data_normal  = df[df['Machine failure'] == 0][col]
    data_failure = df[df['Machine failure'] == 1][col]
    axes[1, i].boxplot([data_normal, data_failure],
                       labels=['Normal', 'Failure'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#E3F2FD'),
                       medianprops=dict(color='#F44336', linewidth=2))
    axes[1, i].set_title(f'{col}\nvs Failure', fontsize=9)

plt.suptitle('Numeric Feature Distribution & Boxplot vs Failure', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_02_numeric_distribution.png', dpi=150)
plt.close()
print(f"    → Saved: eda_02_numeric_distribution.png")


# ============================================================
# SECTION 5: EDA — CORRELATION HEATMAP
# ============================================================
print("\n[5] EDA — Correlation Heatmap")

corr_cols = numeric_cols + ['Machine failure']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.3f',
            cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5)
ax.set_title('Correlation Matrix — Sensor Features vs Machine Failure')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_03_correlation_heatmap.png', dpi=150)
plt.close()
print(f"    → Saved: eda_03_correlation_heatmap.png")


# ============================================================
# SECTION 6: EDA — FAILURE MODE ANALYSIS
# ============================================================
print("\n[6] EDA — Failure Mode Analysis")

failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
mode_counts   = df[failure_modes].sum().sort_values(ascending=False)
print(f"    Failure mode counts:\n{mode_counts}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#EF5350', '#FF7043', '#FFA726', '#66BB6A', '#42A5F5']
bars = axes[0].bar(mode_counts.index, mode_counts.values, color=colors)
axes[0].set_title('Failure Mode — Count per Type')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Failure Mode')
for bar, val in zip(bars, mode_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1, str(val), ha='center', fontsize=11)

pct = mode_counts / len(df) * 100
axes[1].barh(pct.index, pct.values, color='#2196F3')
axes[1].set_title('Failure Mode — Percentage of Total Data')
axes[1].set_xlabel('Percentage (%)')
for i, v in enumerate(pct.values):
    axes[1].text(v + 0.01, i, f'{v:.2f}%', va='center')

plt.suptitle('Failure Mode Analysis', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_04_failure_modes.png', dpi=150)
plt.close()
print(f"    → Saved: eda_04_failure_modes.png")


# ============================================================
# SECTION 7: EDA — FAILURE RATE PER PRODUCT TYPE
# ============================================================
print("\n[7] EDA — Failure Rate per Product Type")

type_failure = df.groupby('Type')['Machine failure'].agg(['sum', 'count'])
type_failure['rate'] = type_failure['sum'] / type_failure['count'] * 100
type_failure = type_failure.reindex(['L', 'M', 'H'])
print(f"\n{type_failure}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(type_failure.index, type_failure['rate'],
            color=['#64B5F6', '#42A5F5', '#1565C0'])
axes[0].set_title('Failure Rate per Product Type (%)')
axes[0].set_ylabel('Failure Rate (%)')
axes[0].set_xlabel('Product Type')
for i, (idx, row) in enumerate(type_failure.iterrows()):
    axes[0].text(i, row['rate'] + 0.05, f"{row['rate']:.2f}%", ha='center')

type_counts = df['Type'].value_counts().reindex(['L', 'M', 'H'])
axes[1].bar(type_counts.index, type_counts.values,
            color=['#64B5F6', '#42A5F5', '#1565C0'])
axes[1].set_title('Data Count per Product Type')
axes[1].set_ylabel('Count')
for i, v in enumerate(type_counts.values):
    axes[1].text(i, v + 30, f'{v} ({v/len(df)*100:.0f}%)', ha='center')

plt.suptitle('Analysis per Product Type', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_05_product_type.png', dpi=150)
plt.close()
print(f"    → Saved: eda_05_product_type.png")


# ============================================================
# SECTION 8: DATA CLEANING
# ============================================================
print("\n[8] Data Cleaning")
print(f"    Shape before: {df.shape}")

# Drop duplicates
df = df.drop_duplicates()
print(f"    After drop duplicates: {df.shape}")

# Drop identifier columns (not useful for ML)
df = df.drop(columns=['UDI', 'Product ID'], errors='ignore')
print(f"    After drop identifiers: {df.shape}")

# Rename columns — lowercase, no spaces or special characters
df.columns = (df.columns
               .str.strip()
               .str.lower()
               .str.replace(' ', '_')
               .str.replace('[', '', regex=False)
               .str.replace(']', '', regex=False))

print(f"    Columns after rename: {df.columns.tolist()}")

# Verify no missing values
assert df.isnull().sum().sum() == 0, "Missing values found after cleaning!"
print(f"    Missing values after cleaning: {df.isnull().sum().sum()} ✅")


# ============================================================
# SECTION 9: FEATURE ENGINEERING
# ============================================================
print("\n[9] Feature Engineering")

# 1. Power = Torque × Angular Velocity (rad/s)
df['power_w'] = df['torque_nm'] * (df['rotational_speed_rpm'] * 2 * np.pi / 60)

# 2. Temperature difference
df['temp_diff_k'] = df['process_temperature_k'] - df['air_temperature_k']

# 3. Overstrain value
df['overstrain_value'] = df['tool_wear_min'] * df['torque_nm']

# 4. Heat Dissipation Failure risk flag
df['hdf_risk'] = (
    (df['temp_diff_k'] < 8.6) &
    (df['rotational_speed_rpm'] < 1380)
).astype(int)

# 5. Power Failure risk flag
df['pwf_risk'] = (
    (df['power_w'] < 3500) |
    (df['power_w'] > 9000)
).astype(int)

# 6. Overstrain Failure risk flag (threshold per product type)
osf_threshold = {'l': 11000, 'm': 12000, 'h': 13000}
df['osf_threshold'] = df['type'].str.lower().map(osf_threshold)
df['osf_risk'] = (df['overstrain_value'] > df['osf_threshold']).astype(int)
df = df.drop(columns=['osf_threshold'])

# 7. Tool wear ratio (proximity to max wear limit 240 min)
df['tool_wear_ratio'] = df['tool_wear_min'] / 240

# 8. Combined risk score
df['risk_score'] = df['hdf_risk'] + df['pwf_risk'] + df['osf_risk']

# 9. Encode product type
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])
type_mapping = {k: int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
print(f"    Type encoding: {type_mapping}")

# Save encoding mapping
with open(f'{OUTPUT_DIR}/type_encoding.json', 'w') as f:
    json.dump(type_mapping, f, indent=2)

print(f"\n    New features created:")
new_feats = ['power_w', 'temp_diff_k', 'overstrain_value',
             'hdf_risk', 'pwf_risk', 'osf_risk',
             'tool_wear_ratio', 'risk_score']
print(df[new_feats].describe().round(3))


# ============================================================
# SECTION 10: FEATURE CORRELATION WITH TARGET
# ============================================================
print("\n[10] Feature Correlation with Target")

feature_cols = [
    'type_encoded', 'air_temperature_k', 'process_temperature_k',
    'rotational_speed_rpm', 'torque_nm', 'tool_wear_min',
    'power_w', 'temp_diff_k', 'overstrain_value',
    'hdf_risk', 'pwf_risk', 'osf_risk', 'tool_wear_ratio', 'risk_score'
]

corr_target = (df[feature_cols + ['machine_failure']]
               .corr()['machine_failure']
               .drop('machine_failure')
               .sort_values(key=abs, ascending=False))

print(f"\n    Correlation with machine_failure:\n{corr_target.round(4)}")

fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#EF5350' if v > 0 else '#2196F3' for v in corr_target.values]
ax.barh(corr_target.index, corr_target.values, color=colors)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_title('Feature Correlation with Machine Failure', fontsize=13)
ax.set_xlabel('Pearson Correlation Coefficient')
for i, v in enumerate(corr_target.values):
    ax.text(v + 0.003 if v >= 0 else v - 0.003, i,
            f'{v:.3f}', va='center',
            ha='left' if v >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_06_feature_correlation.png', dpi=150)
plt.close()
print(f"    → Saved: eda_06_feature_correlation.png")


# ============================================================
# SECTION 11: OUTLIER DETECTION
# ============================================================
print("\n[11] Outlier Detection (3×IQR Method)")

outlier_summary = {}
for col in ['air_temperature_k', 'process_temperature_k',
            'rotational_speed_rpm', 'torque_nm', 'tool_wear_min']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_summary[col] = len(outliers)
    # Flag outliers instead of removing (outliers may be real failure signals)
    df[f'{col}_outlier'] = ((df[col] < lower) | (df[col] > upper)).astype(int)

print("    Outlier counts per feature:")
for col, n in outlier_summary.items():
    print(f"      {col}: {n} ({n/len(df)*100:.2f}%)")
print("    → Decision: FLAG outliers, not remove (may be failure signals)")


# ============================================================
# SECTION 12: SAVE DATASETS
# ============================================================
print("\n[12] Saving Datasets")

# Full cleaned dataset
df.to_csv(f'{DATA_DIR}/dataset_cleaned.csv', index=False)
print(f"    → {DATA_DIR}/dataset_cleaned.csv  ({df.shape})")

# ML-ready dataset (features + target only)
ml_feature_cols = feature_cols + ['machine_failure']
df[ml_feature_cols].to_csv(f'{DATA_DIR}/dataset_ml_ready.csv', index=False)
print(f"    → {DATA_DIR}/dataset_ml_ready.csv  ({df[ml_feature_cols].shape})")

# Tableau dataset
tableau_cols = [
    'type', 'type_encoded',
    'air_temperature_k', 'process_temperature_k',
    'rotational_speed_rpm', 'torque_nm', 'tool_wear_min',
    'power_w', 'temp_diff_k', 'overstrain_value',
    'hdf_risk', 'pwf_risk', 'osf_risk', 'risk_score',
    'twf', 'hdf', 'pwf', 'osf', 'rnf', 'machine_failure'
]
available = [c for c in tableau_cols if c in df.columns]
df[available].to_csv(f'{DATA_DIR}/dataset_tableau.csv', index=False)
print(f"    → {DATA_DIR}/dataset_tableau.csv  ({df[available].shape})")

print("\n" + "=" * 60)
print("  DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"  Final dataset shape : {df.shape}")
print(f"  Failure rate        : {df['machine_failure'].mean():.2%}")
print(f"  Features created    : {len(new_feats)} new features")
print(f"  Output files        : {len(os.listdir(OUTPUT_DIR))} files saved")