import os
import hashlib
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Lock seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print(" LAB 04: REPRODUCIBLE DATA PIPELINE & QUALITY AUDIT ")
print("=" * 60)

# -------------------------------------------------------------
# 1. Ingestion: Read Raw Data
# -------------------------------------------------------------
print("\n[Step 1] Ingesting UCI Adult Dataset from OpenML...")
raw_data = fetch_openml(name='adult', version=2, as_frame=True)
df_raw = raw_data.frame.copy()

# Standardize target column name
target_col = 'income' if 'income' in df_raw.columns else ('class' if 'class' in df_raw.columns else df_raw.columns[-1])
if target_col != 'income':
    df_raw.rename(columns={target_col: 'income'}, inplace=True)

# Standardize target labels: clean trailing periods or whitespace (<=50K, >50K)
df_raw['income'] = df_raw['income'].astype(str).str.strip().str.rstrip('.')

print(f"  -> Ingestion Complete. Rows: {len(df_raw)}, Columns: {len(df_raw.columns)}")

# -------------------------------------------------------------
# 2. Validation: Schema, Types, Ranges & Null Checks
# -------------------------------------------------------------
print("\n[Step 2] Running Schema and Range Validation Checks...")

# Convert numeric columns safely
num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
for col in num_cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

cat_cols = [
    'workclass', 'education', 'marital-status', 'occupation', 
    'relationship', 'race', 'sex', 'native-country'
]
for col in cat_cols:
    df_raw[col] = df_raw[col].astype(str).str.strip()

validation_report = []

# Range definitions from syllabus
ranges = {
    'age': (17, 90),
    'hours-per-week': (1, 99),
    'capital-gain': (0, 99999),
    'capital-loss': (0, 4356)
}

for col, (low, high) in ranges.items():
    violations = df_raw[(df_raw[col] < low) | (df_raw[col] > high)]
    count = len(violations)
    msg = f"Range Check [{col}] ({low}-{high}): {count} violations found."
    validation_report.append(msg)
    print(f"  • {msg}")

# Missing-value policy: '?' represents missing data
missing_counts = {}
for col in df_raw.columns:
    q_count = (df_raw[col] == '?').sum()
    nan_count = df_raw[col].isna().sum()
    total_missing = q_count + nan_count
    if total_missing > 0:
        missing_counts[col] = int(total_missing)

print("  • Missing Values ('?' or NaN):")
for col, cnt in missing_counts.items():
    print(f"    - {col}: {cnt} missing entries ({cnt / len(df_raw) * 100:.2f}%)")

# -------------------------------------------------------------
# 3. Label Quality Audit (Duplicate Conflicts & Sample Check)
# -------------------------------------------------------------
print("\n[Step 3] Running Label-Quality Audit...")
feature_cols = [c for c in df_raw.columns if c != 'income']

# Find exact feature duplicates that have conflicting income labels
feature_duplicates = df_raw[df_raw.duplicated(subset=feature_cols, keep=False)]
conflicting_groups = (
    feature_duplicates.groupby(feature_cols)['income']
    .nunique()
    .reset_index(name='distinct_labels')
)
num_conflicts = len(conflicting_groups[conflicting_groups['distinct_labels'] > 1])
print(f"  • Duplicate-Conflict Audit: Found {num_conflicts} groups of identical feature rows with conflicting income labels.")

# Draw 50 random rows for manual inspection
sample_50 = df_raw.sample(n=50, random_state=RANDOM_STATE)
os.makedirs("lab04/results", exist_ok=True)
sample_50.to_csv("lab04/results/label_audit_sample_50.csv", index=False)
print("  • Saved 50 random rows to 'lab04/results/label_audit_sample_50.csv' for audit.")

# -------------------------------------------------------------
# 4. Leakage-Free Splitting & Preprocessing
# -------------------------------------------------------------
print("\n[Step 4] Splitting Train/Test (70/30 Stratified) BEFORE Cleaning...")

# Split strictly before fitting any imputers, scalers, or encoders
X = df_raw[feature_cols].copy()
y = (df_raw['income'] == '>50K').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

print(f"  -> Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")

# Drop exact duplicates from TRAIN only; keep test untouched
train_full = pd.concat([X_train, y_train], axis=1)
train_dedup = train_full.drop_duplicates()
X_train = train_dedup[feature_cols].copy()
y_train = train_dedup['income'].copy()
print(f"  -> Dropped {len(train_full) - len(train_dedup)} exact duplicates from Train set. Test remains pristine.")

# Compute medians and modes strictly from Train set
num_medians = {c: X_train[c].median() for c in num_cols}
cat_modes = {}
for c in cat_cols:
    clean_series = X_train[c][X_train[c] != '?']
    cat_modes[c] = clean_series.mode()[0]

# Compute 1st and 99th percentiles strictly from Train set for outlier clipping
clip_limits = {}
for c in num_cols:
    p1 = X_train[c].quantile(0.01)
    p99 = X_train[c].quantile(0.99)
    clip_limits[c] = (p1, p99)

def clean_and_transform(df_in):
    df_out = df_in.copy()
    # 1. Fill missing numeric with train median, clip to train 1st/99th
    for c in num_cols:
        df_out[c] = df_out[c].fillna(num_medians[c])
        low, high = clip_limits[c]
        df_out[c] = df_out[c].clip(lower=low, upper=high)
    # 2. Fill missing '?' with train mode
    for c in cat_cols:
        df_out[c] = df_out[c].replace('?', cat_modes[c])
    return df_out

X_train_clean = clean_and_transform(X_train)
X_test_clean = clean_and_transform(X_test)

# -------------------------------------------------------------
# 5. Feature Engineering: capital_net & hours_per_age
# -------------------------------------------------------------
print("\n[Step 5] Engineering Derived Features...")
for df_c in [X_train_clean, X_test_clean]:
    df_c['capital_net'] = df_c['capital-gain'] - df_c['capital-loss']
    df_c['hours_per_age'] = df_c['hours-per-week'] / np.maximum(df_c['age'] - 16, 1)

derived_cols = ['capital_net', 'hours_per_age']

# Build Column Transformers
base_num_cols = num_cols
all_num_cols = num_cols + derived_cols

preprocessor_base = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), base_num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

preprocessor_derived = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), all_num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# Fit on train only, transform train & test
X_train_base = preprocessor_base.fit_transform(X_train_clean[base_num_cols + cat_cols])
X_test_base = preprocessor_base.transform(X_test_clean[base_num_cols + cat_cols])

X_train_derived = preprocessor_derived.fit_transform(X_train_clean[all_num_cols + cat_cols])
X_test_derived = preprocessor_derived.transform(X_test_clean[all_num_cols + cat_cols])

# Save fitted transformer
joblib.dump(preprocessor_derived, "lab04/results/preprocessor.joblib")
print("  -> Saved fitted preprocessor to 'lab04/results/preprocessor.joblib'")

# -------------------------------------------------------------
# 6. Model Baseline: With vs. Without Engineered Features
# -------------------------------------------------------------
print("\n[Step 6] Testing Logistic Regression Baseline...")

# Model A: Without derived features
clf_a = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf_a.fit(X_train_base, y_train)
preds_a = clf_a.predict(X_test_base)
acc_a = accuracy_score(y_test, preds_a)
f1_a = f1_score(y_test, preds_a)

# Model B: With derived features
clf_b = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf_b.fit(X_train_derived, y_train)
preds_b = clf_b.predict(X_test_derived)
acc_b = accuracy_score(y_test, preds_b)
f1_b = f1_score(y_test, preds_b)

print(f"  • Model WITHOUT derived features: Accuracy = {acc_a * 100:.2f}%, F1 Score = {f1_a:.4f}")
print(f"  • Model WITH derived features:    Accuracy = {acc_b * 100:.2f}%, F1 Score = {f1_b:.4f}")

# -------------------------------------------------------------
# 7. Reproducibility: SHA-256 Hash & Config YAML
# -------------------------------------------------------------
print("\n[Step 7] Computing Dataset Hash and Generating config.yaml...")

# Export processed dataset to CSV
processed_df = pd.concat([
    X_train_clean.assign(is_train=1, target=y_train),
    X_test_clean.assign(is_train=0, target=y_test)
])
processed_path = "lab04/results/processed_adult.csv"
processed_df.to_csv(processed_path, index=False)

# Compute SHA-256 hash of the processed dataset
sha256 = hashlib.sha256()
with open(processed_path, 'rb') as f:
    while chunk := f.read(65536):
        sha256.update(chunk)
dataset_hash = sha256.hexdigest()

config_data = {
    "dataset": {
        "name": "UCI Adult Census",
        "openml_version": 2,
        "raw_rows": len(df_raw),
        "raw_columns": len(df_raw.columns),
        "processed_sha256": dataset_hash
    },
    "pipeline": {
        "random_seed": RANDOM_STATE,
        "test_split_ratio": 0.30,
        "stratified": True,
        "missing_symbol": "?",
        "imputation": {"categorical": "mode", "numeric": "median"},
        "outlier_clipping": "1st and 99th percentiles (train-fitted)",
        "derived_features": ["capital_net", "hours_per_age"]
    },
    "baseline_metrics": {
        "without_derived": {"accuracy": float(f"{acc_a:.4f}"), "f1": float(f"{f1_a:.4f}")},
        "with_derived": {"accuracy": float(f"{acc_b:.4f}"), "f1": float(f"{f1_b:.4f}")}
    }
}

config_path = "lab04/results/config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config_data, f, default_flow_style=False)

print(f"  • Processed Dataset SHA-256: {dataset_hash}")
print(f"  • Configuration saved to '{config_path}'")
print("\n[Done] Pipeline executed successfully.")