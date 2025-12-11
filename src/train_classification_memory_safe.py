# src/train_classification_memory_safe.py
"""
Memory-safe classification training.
- Uses frequency / aggregate encodings for high-cardinality categorical columns
- Uses OneHotEncoder (sparse_output=True) only for low-cardinality categorical columns
- Quick sample sanity-check, then conservative full training
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

ROOT = Path(r"D:\Labmentix\RealEstateInvestmentAdvisor")
DATA_PATH = ROOT / "data" / "cleaned_housing_data.csv"
OUT_PATH = ROOT / "models" / "investment_classifier_memsafe.pkl"

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print("Rows,Cols:", df.shape)

# Basic checks
if "Good_Investment" not in df.columns:
    raise RuntimeError("Target column 'Good_Investment' not found in cleaned dataset.")

# Drop auxiliary columns that shouldn't be features
df = df.copy()
drop_if_present = ["City_Median_Price_per_SqFt", "Future_Price_5Y"]
for c in drop_if_present:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# Target and features
y = df["Good_Investment"]
X = df.drop(columns=["Good_Investment"], errors="ignore")

# Identify categorical columns and their cardinality
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
print("Numeric cols:", len(num_cols), "Categorical cols:", len(cat_cols))

# Decide threshold for high-cardinality
CARD_THRESHOLD = 30
high_card_cols = [c for c in cat_cols if X[c].nunique() > CARD_THRESHOLD]
low_card_cols = [c for c in cat_cols if X[c].nunique() <= CARD_THRESHOLD]
print("High-cardinality columns (freq-encoded):", high_card_cols)
print("Low-cardinality columns (onehot sparse):", low_card_cols)

# 1) Frequency encoding for high-cardinality columns (safe, no target leakage)
for c in high_card_cols:
    freq = X[c].value_counts(dropna=False)
    X[f"{c}_freq"] = X[c].map(freq).fillna(0).astype(np.int32)
    # Also add normalized frequency
    X[f"{c}_freq_ratio"] = X[f"{c}_freq"] / float(len(X))
# Drop original high-cardinality columns
X.drop(columns=high_card_cols, inplace=True)

# 2) For low-cardinality columns we will OneHotEncode (sparse_output=True)
# We'll keep column lists updated
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# remove any engineered *_freq columns from categorical lists
low_card_cols = [c for c in low_card_cols if c in X.columns]

print("Final numeric cols used:", len(numeric_cols))
print("Final low-cardinality categorical cols used for OHE:", low_card_cols)

# Quick-sanity sample to ensure pipeline works
sample_size = min(2000, len(X))
print(f"Using a quick sample of {sample_size} rows for a fast sanity check.")
mlflow.set_experiment("RealEstate_Classification")
sample_idx = X.sample(n=sample_size, random_state=42).index
X_small = X.loc[sample_idx].reset_index(drop=True)
y_small = y.loc[sample_idx].reset_index(drop=True)

# Build a memory-safe preprocessor: StandardScale numeric, OneHot sparse for low-card cols
preprocessor_small = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), low_card_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3  # keep data sparse if possible
)

# small RandomForest for quick test
clf_small = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1, verbose=1)
pipe_small = Pipeline([("preproc", preprocessor_small), ("clf", clf_small)])

print("Fitting quick sample pipeline (this should be fast)...")
Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(X_small, y_small, test_size=0.2, random_state=42, stratify=y_small)
pipe_small.fit(Xtr_s, ytr_s)
yhat_s = pipe_small.predict(Xte_s)
print("Quick-sample results -> Accuracy:", accuracy_score(yte_s, yhat_s),
      "Precision:", precision_score(yte_s, yhat_s, zero_division=0),
      "Recall:", recall_score(yte_s, yhat_s, zero_division=0),
      "F1:", f1_score(yte_s, yhat_s, zero_division=0))

# === FULL TRAINING WITH MLFLOW ===
with mlflow.start_run():
    print("\nProceeding to full training (conservative settings)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor_full = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), low_card_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    clf_full = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
        verbose=1
    )

    pipeline = Pipeline([("preproc", preprocessor_full), ("clf", clf_full)])

    print("Training full classification model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(pipeline, "classification_model")

    # Save locally
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_PATH)
    print("Saved:", OUT_PATH)