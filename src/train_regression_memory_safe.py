# src/train_regression_memory_safe.py
"""
Memory-safe regression model:
- Predicts Future_Price_5Y
- Frequency encoding for high-cardinality categorical columns
- Sparse OneHot for low-cardinality columns
- Conservative RandomForestRegressor for stability
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Paths
ROOT = Path(r"D:\Labmentix\RealEstateInvestmentAdvisor")
DATA_PATH = ROOT / "data" / "cleaned_housing_data.csv"
OUT_PATH = ROOT / "models" / "future_price_regressor_memsafe.pkl"

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print("Rows,Cols:", df.shape)

# --- Target Engineering ---
print("Creating Future_Price_5Y...")
df["Future_Price_5Y"] = df["Price_in_Lakhs"] * ((1.08) ** 5)  # 8% appreciation for 5 years

# --- Feature / Target Split ---
y = df["Future_Price_5Y"]
X = df.drop(columns=["Future_Price_5Y", "Good_Investment"], errors="ignore")

# Identify categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Numeric cols:", len(num_cols), "Categorical cols:", len(cat_cols))

# High-cardinality split
CARD_THRESHOLD = 30
high_card_cols = [c for c in cat_cols if X[c].nunique() > CARD_THRESHOLD]
low_card_cols = [c for c in cat_cols if X[c].nunique() <= CARD_THRESHOLD]

print("High-cardinality columns (freq-encoded):", high_card_cols)
print("Low-cardinality columns (onehot sparse):", low_card_cols)

# --- Frequency Encoding ---
for c in high_card_cols:
    freq = X[c].value_counts(dropna=False)
    X[f"{c}_freq"] = X[c].map(freq).fillna(0).astype(np.int32)
    X[f"{c}_freq_ratio"] = X[f"{c}_freq"] / float(len(X))

# Drop original high-card columns
X.drop(columns=high_card_cols, inplace=True)

# Update feature lists
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
low_card_cols = [c for c in low_card_cols if c in X.columns]

print("Final numeric cols:", len(numeric_cols))
print("Final low-cardinality categorical cols:", low_card_cols)

# --- Quick sanity sample ---
sample_size = min(2000, len(X))
print(f"Quick sample size: {sample_size}")

sample_idx = X.sample(n=sample_size, random_state=42).index
X_small = X.loc[sample_idx].reset_index(drop=True)
y_small = y.loc[sample_idx].reset_index(drop=True)

preproc_small = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), low_card_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

reg_small = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=1,
    verbose=1
)

pipe_small = Pipeline([
    ("preproc", preproc_small),
    ("reg", reg_small)
])

print("Fitting quick regression sample...")
pipe_small.fit(X_small, y_small)
pred_small = pipe_small.predict(X_small)

print("\nQuick-sample RMSE:", np.sqrt(mean_squared_error(y_small, pred_small)))
print("Quick-sample MAE:", mean_absolute_error(y_small, pred_small))
mlflow.set_experiment("RealEstate_Regression")

# === FULL TRAINING WITH MLFLOW ===
with mlflow.start_run():
    print("\nProceeding to full training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preproc_full = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), low_card_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    reg_full = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
        verbose=1
    )

    pipeline = Pipeline([("preproc", preproc_full), ("reg", reg_full)])

    print("Training full regression model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    pred_full = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred_full))
    mae  = mean_absolute_error(y_test, pred_full)
    r2   = r2_score(y_test, pred_full)

    print("\nFULL MODEL RESULTS:")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log model
    mlflow.sklearn.log_model(pipeline, "regression_model")

    # Save locally
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_PATH)
    print("Saved:", OUT_PATH)