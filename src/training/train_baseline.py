from pathlib import Path
import json
import math

import pandas as pd
import numpy as np
import pyarrow.parquet as pq


# ============================================================
# DEMANDPILOT - BASELINE MODEL
# ============================================================

print("=" * 60)
print("DEMANDPILOT BASELINE MODEL")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_FILE = (
    REPORT_DIR
    / "baseline_metrics.json"
)


print("\nProject root:")
print(PROJECT_ROOT)

print("\nFeature file:")
print(FEATURE_FILE)


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Loading feature dataset")
print("=" * 60)


# Use PyArrow directly because pandas.read_parquet()
# caused an ArrowKeyError in your environment.

table = pq.read_table(
    FEATURE_FILE,
    use_threads=False
)

df = table.to_pandas()


print("Feature dataset loaded.")

print("Shape:", df.shape)


# ============================================================
# 3. BASIC PREPARATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Preparing data")
print("=" * 60)


df["date"] = pd.to_datetime(
    df["date"]
)


df["demand"] = pd.to_numeric(
    df["demand"],
    errors="coerce"
)


df["lag_7"] = pd.to_numeric(
    df["lag_7"],
    errors="coerce"
)


# Sort chronologically

df = df.sort_values(
    [
        "date",
        "store_id",
        "item_id"
    ]
).reset_index(drop=True)


print("Date range:")
print(
    df["date"].min(),
    "to",
    df["date"].max()
)


# ============================================================
# 4. CREATE TIME-BASED TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Creating train/validation split")
print("=" * 60)


# Use the final 28 days as validation.
#
# This simulates the real forecasting situation:
#
# Past data → train
# Future 28 days → validation
#
# We DO NOT randomly split time-series data.

max_date = df["date"].max()

validation_start = (
    max_date
    - pd.Timedelta(days=27)
)


train_df = df[
    df["date"] < validation_start
].copy()


validation_df = df[
    df["date"] >= validation_start
].copy()


print("\nMaximum date:")
print(max_date)


print("\nValidation starts:")
print(validation_start)


print("\nTraining date range:")
print(
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)


print("\nValidation date range:")
print(
    validation_df["date"].min(),
    "to",
    validation_df["date"].max()
)


print("\nTraining rows:")
print(len(train_df))


print("\nValidation rows:")
print(len(validation_df))


# ============================================================
# 5. BASELINE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Creating baseline predictions")
print("=" * 60)


# Baseline:
#
# Today's demand prediction
# =
# Demand from 7 days ago
#
# lag_7 was already created during feature engineering.

validation_df["prediction"] = (
    validation_df["lag_7"]
)


# Remove rows where baseline prediction is unavailable

validation_df = validation_df.dropna(
    subset=[
        "prediction",
        "demand"
    ]
).copy()


print(
    "Validation rows with predictions:",
    len(validation_df)
)


# ============================================================
# 6. METRICS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Evaluating baseline")
print("=" * 60)


actual = (
    validation_df["demand"]
    .to_numpy(dtype=float)
)


predicted = (
    validation_df["prediction"]
    .to_numpy(dtype=float)
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

mae = np.mean(
    np.abs(
        actual - predicted
    )
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

rmse = math.sqrt(
    np.mean(
        (actual - predicted) ** 2
    )
)


# ------------------------------------------------------------
# sMAPE
# ------------------------------------------------------------

denominator = (
    np.abs(actual)
    + np.abs(predicted)
)


smape_values = np.where(
    denominator == 0,
    0,
    2
    * np.abs(actual - predicted)
    / denominator
)


smape = (
    np.mean(smape_values)
    * 100
)


# ------------------------------------------------------------
# WAPE
# ------------------------------------------------------------

total_actual = np.sum(
    np.abs(actual)
)


if total_actual == 0:

    wape = 0

else:

    wape = (
        np.sum(
            np.abs(actual - predicted)
        )
        / total_actual
        * 100
    )


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BASELINE RESULTS")
print("=" * 60)


print(
    f"\nMAE:   {mae:.4f}"
)


print(
    f"RMSE:  {rmse:.4f}"
)


print(
    f"sMAPE: {smape:.2f}%"
)


print(
    f"WAPE:  {wape:.2f}%"
)


# ============================================================
# 8. ADDITIONAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION INFORMATION")
print("=" * 60)


print(
    "\nNumber of validation days:",
    validation_df["date"].nunique()
)


print(
    "Number of stores:",
    validation_df["store_id"].nunique()
)


print(
    "Number of items:",
    validation_df["item_id"].nunique()
)


print(
    "Actual total demand:",
    round(
        validation_df["demand"].sum(),
        2
    )
)


print(
    "Predicted total demand:",
    round(
        validation_df["prediction"].sum(),
        2
    )
)


# ============================================================
# 9. SAVE METRICS
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Saving baseline metrics")
print("=" * 60)


metrics = {

    "model": "Seasonal Naive - Lag 7",

    "validation_days": int(
        validation_df["date"].nunique()
    ),

    "train_start": str(
        train_df["date"].min().date()
    ),

    "train_end": str(
        train_df["date"].max().date()
    ),

    "validation_start": str(
        validation_df["date"].min().date()
    ),

    "validation_end": str(
        validation_df["date"].max().date()
    ),

    "training_rows": int(
        len(train_df)
    ),

    "validation_rows": int(
        len(validation_df)
    ),

    "mae": float(mae),

    "rmse": float(rmse),

    "smape_percent": float(smape),

    "wape_percent": float(wape)
}


with open(
    METRICS_FILE,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    "\nMetrics saved to:"
)

print(
    METRICS_FILE
)


# ============================================================
# 10. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)


sample_columns = [
    "date",
    "store_id",
    "item_id",
    "demand",
    "lag_7",
    "prediction"
]


print(
    validation_df[
        sample_columns
    ].head(10)
)


# ============================================================
# 11. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL COMPLETED SUCCESSFULLY! ✅")
print("=" * 60)

print("\nBaseline model:")
print("Seasonal Naive using lag_7")

print("\nNext step:")
print("Train an ML model and compare it against this baseline.")