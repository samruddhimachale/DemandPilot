import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


# ============================================================
# DEMANDPILOT BASELINE MODEL
# Memory-Efficient Seasonal Naive Baseline
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "features.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "baseline_metrics.json"

VALIDATION_DAYS = 28

print("=" * 60)
print("DEMANDPILOT BASELINE MODEL")
print("=" * 60)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nFeature file:")
print(FEATURE_FILE)


# ============================================================
# STEP 1: Define validation period
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Creating validation period")
print("=" * 60)

validation_start = "2016-03-28"
validation_end = "2016-04-24"

print(f"Validation start: {validation_start}")
print(f"Validation end:   {validation_end}")
print(f"Validation days:  {VALIDATION_DAYS}")


# ============================================================
# STEP 2: Open Parquet dataset
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Opening feature dataset")
print("=" * 60)

dataset = ds.dataset(
    FEATURE_FILE,
    format="parquet"
)

print("Feature dataset opened successfully.")


# ============================================================
# STEP 3: Load ONLY validation rows
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Loading validation data efficiently")
print("=" * 60)

print("Loading only:")
print(" - date")
print(" - demand")
print(" - lag_7")

# Convert dates to PyArrow-compatible timestamps
start_date = pd.Timestamp(validation_start).to_pydatetime()
end_date = pd.Timestamp(validation_end).to_pydatetime()

# Create filter
date_filter = (
    (ds.field("date") >= start_date)
    &
    (ds.field("date") <= end_date)
)

# Read only 3 columns and only validation rows
table = dataset.to_table(
    columns=[
        "date",
        "demand",
        "lag_7"
    ],
    filter=date_filter
)

validation = table.to_pandas()

del table

print(f"\nValidation rows loaded: {len(validation):,}")


# ============================================================
# STEP 4: Create Seasonal Naive Predictions
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Creating seasonal naive predictions")
print("=" * 60)

# Seasonal Naive:
# Prediction = demand from 7 days ago

validation["prediction"] = validation["lag_7"]

# Remove missing values
validation = validation.dropna(
    subset=[
        "demand",
        "prediction"
    ]
)

print(
    f"Rows used for evaluation: "
    f"{len(validation):,}"
)


# ============================================================
# STEP 5: Convert to NumPy
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Preparing evaluation arrays")
print("=" * 60)

actual = validation["demand"].to_numpy(
    dtype=np.float64
)

predicted = validation["prediction"].to_numpy(
    dtype=np.float64
)

print(f"Actual values:    {len(actual):,}")
print(f"Predicted values: {len(predicted):,}")


# ============================================================
# STEP 6: Calculate errors
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Calculating errors")
print("=" * 60)

errors = actual - predicted


# ============================================================
# STEP 7: Calculate MAE
# ============================================================

mae = np.mean(
    np.abs(errors)
)


# ============================================================
# STEP 8: Calculate RMSE
# ============================================================

rmse = np.sqrt(
    np.mean(errors ** 2)
)


# ============================================================
# STEP 9: Calculate sMAPE
# ============================================================

smape_denominator = (
    np.abs(actual)
    +
    np.abs(predicted)
)

smape_values = np.zeros_like(
    errors,
    dtype=np.float64
)

non_zero = smape_denominator != 0

smape_values[non_zero] = (
    2
    * np.abs(errors[non_zero])
    / smape_denominator[non_zero]
)

smape = np.mean(
    smape_values
) * 100


# ============================================================
# STEP 10: Calculate WAPE
# ============================================================

total_actual = np.sum(
    np.abs(actual)
)

if total_actual == 0:

    wape = 0.0

else:

    wape = (
        np.sum(np.abs(errors))
        /
        total_actual
        *
        100
    )


# ============================================================
# STEP 11: Display results
# ============================================================

print("\n" + "=" * 60)
print("BASELINE PERFORMANCE")
print("=" * 60)

print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"sMAPE: {smape:.2f}%")
print(f"WAPE:  {wape:.2f}%")


# ============================================================
# STEP 12: Save metrics
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: Saving baseline metrics")
print("=" * 60)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

metrics = {
    "model": "Seasonal Naive",
    "validation_days": VALIDATION_DAYS,
    "validation_start": validation_start,
    "validation_end": validation_end,
    "validation_rows": int(len(actual)),
    "mae": float(mae),
    "rmse": float(rmse),
    "smape_percent": float(smape),
    "wape_percent": float(wape)
}

with open(
    REPORT_FILE,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print(f"\nSaved metrics to:")
print(REPORT_FILE)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL COMPLETE")
print("=" * 60)