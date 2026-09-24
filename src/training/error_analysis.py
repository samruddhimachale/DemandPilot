from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from xgboost import XGBRegressor


# ============================================================
# DEMANDPILOT - ERROR ANALYSIS
# ============================================================

print("=" * 60)
print("DEMANDPILOT ERROR ANALYSIS")
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

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xgboost_demand_model.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

ERROR_FILE = (
    REPORT_DIR
    / "error_analysis.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Loading feature data")
print("=" * 60)


table = pq.read_table(
    FEATURE_FILE,
    use_threads=False
)

df = table.to_pandas()


df["date"] = pd.to_datetime(
    df["date"]
)


df["demand"] = pd.to_numeric(
    df["demand"],
    errors="coerce"
)


# ============================================================
# 3. VALIDATION DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Preparing validation data")
print("=" * 60)


max_date = df["date"].max()

validation_start = (
    max_date
    - pd.Timedelta(days=27)
)


validation_df = df[
    df["date"] >= validation_start
].copy()


print(
    "Validation period:",
    validation_start.date(),
    "to",
    max_date.date()
)


# ============================================================
# 4. FEATURES
# ============================================================

feature_columns = [

    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",

    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",

    "is_event",
    "snap",

    "sell_price",
    "price_missing",
    "price_change",
    "price_rolling_mean_7",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",

    "days_since_start"
]


categorical_columns = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]


# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Encoding categorical features")
print("=" * 60)


for column in categorical_columns:

    categories = (
        df[column]
        .astype("category")
        .cat
        .categories
    )

    validation_df[column] = (
        pd.Categorical(
            validation_df[column],
            categories=categories
        ).codes
    )


# ============================================================
# 6. PREPARE X
# ============================================================

X_valid = validation_df[
    feature_columns
].copy()


X_valid = X_valid.replace(
    [np.inf, -np.inf],
    np.nan
)


X_valid = X_valid.fillna(0)


# ============================================================
# 7. LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Loading XGBoost model")
print("=" * 60)


model = XGBRegressor()

model.load_model(
    MODEL_FILE
)


print("Model loaded successfully.")


# ============================================================
# 8. PREDICT
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Generating predictions")
print("=" * 60)


validation_df["prediction"] = (
    model.predict(X_valid)
)


validation_df["prediction"] = (
    validation_df["prediction"]
    .clip(lower=0)
)


# ============================================================
# 9. CALCULATE ERRORS
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Calculating forecast errors")
print("=" * 60)


validation_df["error"] = (
    validation_df["demand"]
    - validation_df["prediction"]
)


validation_df["absolute_error"] = (
    validation_df["error"]
    .abs()
)


validation_df["squared_error"] = (
    validation_df["error"]
    ** 2
)


# ============================================================
# 10. OVERALL ERROR STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("OVERALL ERROR STATISTICS")
print("=" * 60)


print(
    "\nMean absolute error:",
    round(
        validation_df["absolute_error"].mean(),
        4
    )
)


print(
    "Maximum absolute error:",
    round(
        validation_df["absolute_error"].max(),
        4
    )
)


print(
    "Mean actual demand:",
    round(
        validation_df["demand"].mean(),
        4
    )
)


print(
    "Mean predicted demand:",
    round(
        validation_df["prediction"].mean(),
        4
    )
)


# ============================================================
# 11. WORST FORECASTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Finding largest forecast errors")
print("=" * 60)


worst_predictions = (
    validation_df[
        [
            "date",
            "store_id",
            "item_id",
            "demand",
            "prediction",
            "error",
            "absolute_error"
        ]
    ]
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)


print(
    worst_predictions.to_string(
        index=False
    )
)


# ============================================================
# 12. ERROR BY STORE
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Error by store")
print("=" * 60)


store_errors = (
    validation_df
    .groupby("store_id")
    .agg(
        actual_demand=(
            "demand",
            "sum"
        ),

        predicted_demand=(
            "prediction",
            "sum"
        ),

        MAE=(
            "absolute_error",
            "mean"
        ),

        observations=(
            "demand",
            "count"
        )
    )
    .reset_index()
)


store_errors = store_errors.sort_values(
    "MAE",
    ascending=False
)


print(
    store_errors.to_string(
        index=False
    )
)


# ============================================================
# 13. ERROR BY CATEGORY
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Error by product category")
print("=" * 60)


category_errors = (
    validation_df
    .groupby("cat_id")
    .agg(
        actual_demand=(
            "demand",
            "sum"
        ),

        predicted_demand=(
            "prediction",
            "sum"
        ),

        MAE=(
            "absolute_error",
            "mean"
        ),

        observations=(
            "demand",
            "count"
        )
    )
    .reset_index()
)


category_errors = category_errors.sort_values(
    "MAE",
    ascending=False
)


print(
    category_errors.to_string(
        index=False
    )
)


# ============================================================
# 14. ZERO-DEMAND ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Zero-demand error analysis")
print("=" * 60)


zero_demand = validation_df[
    validation_df["demand"] == 0
]


positive_demand = validation_df[
    validation_df["demand"] > 0
]


print(
    "\nZero-demand observations:",
    len(zero_demand)
)


if len(zero_demand) > 0:

    print(
        "Average prediction when actual demand = 0:",
        round(
            zero_demand["prediction"].mean(),
            4
        )
    )


print(
    "\nPositive-demand observations:",
    len(positive_demand)
)


if len(positive_demand) > 0:

    print(
        "MAE on positive demand:",
        round(
            positive_demand["absolute_error"].mean(),
            4
        )
    )


# ============================================================
# 15. SAVE ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: Saving error analysis")
print("=" * 60)


output_columns = [
    "date",
    "store_id",
    "item_id",
    "dept_id",
    "cat_id",
    "demand",
    "prediction",
    "error",
    "absolute_error"
]


validation_df[
    output_columns
].to_csv(
    ERROR_FILE,
    index=False
)


print(
    "Error analysis saved to:"
)

print(
    ERROR_FILE
)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETED! ✅")
print("=" * 60)