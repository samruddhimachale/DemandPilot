from pathlib import Path
import json
import math

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from xgboost import XGBRegressor


# ============================================================
# DEMANDPILOT - XGBOOST FORECASTING MODEL
# ============================================================

print("=" * 60)
print("DEMANDPILOT XGBOOST MODEL")
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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_FILE = (
    REPORT_DIR
    / "xgboost_metrics.json"
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_demand_model.json"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Loading feature dataset")
print("=" * 60)


table = pq.read_table(
    FEATURE_FILE,
    use_threads=False
)

df = table.to_pandas()

print("Dataset loaded.")
print("Shape:", df.shape)


# ============================================================
# 3. PREPARE DATA
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


df = df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# 4. TIME SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Creating time-based split")
print("=" * 60)


max_date = df["date"].max()

validation_start = (
    max_date
    - pd.Timedelta(days=27)
)


# Final 28 days = validation

validation_df = df[
    df["date"] >= validation_start
].copy()


# ------------------------------------------------------------
# Training window
#
# We use the most recent 90 days before validation.
#
# This keeps XGBoost computationally manageable while
# retaining recent demand patterns.
# ------------------------------------------------------------

training_start = (
    validation_start
    - pd.Timedelta(days=90)
)


train_df = df[
    (df["date"] >= training_start)
    &
    (df["date"] < validation_start)
].copy()


print("\nTraining period:")
print(
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)


print("\nValidation period:")
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
# 5. DEFINE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Preparing model features")
print("=" * 60)


feature_columns = [

    # --------------------------------------------------------
    # Product / store
    # --------------------------------------------------------

    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    "is_event",
    "snap",

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    "sell_price",
    "price_missing",
    "price_change",
    "price_rolling_mean_7",

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    # --------------------------------------------------------
    # Rolling demand
    # --------------------------------------------------------

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    "days_since_start"
]


target_column = "demand"


# ============================================================
# 6. CONVERT CATEGORICAL VARIABLES TO CODES
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Encoding categorical features")
print("=" * 60)


categorical_columns = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]


for column in categorical_columns:

    # Make sure train and validation use the same categories.

    combined_categories = pd.concat(
        [
            train_df[column],
            validation_df[column]
        ]
    ).astype("category").cat.categories

    train_df[column] = (
        pd.Categorical(
            train_df[column],
            categories=combined_categories
        ).codes
    )

    validation_df[column] = (
        pd.Categorical(
            validation_df[column],
            categories=combined_categories
        ).codes
    )


print("Categorical features encoded.")


# ============================================================
# 7. CREATE X AND Y
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Creating training matrices")
print("=" * 60)


X_train = train_df[
    feature_columns
].copy()


y_train = train_df[
    target_column
].copy()


X_valid = validation_df[
    feature_columns
].copy()


y_valid = validation_df[
    target_column
].copy()


# Replace infinite values

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_valid = X_valid.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill missing numerical values

X_train = X_train.fillna(0)
X_valid = X_valid.fillna(0)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)


# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Training XGBoost")
print("=" * 60)


model = XGBRegressor(

    # Number of trees
    n_estimators=500,

    # Learning rate
    learning_rate=0.05,

    # Tree depth
    max_depth=8,

    # Minimum child weight
    min_child_weight=5,

    # Row subsampling
    subsample=0.8,

    # Feature subsampling
    colsample_bytree=0.8,

    # L2 regularization
    reg_lambda=1.0,

    # Objective
    objective="reg:squarederror",

    # Evaluation metric
    eval_metric="mae",

    # Use all available CPU threads
    n_jobs=-1,

    # Reproducibility
    random_state=42
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_valid, y_valid)
    ],

    verbose=True
)


print("\nXGBoost training completed.")


# ============================================================
# 9. PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Generating predictions")
print("=" * 60)


predictions = model.predict(
    X_valid
)


predictions = np.maximum(
    predictions,
    0
)


print("Predictions generated.")


# ============================================================
# 10. EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Evaluating XGBoost")
print("=" * 60)


actual = y_valid.to_numpy(
    dtype=float
)


predicted = predictions.astype(
    float
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
    np.mean(
        smape_values
    )
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
            np.abs(
                actual - predicted
            )
        )
        / total_actual
        * 100
    )


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST RESULTS")
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
# 12. SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Saving model")
print("=" * 60)


model.save_model(
    MODEL_FILE
)


print(
    "Model saved to:"
)

print(
    MODEL_FILE
)


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics = {

    "model": "XGBoost",

    "training_window_days": 90,

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
    REPORT_DIR / "xgboost_metrics.json",
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
    REPORT_DIR / "xgboost_metrics.json"
)


# ============================================================
# 14. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)


result_sample = validation_df[
    [
        "date",
        "store_id",
        "item_id",
        "demand"
    ]
].copy()


result_sample["prediction"] = (
    predictions
)


print(
    result_sample.head(10)
)


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)


importance = pd.DataFrame({

    "feature": feature_columns,

    "importance": model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    importance.head(15)
)


# Save feature importance

importance.to_csv(
    REPORT_DIR / "xgboost_feature_importance.csv",
    index=False
)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST TRAINING COMPLETED SUCCESSFULLY! ✅")
print("=" * 60)


print("\nModel:")
print(MODEL_FILE)


print("\nMetrics:")
print(REPORT_DIR / "xgboost_metrics.json")


print("\nFeature importance:")
print(
    REPORT_DIR
    / "xgboost_feature_importance.csv"
)