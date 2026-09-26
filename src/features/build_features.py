from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq


# ============================================================
# DEMANDPILOT - FEATURE ENGINEERING
# ============================================================

print("=" * 60)
print("DEMANDPILOT FEATURE ENGINEERING")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "train.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "features.parquet"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nInput file:")
print(INPUT_FILE)

print("\nOutput file:")
print(OUTPUT_FILE)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Loading processed data")
print("=" * 60)

# We use PyArrow directly because pandas.read_parquet()
# was giving an ArrowKeyError in your environment.

table = pq.read_table(
    INPUT_FILE,
    use_threads=False
)

df = table.to_pandas()

print("Data loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# 3. BASIC CLEANING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Basic cleaning")
print("=" * 60)

# Convert date to datetime
df["date"] = pd.to_datetime(
    df["date"]
)


# Sort by store, item and date.
# This is essential before creating lag features.

df = df.sort_values(
    [
        "store_id",
        "item_id",
        "date"
    ]
).reset_index(drop=True)

print("Data sorted successfully.")


# ============================================================
# 4. PREPARE TARGET VARIABLE
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Preparing target variable")
print("=" * 60)

df["demand"] = pd.to_numeric(
    df["demand"],
    errors="coerce"
)

df["demand"] = df["demand"].fillna(0)

print("Target variable: demand")

print(
    "Average demand:",
    round(df["demand"].mean(), 2)
)

print(
    "Maximum demand:",
    df["demand"].max()
)


# ============================================================
# 5. CREATE LAG FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Creating lag features")
print("=" * 60)


# ------------------------------------------------------------
# Previous day
# ------------------------------------------------------------

df["lag_1"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["demand"]
    .shift(1)
)


# ------------------------------------------------------------
# Previous week
# ------------------------------------------------------------

df["lag_7"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["demand"]
    .shift(7)
)


# ------------------------------------------------------------
# Previous two weeks
# ------------------------------------------------------------

df["lag_14"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["demand"]
    .shift(14)
)


# ------------------------------------------------------------
# Previous four weeks
# ------------------------------------------------------------

df["lag_28"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["demand"]
    .shift(28)
)


print("Created:")
print("  lag_1")
print("  lag_7")
print("  lag_14")
print("  lag_28")


# ============================================================
# 6. CREATE ROLLING DEMAND FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Creating rolling demand features")
print("=" * 60)


# ------------------------------------------------------------
# 7-day rolling average
# ------------------------------------------------------------

df["rolling_mean_7"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["demand"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )
)


# ------------------------------------------------------------
# 14-day rolling average
# ------------------------------------------------------------

df["rolling_mean_14"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["demand"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=14,
            min_periods=14
        )
        .mean()
    )
)


# ------------------------------------------------------------
# 28-day rolling average
# ------------------------------------------------------------

df["rolling_mean_28"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["demand"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=28,
            min_periods=28
        )
        .mean()
    )
)


print("Created:")
print("  rolling_mean_7")
print("  rolling_mean_14")
print("  rolling_mean_28")


# ============================================================
# 7. DEMAND VOLATILITY
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Creating demand volatility feature")
print("=" * 60)


df["rolling_std_7"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["demand"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .std()
    )
)


print("Created:")
print("  rolling_std_7")


# ============================================================
# 8. PRICE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Creating price features")
print("=" * 60)


# Make sure sell_price is numeric

df["sell_price"] = pd.to_numeric(
    df["sell_price"],
    errors="coerce"
)


# ------------------------------------------------------------
# Price percentage change
# ------------------------------------------------------------

df["price_change"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["sell_price"]
    .pct_change()
)


# Replace infinity values

df["price_change"] = df["price_change"].replace(
    [np.inf, -np.inf],
    np.nan
)


print("Created:")
print("  price_change")


# ============================================================
# 9. PRICE ROLLING AVERAGE
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Creating price rolling average")
print("=" * 60)


df["price_rolling_mean_7"] = (
    df.groupby(
        ["store_id", "item_id"],
        observed=True
    )["sell_price"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )
)


print("Created:")
print("  price_rolling_mean_7")


# ============================================================
# 10. ADDITIONAL TIME FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Creating additional time features")
print("=" * 60)


# First day of month

df["is_month_start"] = (
    df["date"]
    .dt
    .is_month_start
    .astype("int8")
)


# Last day of month

df["is_month_end"] = (
    df["date"]
    .dt
    .is_month_end
    .astype("int8")
)


# Number of days since beginning of dataset

df["days_since_start"] = (
    df["date"] - df["date"].min()
).dt.days


print("Created:")
print("  is_month_start")
print("  is_month_end")
print("  days_since_start")


# ============================================================
# 11. REMOVE ROWS WITHOUT SUFFICIENT HISTORY
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Removing rows without sufficient history")
print("=" * 60)


before_rows = len(df)


# We need at least 28 previous days
# for lag_28 and rolling_mean_28.

df = df.dropna(
    subset=[
        "lag_28",
        "rolling_mean_28"
    ]
).reset_index(drop=True)


after_rows = len(df)


print("Rows before:", before_rows)
print("Rows after:", after_rows)
print("Rows removed:", before_rows - after_rows)


# ============================================================
# 12. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: Handling remaining missing values")
print("=" * 60)


numeric_feature_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "price_change",
    "price_rolling_mean_7"
]


for column in numeric_feature_columns:

    if column in df.columns:

        df[column] = df[column].fillna(0)


print("Missing numerical feature values handled.")


# ============================================================
# 13. DEFINE MODEL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: Selecting model features")
print("=" * 60)


feature_columns = [

    # --------------------------------------------------------
    # Product and store information
    # --------------------------------------------------------

    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",


    # --------------------------------------------------------
    # Calendar features
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
    # Event features
    # --------------------------------------------------------

    "is_event",
    "snap",


    # --------------------------------------------------------
    # Price features
    # --------------------------------------------------------

    "sell_price",
    "price_missing",
    "price_change",
    "price_rolling_mean_7",


    # --------------------------------------------------------
    # Demand history
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
# 14. CHECK FEATURE COLUMNS
# ============================================================

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]


if missing_features:

    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )


print("All feature columns are present.")


# ============================================================
# 15. CREATE FINAL FEATURE DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 13: Creating final feature dataset")
print("=" * 60)


# Keep important identification columns
# plus model features.

model_columns = [
    "date",
    "d",
    "item_id",
    "store_id",
    "demand"
]


for column in feature_columns:

    if column not in model_columns:

        model_columns.append(column)


features = df[model_columns].copy()


print("Final feature dataset created.")

print(
    "Final shape:",
    features.shape
)


# ============================================================
# 16. CHECK FINAL DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 14: Final data quality check")
print("=" * 60)


print("\nMissing values:")

missing_summary = (
    features.isnull()
    .sum()
)

print(
    missing_summary[
        missing_summary > 0
    ]
)


print("\nFeature dataset preview:")

print(
    features.head()
)


# ============================================================
# 17. SAVE FEATURE DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 15: Saving feature dataset")
print("=" * 60)


features.to_parquet(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY! ✅")
print("=" * 60)


print("\nFinal shape:")
print(features.shape)


print("\nOutput file:")
print(OUTPUT_FILE)


print("\nTarget:")
print(target_column)


print("\nNumber of model features:")
print(len(feature_columns))


print("\nModel features:")

for feature in feature_columns:

    print(" -", feature)