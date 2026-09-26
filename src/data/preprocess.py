from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FILE PATHS
# ============================================================

CALENDAR_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE = RAW_DIR / "sell_prices.csv"
TRAIN_FILE = RAW_DIR / "sales_train_validation.csv"
EVALUATION_FILE = RAW_DIR / "sales_train_evaluation.csv"


# ============================================================
# 3. LOAD CALENDAR DATA
# ============================================================

print("=" * 60)
print("STEP 1: Loading calendar data")
print("=" * 60)

calendar = pd.read_csv(
    RAW_DIR / "calendar.csv"
)

print("Calendar shape:", calendar.shape)


# ------------------------------------------------------------
# M5 calendar does not contain a "d" column.
# Create d_1, d_2, ..., d_1969 based on row order.
# ------------------------------------------------------------

calendar["d"] = [
    f"d_{i}"
    for i in range(1, len(calendar) + 1)
]


# Convert date to datetime
calendar["date"] = pd.to_datetime(
    calendar["date"]
)


# ------------------------------------------------------------
# Create event indicator
# ------------------------------------------------------------

calendar["is_event"] = (
    calendar["event_name_1"].notna()
    | calendar["event_name_2"].notna()
).astype("int8")


# ------------------------------------------------------------
# Combine event names
# ------------------------------------------------------------

calendar["event_name"] = (
    calendar["event_name_1"]
    .fillna("")
    .astype(str)
    + " "
    + calendar["event_name_2"]
    .fillna("")
    .astype(str)
).str.strip()


# ------------------------------------------------------------
# Keep required columns
# ------------------------------------------------------------

calendar = calendar[
    [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name",
        "event_type_1",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "is_event"
    ]
]


print("Calendar processed successfully.")
print("Calendar shape:", calendar.shape)

print("\nCalendar sample:")
print(calendar.head())

# ============================================================
# 5. LOAD SELL PRICES
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Loading price data")
print("=" * 60)

prices = pd.read_csv(
    PRICES_FILE,
    dtype={
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": "int32",
        "sell_price": "float32",
    },
)

print("Prices shape:", prices.shape)


# ============================================================
# 6. LOAD SALES DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Loading sales data")
print("=" * 60)

sales = pd.read_csv(
    TRAIN_FILE
)

print("Original sales shape:", sales.shape)


# ============================================================
# 7. IDENTIFY SALES COLUMNS
# ============================================================

id_columns = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

sales_columns = [
    column
    for column in sales.columns
    if column.startswith("d_")
]

print("Number of daily sales columns:", len(sales_columns))


# ============================================================
# 8. DOWNCAST SALES DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Optimizing data types")
print("=" * 60)

for column in id_columns:
    sales[column] = sales[column].astype("category")


for column in sales_columns:
    sales[column] = pd.to_numeric(
        sales[column],
        downcast="unsigned"
    )


print("Data types optimized.")


# ============================================================
# 9. WIDE → LONG FORMAT
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Converting sales data from wide to long")
print("=" * 60)

sales_long = sales.melt(
    id_vars=id_columns,
    value_vars=sales_columns,
    var_name="d",
    value_name="demand",
)

print("Long-format shape:", sales_long.shape)


# ============================================================
# 10. CONVERT DEMAND TYPE
# ============================================================

sales_long["demand"] = sales_long["demand"].astype("uint32")


# ============================================================
# 11. MERGE CALENDAR
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Merging calendar data")
print("=" * 60)

sales_long = sales_long.merge(
    calendar,
    on="d",
    how="left",
    validate="many_to_one",
)

print("After calendar merge:", sales_long.shape)


# ============================================================
# 12. CREATE STATE-SPECIFIC SNAP FEATURE
# ============================================================

print("\nCreating SNAP feature...")


sales_long["snap"] = np.select(
    [
        sales_long["state_id"].eq("CA"),
        sales_long["state_id"].eq("TX"),
        sales_long["state_id"].eq("WI"),
    ],
    [
        sales_long["snap_CA"],
        sales_long["snap_TX"],
        sales_long["snap_WI"],
    ],
    default=0,
).astype("int8")


# ============================================================
# 13. MERGE SELL PRICE
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Merging price data")
print("=" * 60)

sales_long = sales_long.merge(
    prices,
    on=[
        "store_id",
        "item_id",
        "wm_yr_wk",
    ],
    how="left",
    validate="many_to_one",
)

print("After price merge:", sales_long.shape)


# ============================================================
# 14. PRICE MISSINGNESS FLAG
# ============================================================

sales_long["price_missing"] = (
    sales_long["sell_price"].isna()
).astype("int8")


# ============================================================
# 15. BASIC TIME FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Creating basic time features")
print("=" * 60)

sales_long["day_of_week"] = (
    sales_long["date"].dt.dayofweek
).astype("int8")


sales_long["day_of_month"] = (
    sales_long["date"].dt.day
).astype("int8")


sales_long["week_of_year"] = (
    sales_long["date"].dt.isocalendar().week
    .astype("int8")
)


sales_long["quarter"] = (
    sales_long["date"].dt.quarter
).astype("int8")


sales_long["is_weekend"] = (
    sales_long["day_of_week"] >= 5
).astype("int8")


# ============================================================
# 16. SORT DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Sorting time-series data")
print("=" * 60)

sales_long = sales_long.sort_values(
    [
        "store_id",
        "item_id",
        "date",
    ]
).reset_index(drop=True)


# ============================================================
# 17. SELECT FINAL COLUMNS
# ============================================================

final_columns = [
    "date",
    "d",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "demand",
    "sell_price",
    "price_missing",
    "weekday",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "event_name",
    "event_type_1",
    "event_type_2",
    "is_event",
    "snap",
]


sales_long = sales_long[final_columns]


# ============================================================
# 18. REMOVE INVALID ROWS
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Data quality checks")
print("=" * 60)

print("Missing dates:", sales_long["date"].isna().sum())
print("Missing demand:", sales_long["demand"].isna().sum())
print("Missing item IDs:", sales_long["item_id"].isna().sum())
print("Missing store IDs:", sales_long["store_id"].isna().sum())


# Demand should never be negative
negative_demand = (
    sales_long["demand"] < 0
).sum()

print("Negative demand rows:", negative_demand)


if negative_demand > 0:
    raise ValueError(
        "Negative demand values detected."
    )


# ============================================================
# 19. SAVE PROCESSED TRAIN DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: Saving processed data")
print("=" * 60)

train_output = PROCESSED_DIR / "train.parquet"

sales_long.to_parquet(
    train_output,
    engine="pyarrow",
    index=False,
)


print("Saved:", train_output)
print("Final shape:", sales_long.shape)


# ============================================================
# 20. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print("Rows:", len(sales_long))
print("Columns:", len(sales_long.columns))
print(
    "Date range:",
    sales_long["date"].min(),
    "to",
    sales_long["date"].max(),
)

print(
    "Unique products:",
    sales_long["item_id"].nunique()
)

print(
    "Unique stores:",
    sales_long["store_id"].nunique()
)

print(
    "Missing prices:",
    sales_long["sell_price"].isna().sum()
)

print(
    "Output file:",
    train_output
)