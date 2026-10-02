from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data for drift detection...")

df = pd.read_parquet(
    FEATURES_PATH,
    columns=[
        "date",
        "demand",
        "sell_price",
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
)

df["date"] = pd.to_datetime(df["date"])


# ============================================================
# REFERENCE AND CURRENT DATA
# ============================================================

reference_start = pd.Timestamp("2016-01-01")
reference_end = pd.Timestamp("2016-02-28")

current_start = pd.Timestamp("2016-03-28")
current_end = pd.Timestamp("2016-04-24")

reference = df[
    (df["date"] >= reference_start)
    & (df["date"] <= reference_end)
].copy()

current = df[
    (df["date"] >= current_start)
    & (df["date"] <= current_end)
].copy()


# ============================================================
# REMOVE DATE COLUMN
# ============================================================

reference = reference.drop(columns=["date"])
current = current.drop(columns=["date"])


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

reference = reference.fillna(0)
current = current.fillna(0)


# ============================================================
# DRIFT REPORT
# ============================================================

print("Generating Evidently drift report...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

result = report.run(
    reference_data=reference,
    current_data=current
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    REPORTS_DIR
    / "data_drift_report.html"
)

result.save_html(
    str(report_path)
)


# ============================================================
# OUTPUT
# ============================================================

print("\nDrift detection completed successfully.")

print(
    f"Reference observations: "
    f"{len(reference):,}"
)

print(
    f"Current observations: "
    f"{len(current):,}"
)

print(f"\nReport saved to:")
print(report_path)