from pathlib import Path
import json

import pandas as pd
import numpy as np
import pyarrow.parquet as pq


# ============================================================
# DEMANDPILOT - MODEL DIAGNOSTICS
# ============================================================

print("=" * 60)
print("DEMANDPILOT MODEL DIAGNOSTICS")
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

DIAGNOSTIC_FILE = (
    REPORT_DIR
    / "model_diagnostics.json"
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


print("Dataset shape:", df.shape)


# ============================================================
# 3. IDENTIFY VALIDATION PERIOD
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Identifying validation period")
print("=" * 60)


max_date = df["date"].max()

validation_start = (
    max_date
    - pd.Timedelta(days=27)
)


validation_df = df[
    df["date"] >= validation_start
].copy()


print("Validation start:")
print(validation_start)


print("Validation end:")
print(max_date)


print(
    "Validation rows:",
    len(validation_df)
)


# ============================================================
# 4. DEMAND DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Analyzing demand distribution")
print("=" * 60)


total_rows = len(validation_df)


zero_demand_rows = (
    validation_df["demand"] == 0
).sum()


low_demand_rows = (
    validation_df["demand"] <= 1
).sum()


positive_demand_rows = (
    validation_df["demand"] > 0
).sum()


zero_percentage = (
    zero_demand_rows
    / total_rows
    * 100
)


low_demand_percentage = (
    low_demand_rows
    / total_rows
    * 100
)


positive_percentage = (
    positive_demand_rows
    / total_rows
    * 100
)


print(
    f"\nZero-demand observations: "
    f"{zero_demand_rows:,}"
)


print(
    f"Zero-demand percentage: "
    f"{zero_percentage:.2f}%"
)


print(
    f"\nDemand <= 1 observations: "
    f"{low_demand_rows:,}"
)


print(
    f"Demand <= 1 percentage: "
    f"{low_demand_percentage:.2f}%"
)


print(
    f"\nPositive-demand observations: "
    f"{positive_demand_rows:,}"
)


print(
    f"Positive-demand percentage: "
    f"{positive_percentage:.2f}%"
)


# ============================================================
# 5. DEMAND STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Demand statistics")
print("=" * 60)


print(
    validation_df["demand"].describe()
)


# ============================================================
# 6. BASELINE PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Checking baseline performance")
print("=" * 60)


baseline_file = (
    REPORT_DIR
    / "baseline_metrics.json"
)


xgboost_file = (
    REPORT_DIR
    / "xgboost_metrics.json"
)


with open(
    baseline_file,
    "r"
) as file:

    baseline = json.load(file)


with open(
    xgboost_file,
    "r"
) as file:

    xgboost = json.load(file)


print("\nBaseline:")

print(
    "MAE:",
    round(
        baseline["mae"],
        4
    )
)


print(
    "RMSE:",
    round(
        baseline["rmse"],
        4
    )
)


print(
    "sMAPE:",
    round(
        baseline["smape_percent"],
        2
    ),
    "%"
)


print(
    "WAPE:",
    round(
        baseline["wape_percent"],
        2
    ),
    "%"
)


print("\nXGBoost:")

print(
    "MAE:",
    round(
        xgboost["mae"],
        4
    )
)


print(
    "RMSE:",
    round(
        xgboost["rmse"],
        4
    )
)


print(
    "sMAPE:",
    round(
        xgboost["smape_percent"],
        2
    ),
    "%"
)


print(
    "WAPE:",
    round(
        xgboost["wape_percent"],
        2
    ),
    "%"
)


# ============================================================
# 7. MODEL IMPROVEMENT
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Calculating model improvement")
print("=" * 60)


mae_improvement = (
    (
        baseline["mae"]
        - xgboost["mae"]
    )
    / baseline["mae"]
    * 100
)


rmse_improvement = (
    (
        baseline["rmse"]
        - xgboost["rmse"]
    )
    / baseline["rmse"]
    * 100
)


wape_improvement = (
    (
        baseline["wape_percent"]
        - xgboost["wape_percent"]
    )
    / baseline["wape_percent"]
    * 100
)


print(
    f"\nMAE improvement: "
    f"{mae_improvement:.2f}%"
)


print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)


print(
    f"WAPE improvement: "
    f"{wape_improvement:.2f}%"
)


# ============================================================
# 8. DIAGNOSTIC INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Diagnostic interpretation")
print("=" * 60)


if zero_percentage > 10:

    smape_note = (
        "Validation data contains a substantial number "
        "of zero-demand observations. sMAPE can become "
        "very large for zero or near-zero actual demand."
    )

else:

    smape_note = (
        "Zero-demand observations are relatively limited. "
        "The sMAPE value should be investigated further."
    )


print("\nObservation:")

print(smape_note)


print(
    "\nFor DemandPilot, the primary evaluation metrics "
    "will be MAE and WAPE."
)


# ============================================================
# 9. SAVE DIAGNOSTICS
# ============================================================

diagnostics = {

    "validation_rows": int(
        total_rows
    ),

    "zero_demand_rows": int(
        zero_demand_rows
    ),

    "zero_demand_percentage": float(
        zero_percentage
    ),

    "low_demand_rows": int(
        low_demand_rows
    ),

    "low_demand_percentage": float(
        low_demand_percentage
    ),

    "positive_demand_percentage": float(
        positive_percentage
    ),

    "baseline_mae": float(
        baseline["mae"]
    ),

    "xgboost_mae": float(
        xgboost["mae"]
    ),

    "baseline_rmse": float(
        baseline["rmse"]
    ),

    "xgboost_rmse": float(
        xgboost["rmse"]
    ),

    "baseline_wape": float(
        baseline["wape_percent"]
    ),

    "xgboost_wape": float(
        xgboost["wape_percent"]
    ),

    "mae_improvement_percent": float(
        mae_improvement
    ),

    "rmse_improvement_percent": float(
        rmse_improvement
    ),

    "wape_improvement_percent": float(
        wape_improvement
    ),

    "primary_metrics": [
        "MAE",
        "WAPE"
    ],

    "smape_note": smape_note
}


with open(
    DIAGNOSTIC_FILE,
    "w"
) as file:

    json.dump(
        diagnostics,
        file,
        indent=4
    )


# ============================================================
# 10. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("MODEL DIAGNOSTICS COMPLETED! ✅")
print("=" * 60)


print("\nDiagnostic report:")

print(
    DIAGNOSTIC_FILE
)