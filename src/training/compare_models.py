from pathlib import Path
import json
import pandas as pd


# ============================================================
# DEMANDPILOT - MODEL COMPARISON
# ============================================================

print("=" * 60)
print("DEMANDPILOT MODEL COMPARISON")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports"

BASELINE_FILE = REPORT_DIR / "baseline_metrics.json"
XGBOOST_FILE = REPORT_DIR / "xgboost_metrics.json"

COMPARISON_FILE = REPORT_DIR / "model_comparison.csv"
SUMMARY_FILE = REPORT_DIR / "model_selection.json"


# ============================================================
# 2. LOAD METRICS
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Loading model metrics")
print("=" * 60)


with open(BASELINE_FILE, "r") as file:
    baseline = json.load(file)


with open(XGBOOST_FILE, "r") as file:
    xgboost = json.load(file)


print("Baseline metrics loaded.")
print("XGBoost metrics loaded.")


# ============================================================
# 3. CREATE COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Creating comparison")
print("=" * 60)


comparison = pd.DataFrame({

    "Model": [
        baseline["model"],
        xgboost["model"]
    ],

    "MAE": [
        baseline["mae"],
        xgboost["mae"]
    ],

    "RMSE": [
        baseline["rmse"],
        xgboost["rmse"]
    ],

    "sMAPE (%)": [
        baseline["smape_percent"],
        xgboost["smape_percent"]
    ],

    "WAPE (%)": [
        baseline["wape_percent"],
        xgboost["wape_percent"]
    ]
})


print("\nModel comparison:")

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 4. CALCULATE IMPROVEMENT
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Calculating improvement")
print("=" * 60)


baseline_mae = baseline["mae"]
xgb_mae = xgboost["mae"]

baseline_rmse = baseline["rmse"]
xgb_rmse = xgboost["rmse"]

baseline_wape = baseline["wape_percent"]
xgb_wape = xgboost["wape_percent"]


# Lower is better.

mae_improvement = (
    (baseline_mae - xgb_mae)
    / baseline_mae
    * 100
)


rmse_improvement = (
    (baseline_rmse - xgb_rmse)
    / baseline_rmse
    * 100
)


wape_improvement = (
    (baseline_wape - xgb_wape)
    / baseline_wape
    * 100
)


print(
    f"\nMAE improvement:  {mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: {rmse_improvement:.2f}%"
)

print(
    f"WAPE improvement: {wape_improvement:.2f}%"
)


# ============================================================
# 5. SAVE COMPARISON TABLE
# ============================================================

comparison.to_csv(
    COMPARISON_FILE,
    index=False
)


print("\nComparison saved to:")

print(
    COMPARISON_FILE
)


# ============================================================
# 6. DETERMINE METRIC RESULTS
# ============================================================

xgb_better_mae = (
    xgb_mae < baseline_mae
)

xgb_better_rmse = (
    xgb_rmse < baseline_rmse
)

xgb_better_wape = (
    xgb_wape < baseline_wape
)


# ============================================================
# 7. MODEL SELECTION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Model selection analysis")
print("=" * 60)


print(
    "\nXGBoost better on MAE:",
    xgb_better_mae
)

print(
    "XGBoost better on RMSE:",
    xgb_better_rmse
)

print(
    "XGBoost better on WAPE:",
    xgb_better_wape
)


# We use MAE as the primary model-selection metric.
#
# MAE is easy to interpret:
# average absolute forecasting error in demand units.

if xgb_better_mae:

    selected_model = "XGBoost"

else:

    selected_model = baseline["model"]


print(
    "\nPrimary selection metric: MAE"
)

print(
    "Selected model:",
    selected_model
)


# ============================================================
# 8. SAVE MODEL SELECTION RECORD
# ============================================================

selection = {

    "primary_metric": "MAE",

    "selection_rule": (
        "Lower MAE is preferred"
    ),

    "selected_model": selected_model,

    "baseline": {
        "model": baseline["model"],
        "mae": baseline["mae"],
        "rmse": baseline["rmse"],
        "smape_percent": baseline["smape_percent"],
        "wape_percent": baseline["wape_percent"]
    },

    "xgboost": {
        "model": xgboost["model"],
        "mae": xgboost["mae"],
        "rmse": xgboost["rmse"],
        "smape_percent": xgboost["smape_percent"],
        "wape_percent": xgboost["wape_percent"]
    },

    "improvement": {
        "mae_percent": mae_improvement,
        "rmse_percent": rmse_improvement,
        "wape_percent": wape_improvement
    }
}


with open(
    SUMMARY_FILE,
    "w"
) as file:

    json.dump(
        selection,
        file,
        indent=4
    )


print("\nModel selection record saved to:")

print(
    SUMMARY_FILE
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETED! ✅")
print("=" * 60)


print("\nFiles created:")

print(
    "1.",
    COMPARISON_FILE
)

print(
    "2.",
    SUMMARY_FILE
)


print("\nSelected model based on MAE:")

print(
    selected_model
)