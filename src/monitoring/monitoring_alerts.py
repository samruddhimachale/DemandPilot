from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring_metrics.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring_alerts.csv"
)


# ============================================================
# THRESHOLDS
# ============================================================

# These thresholds are based on the current model's
# validation performance and are used to flag degradation.

MAX_MAE = 1.20
MAX_WAPE = 90.0


# ============================================================
# LOAD METRICS
# ============================================================

print("Loading monitoring metrics...")

metrics = pd.read_csv(
    METRICS_PATH
)

mae = float(
    metrics.loc[0, "mae"]
)

wape = float(
    metrics.loc[0, "wape_percent"]
)


# ============================================================
# CHECK PERFORMANCE
# ============================================================

mae_status = (
    "ALERT"
    if mae > MAX_MAE
    else "OK"
)

wape_status = (
    "ALERT"
    if wape > MAX_WAPE
    else "OK"
)


overall_status = (
    "ALERT"
    if "ALERT" in [mae_status, wape_status]
    else "OK"
)


# ============================================================
# CREATE ALERT REPORT
# ============================================================

alerts = pd.DataFrame(
    [
        {
            "metric": "MAE",
            "value": mae,
            "threshold": MAX_MAE,
            "status": mae_status
        },
        {
            "metric": "WAPE",
            "value": wape,
            "threshold": MAX_WAPE,
            "status": wape_status
        },
        {
            "metric": "overall",
            "value": "",
            "threshold": "",
            "status": overall_status
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

alerts.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("\nMonitoring alert check completed.")

print(f"MAE: {mae:.4f} → {mae_status}")
print(f"WAPE: {wape:.2f}% → {wape_status}")
print(f"Overall status: {overall_status}")

print(f"\nAlert report saved to:")
print(OUTPUT_PATH)