from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_demand_model.json"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading XGBoost model...")

model = xgb.XGBRegressor()

model.load_model(
    MODEL_PATH
)


# ============================================================
# GET EXACT MODEL FEATURES
# ============================================================

FEATURES = model.get_booster().feature_names

print(f"Model features: {len(FEATURES)}")


# ============================================================
# LOAD DATA
# ============================================================

print("Loading monitoring data...")

columns = [
    "date",
    "demand"
] + FEATURES

columns = list(dict.fromkeys(columns))

df = pd.read_parquet(
    FEATURES_PATH,
    columns=columns
)

df["date"] = pd.to_datetime(df["date"])


# ============================================================
# VALIDATION PERIOD
# ============================================================

validation_start = pd.Timestamp("2016-03-28")
validation_end = pd.Timestamp("2016-04-24")

validation = df[
    (df["date"] >= validation_start)
    & (df["date"] <= validation_end)
].copy()

print(
    f"Validation observations: {len(validation):,}"
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = validation[FEATURES].copy()

X = X.fillna(0)

y_actual = validation["demand"].to_numpy()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("Generating predictions...")

predictions = model.predict(X)
# ============================================================
# PERFORMANCE METRICS
# ============================================================

errors = y_actual - predictions

absolute_errors = np.abs(errors)

mae = np.mean(absolute_errors)

wape = (
    np.sum(absolute_errors)
    / np.sum(np.abs(y_actual))
    * 100
)

zero_demand_rate = (
    np.mean(y_actual == 0)
    * 100
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

monitoring_df = pd.DataFrame(
    {
        "date": validation["date"].values,
        "actual_demand": y_actual,
        "predicted_demand": predictions,
        "absolute_error": absolute_errors
    }
)

monitoring_path = (
    REPORTS_DIR
    / "monitoring_predictions.csv"
)

monitoring_df.to_csv(
    monitoring_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "validation_start": str(validation_start.date()),
    "validation_end": str(validation_end.date()),
    "observations": int(len(validation)),
    "mae": float(mae),
    "wape_percent": float(wape),
    "zero_demand_rate_percent": float(zero_demand_rate)
}

metrics_df = pd.DataFrame(
    [metrics]
)

metrics_path = (
    REPORTS_DIR
    / "monitoring_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("\nMonitoring completed successfully.")

print(f"MAE: {mae:.4f}")
print(f"WAPE: {wape:.2f}%")
print(
    f"Zero-demand rate: "
    f"{zero_demand_rate:.2f}%"
)

print("\nGenerated files:")
print(monitoring_path)
print(metrics_path)

