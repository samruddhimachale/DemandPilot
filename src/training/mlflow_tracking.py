import json
from pathlib import Path

import mlflow


# ============================================================
# DemandPilot - MLflow Experiment Tracking
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

METRICS_FILE = REPORTS_DIR / "xgboost_metrics.json"
MODEL_FILE = MODELS_DIR / "xgboost_demand_model.json"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "xgboost_feature_importance.csv"
FINAL_SUMMARY_FILE = REPORTS_DIR / "final_model_summary.json"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"


# ============================================================
# STEP 1: Configure MLflow
# ============================================================

print("=" * 60)
print("STEP 1: Configuring MLflow")
print("=" * 60)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

print("Tracking URI:")
print(tracking_uri)

print("\nArtifact directory:")
print(ARTIFACT_DIR)


# ============================================================
# STEP 2: Set experiment
# ============================================================

EXPERIMENT_NAME = "DemandPilot-Forecasting"

mlflow.set_experiment(EXPERIMENT_NAME)

print("\nExperiment:")
print(EXPERIMENT_NAME)


# ============================================================
# STEP 3: Load results
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Loading model results")
print("=" * 60)

with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

with open(FINAL_SUMMARY_FILE, "r") as f:
    final_summary = json.load(f)

print("XGBoost metrics loaded.")
print("Final model summary loaded.")


# ============================================================
# STEP 4: Read actual metric names
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Reading evaluation metrics")
print("=" * 60)

print("\nMetrics found in xgboost_metrics.json:")

for key, value in metrics.items():
    print(f"  {key}: {value}")


# Your actual JSON uses these names:
mae = float(metrics["mae"])
rmse = float(metrics["rmse"])
smape = float(metrics["smape_percent"])
wape = float(metrics["wape_percent"])


print("\nMetrics successfully identified:")
print(f"MAE   = {mae}")
print(f"RMSE  = {rmse}")
print(f"sMAPE = {smape}%")
print(f"WAPE  = {wape}%")


# ============================================================
# STEP 5: Start MLflow run
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Starting MLflow run")
print("=" * 60)

with mlflow.start_run(run_name="XGBoost_Final_Model") as run:

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_param(
        "model_type",
        "XGBoost"
    )

    mlflow.log_param(
        "dataset",
        "M5 Forecasting Accuracy"
    )

    mlflow.log_param(
        "validation_type",
        "time_based"
    )

    mlflow.log_param(
        "validation_horizon_days",
        28
    )

    mlflow.log_param(
        "training_window_days",
        90
    )

    mlflow.log_param(
        "n_estimators",
        500
    )

    mlflow.log_param(
        "learning_rate",
        0.05
    )

    mlflow.log_param(
        "max_depth",
        8
    )

    mlflow.log_param(
        "min_child_weight",
        5
    )

    mlflow.log_param(
        "subsample",
        0.8
    )

    mlflow.log_param(
        "colsample_bytree",
        0.8
    )

    mlflow.log_param(
        "reg_lambda",
        1.0
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print("\nLogging metrics...")

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("sMAPE", smape)
    mlflow.log_metric("WAPE", wape)


    # --------------------------------------------------------
    # Tags
    # --------------------------------------------------------

    mlflow.set_tag(
        "project",
        "DemandPilot"
    )

    mlflow.set_tag(
        "task",
        "Retail Demand Forecasting"
    )

    mlflow.set_tag(
        "model_status",
        "Final Candidate"
    )

    mlflow.set_tag(
        "primary_metric",
        "MAE"
    )

    mlflow.set_tag(
        "secondary_metric",
        "WAPE"
    )

    mlflow.set_tag(
        "validation_strategy",
        "28-day time-based validation"
    )


    # --------------------------------------------------------
    # Log model artifact
    # --------------------------------------------------------

    print("\nLogging artifacts...")

    if MODEL_FILE.exists():

        mlflow.log_artifact(
            str(MODEL_FILE),
            artifact_path="model"
        )

        print("✓ XGBoost model logged")

    else:

        print("⚠ XGBoost model file not found")


    # --------------------------------------------------------
    # Log feature importance
    # --------------------------------------------------------

    if FEATURE_IMPORTANCE_FILE.exists():

        mlflow.log_artifact(
            str(FEATURE_IMPORTANCE_FILE),
            artifact_path="feature_importance"
        )

        print("✓ Feature importance logged")

    else:

        print("⚠ Feature importance file not found")


    # --------------------------------------------------------
    # Log metrics report
    # --------------------------------------------------------

    if METRICS_FILE.exists():

        mlflow.log_artifact(
            str(METRICS_FILE),
            artifact_path="reports"
        )

        print("✓ XGBoost metrics logged")


    # --------------------------------------------------------
    # Log final model summary
    # --------------------------------------------------------

    if FINAL_SUMMARY_FILE.exists():

        mlflow.log_artifact(
            str(FINAL_SUMMARY_FILE),
            artifact_path="reports"
        )

        print("✓ Final model summary logged")


    # --------------------------------------------------------
    # Run information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MLFLOW RUN COMPLETED")
    print("=" * 60)

    print("Run ID:")
    print(run.info.run_id)

    print("\nExperiment ID:")
    print(run.info.experiment_id)

    print("\nModel:")
    print("XGBoost")

    print("\nMetrics:")
    print(f"MAE   : {mae}")
    print(f"RMSE  : {rmse}")
    print(f"sMAPE : {smape}%")
    print(f"WAPE  : {wape}%")

    print("\nTracking database:")
    print(MLFLOW_DB)

    print("\nArtifact directory:")
    print(ARTIFACT_DIR)

    print("\n" + "=" * 60)
    print("MLFLOW TRACKING COMPLETED! ✅")
    print("=" * 60)