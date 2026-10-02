from pathlib import Path

import pandas as pd
import xgboost as xgb

from fastapi import FastAPI

from api.schemas import (
    DemandRequest,
    DemandResponse
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_demand_model.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = xgb.XGBRegressor()

model.load_model(
    MODEL_PATH
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="DemandPilot API",
    description="Retail demand forecasting API",
    version="1.0.0"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "DemandPilot API is running",
        "model": "XGBoost"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=DemandResponse
)
def predict(
    request: DemandRequest
):

    # --------------------------------------------------------
    # Create dictionary from request
    # --------------------------------------------------------

    data = request.model_dump()


    # --------------------------------------------------------
    # IMPORTANT:
    # Keep exactly the same feature order used during training
    # --------------------------------------------------------

    feature_order = [
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


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [data],
        columns=feature_order
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return DemandResponse(
        predicted_demand=float(prediction)
    )