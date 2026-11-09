from pathlib import Path
import json

import pandas as pd
import xgboost as xgb

from fastapi import FastAPI, HTTPException

from api.schemas import DemandRequest, DemandResponse


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_demand_model.json"

MAPPING_PATH = PROJECT_ROOT / "reports" / "category_mappings.json"


# ============================================================
# LOAD MODEL
# ============================================================

model = xgb.XGBRegressor()
model_loaded = False

if MODEL_PATH.exists():
    model.load_model(MODEL_PATH)
    model_loaded = True


# ============================================================
# LOAD CATEGORY MAPPINGS
# ============================================================

category_mappings = {}
mappings_loaded = False

if MAPPING_PATH.exists():

    with open(MAPPING_PATH, "r") as file:
        mapping_data = json.load(file)

    category_mappings = mapping_data["mappings"]
    mappings_loaded = True


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_ORDER = [
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


CATEGORICAL_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="DemandPilot API",
    description="Retail demand forecasting API",
    version="1.1.0"
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
        "model_loaded": model_loaded,
        "mappings_loaded": mappings_loaded
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=DemandResponse
)
def predict(request: DemandRequest):

    # Check model and mappings

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="XGBoost model is not available."
        )

    if not mappings_loaded:
        raise HTTPException(
            status_code=503,
            detail="Category mappings are not available."
        )

    # Convert request into dictionary

    data = request.model_dump()

    # Encode categorical identifiers

    for column in CATEGORICAL_COLUMNS:

        original_value = data[column]

        mapping = category_mappings[column]

        if original_value not in mapping:

            raise HTTPException(
                status_code=422,
                detail=f"Unknown {column}: {original_value}"
            )

        data[column] = mapping[original_value]

    # Create input DataFrame in training feature order

    input_data = pd.DataFrame(
        [data],
        columns=FEATURE_ORDER
    )

    # Generate prediction

    prediction = model.predict(input_data)[0]

    # Demand cannot be negative

    prediction = max(float(prediction), 0.0)

    return DemandResponse(
        predicted_demand=prediction
    )