from pydantic import BaseModel


class DemandRequest(BaseModel):

    # Product and store identifiers
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str

    # Calendar features
    day_of_week: int
    day_of_month: int
    week_of_year: int
    month: int
    quarter: int
    year: int

    is_weekend: int
    is_month_start: int
    is_month_end: int

    # Event features
    is_event: int
    snap: int

    # Price features
    sell_price: float
    price_missing: int
    price_change: float
    price_rolling_mean_7: float

    # Lag features
    lag_1: float
    lag_7: float
    lag_14: float
    lag_28: float

    # Rolling features
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_mean_28: float
    rolling_std_7: float

    # Trend
    days_since_start: int


class DemandResponse(BaseModel):

    predicted_demand: float