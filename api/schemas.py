from pydantic import BaseModel


class DemandRequest(BaseModel):

    item_id: int
    dept_id: int
    cat_id: int
    store_id: int
    state_id: int

    day_of_week: int
    day_of_month: int
    week_of_year: int
    month: int
    quarter: int
    year: int

    is_weekend: int
    is_month_start: int
    is_month_end: int

    is_event: int
    snap: int

    sell_price: float
    price_missing: int
    price_change: float
    price_rolling_mean_7: float

    lag_1: float
    lag_7: float
    lag_14: float
    lag_28: float

    rolling_mean_7: float
    rolling_mean_14: float
    rolling_mean_28: float
    rolling_std_7: float

    days_since_start: int


class DemandResponse(BaseModel):

    predicted_demand: float