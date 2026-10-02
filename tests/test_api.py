from fastapi.testclient import TestClient

import api.main as main
from api.main import app


client = TestClient(app)


def test_health(monkeypatch):

    monkeypatch.setattr(main, "model_loaded", True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction(monkeypatch):

    class FakeModel:

        def predict(self, data):
            return [5.0]

    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "model_loaded", True)

    payload = {
        "item_id": 1,
        "dept_id": 1,
        "cat_id": 1,
        "store_id": 1,
        "state_id": 1,
        "day_of_week": 1,
        "day_of_month": 15,
        "week_of_year": 20,
        "month": 5,
        "quarter": 2,
        "year": 2016,
        "is_weekend": 0,
        "is_month_start": 0,
        "is_month_end": 0,
        "is_event": 0,
        "snap": 0,
        "sell_price": 10.0,
        "price_missing": 0,
        "price_change": 0.0,
        "price_rolling_mean_7": 10.0,
        "lag_1": 5.0,
        "lag_7": 5.0,
        "lag_14": 5.0,
        "lag_28": 5.0,
        "rolling_mean_7": 5.0,
        "rolling_mean_14": 5.0,
        "rolling_mean_28": 5.0,
        "rolling_std_7": 1.0,
        "days_since_start": 2000
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200
    assert response.json()["predicted_demand"] == 5.0