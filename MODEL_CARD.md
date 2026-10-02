# DemandPilot — Model Card

## 1. Model Overview

**Project:** DemandPilot — Retail Demand and Inventory Forecasting

**Model:** XGBoost Regressor

**Task:** Retail demand forecasting

**Problem Type:** Regression / Time-series forecasting

**Primary Objective:** Predict future product demand to support inventory planning and reduce the risk of stock-outs and overstocking.

---

## 2. Intended Use

The model is intended to:

- Forecast retail demand.
- Identify demand patterns.
- Support inventory planning.
- Assist stock-out and overstock analysis.
- Provide demand estimates for business decision-making.

The model is a decision-support tool and should not independently make purchasing or inventory decisions.

---

## 3. Dataset

The model was trained using the **M5 Forecasting Accuracy** dataset.

The dataset contains:

- Historical daily sales
- Product information
- Department and category information
- Store information
- State information
- Selling prices
- Calendar information
- Events
- SNAP indicators

The project uses DVC for dataset and pipeline versioning.

Raw data is stored in an Amazon S3-backed DVC remote.

---

## 4. Data Processing

The preprocessing pipeline transforms the original M5 data into a model-ready format.

Major preprocessing steps include:

- Combining sales, calendar, and price information.
- Converting sales data into long format.
- Handling missing prices.
- Adding calendar attributes.
- Adding event information.
- Adding SNAP indicators.

Feature engineering generates:

### Lag features

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

### Rolling features

- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_mean_28`
- `rolling_std_7`

### Calendar features

- `day_of_week`
- `day_of_month`
- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_weekend`
- `is_month_start`
- `is_month_end`

### Price features

- `sell_price`
- `price_missing`
- `price_change`
- `price_rolling_mean_7`

### Other features

- `is_event`
- `snap`
- `days_since_start`

---

## 5. Model Architecture

DemandPilot uses an XGBoost regression model.

### Model configuration

```text
n_estimators = 500
learning_rate = 0.05
max_depth = 8
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
reg_lambda = 1.0
objective = reg:squarederror
eval_metric = mae
random_state = 42