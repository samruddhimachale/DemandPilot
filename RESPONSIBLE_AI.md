# Responsible AI — DemandPilot

## 1. Purpose

DemandPilot is a retail demand forecasting and inventory-support system designed to predict future product demand using historical sales, pricing, calendar, event, and promotional information.

The system is intended to support inventory planning and decision-making. It is not designed to make autonomous purchasing or inventory decisions.

---

## 2. Intended Use

DemandPilot can be used to:

- Forecast retail product demand.
- Identify demand trends and seasonal patterns.
- Support inventory planning.
- Help identify potential stock-out and overstock risks.
- Monitor forecasting performance over time.
- Analyze which features influence model predictions.

The final inventory decision should remain under human supervision.

---

## 3. Data Considerations

DemandPilot uses the M5 retail forecasting dataset.

The dataset contains historical retail sales, product information, store information, prices, calendar information, events, and SNAP-related indicators.

The project does not use personally identifiable customer information.

### Data quality considerations

The pipeline includes preprocessing and feature-engineering steps to handle:

- Missing prices
- Calendar information
- Historical demand
- Lag features
- Rolling statistics
- Events and promotions
- Time-based features

Data versioning is implemented using DVC with Amazon S3 as the remote storage backend.

---

## 4. Model

The final forecasting model is an XGBoost regression model.

The model uses historical demand, rolling demand statistics, lag features, calendar information, pricing information, events, and SNAP-related features.

The model was compared against a seasonal-naive baseline.

### Validation

The model uses a time-based validation strategy rather than a random train-test split.

Validation period:

**2016-03-28 to 2016-04-24**

The final model was evaluated using:

- MAE
- RMSE
- WAPE
- sMAPE

MAE and WAPE are treated as the primary evaluation metrics because the dataset contains a large number of zero-demand observations.

---

## 5. Zero-Demand Limitation

A significant portion of the validation data contains zero demand.

During validation:

- Zero-demand observations: 480,460
- Zero-demand percentage: 56.28%

Because sMAPE can become very large when actual demand is zero or close to zero, sMAPE is not used as the primary model-selection metric.

MAE and WAPE are therefore emphasized for model evaluation.

---

## 6. Model Performance

### Seasonal-naive baseline

- MAE: 1.2054
- RMSE: 2.6601
- WAPE: 86.94%

### XGBoost

- MAE: 0.9940
- RMSE: 2.0223
- WAPE: 71.70%

Compared with the baseline, the XGBoost model achieved:

- 17.53% improvement in MAE
- 23.98% improvement in RMSE
- 17.53% improvement in WAPE

These measurements are based on the project's 28-day validation period.

---

## 7. Explainability

SHAP is used to explain the model's predictions.

The top features identified through mean absolute SHAP values include:

1. rolling_mean_7
2. rolling_mean_14
3. rolling_mean_28
4. lag_1
5. day_of_week
6. days_since_start
7. lag_28
8. rolling_std_7
9. snap
10. day_of_month

The results indicate that recent historical demand and temporal patterns have a substantial influence on model predictions.

SHAP explanations are intended to improve transparency and help developers and users understand model behavior.

---

## 8. Monitoring

DemandPilot includes automated model monitoring.

The monitoring pipeline tracks:

- MAE
- WAPE
- Zero-demand rate

The project also includes data-drift detection using Evidently.

Reference and current time periods are compared to identify changes in important model inputs.

---

## 9. Alerting

The monitoring system uses predefined thresholds to identify potential model-performance degradation.

Current thresholds:

- Maximum MAE: 1.20
- Maximum WAPE: 90%

If these thresholds are exceeded, the monitoring system reports an alert condition.

These thresholds should be reviewed and recalibrated as additional production data becomes available.

---

## 10. Human-in-the-Loop Decision Making

DemandPilot is intended as a decision-support system.

Forecasts should not automatically trigger:

- Purchase orders
- Supplier decisions
- Inventory transfers
- Product discontinuation

Business users should consider additional information such as:

- Current inventory
- Supplier lead time
- Safety-stock requirements
- Budget constraints
- Promotions
- Supply disruptions
- Business priorities

before taking action.

---

## 11. Known Limitations

The model has several limitations:

- Historical patterns may not represent future market conditions.
- Unexpected events can reduce forecast accuracy.
- New products may have insufficient historical information.
- Sudden demand changes may not be captured immediately.
- Zero-demand observations make some percentage-based metrics difficult to interpret.
- The M5 dataset represents a specific retail environment and may not generalize directly to every retailer.
- Forecast accuracy may change as product, store, pricing, and customer behavior changes.

---

## 12. Drift and Retraining

Model performance and input-data distributions should be monitored continuously.

If significant data drift or performance degradation is detected, the model should be investigated and potentially retrained using more recent data.

Retraining should include:

1. Data validation
2. Feature generation
3. Model training
4. Model evaluation
5. Comparison against the current model
6. Explainability analysis
7. Approval before deployment

---

## 13. Security and Privacy

The project does not intentionally process personally identifiable customer information.

The deployed API should be protected using appropriate authentication, authorization, network controls, logging, and rate limiting before being exposed to production users.

AWS credentials and other secrets must never be committed to the Git repository.

---

## 14. Responsible Deployment

Before production deployment, the following should be considered:

- Validate data quality.
- Verify model performance on recent data.
- Review drift reports.
- Review SHAP explanations.
- Monitor prediction errors.
- Establish model rollback procedures.
- Keep human oversight for inventory decisions.
- Periodically retrain and reevaluate the model.

---

## 15. Summary

DemandPilot follows a responsible ML approach by combining:

- Data versioning
- Reproducible pipelines
- Time-based model validation
- Performance monitoring
- Data-drift detection
- Threshold-based alerts
- SHAP explainability
- Human oversight
- Documented model limitations

The system is designed to assist inventory planning rather than replace human decision-making.