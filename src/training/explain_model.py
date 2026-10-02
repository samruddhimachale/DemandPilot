from pathlib import Path

import pandas as pd
import shap
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
# GET MODEL FEATURES
# ============================================================

FEATURES = model.get_booster().feature_names

print(f"Model features: {len(FEATURES)}")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading data...")

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
# SELECT VALIDATION PERIOD
# ============================================================

validation = df[
    (df["date"] >= "2016-03-28")
    & (df["date"] <= "2016-04-24")
].copy()


# ============================================================
# SAMPLE DATA
# ============================================================

# SHAP can be computationally expensive on millions of rows.
# Use a representative sample.

sample_size = min(
    5000,
    len(validation)
)

sample = validation.sample(
    n=sample_size,
    random_state=42
)

X = sample[FEATURES].copy()

# Convert categorical columns to their numeric category codes
categorical_features = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

for feature in categorical_features:
    if str(X[feature].dtype) == "category":
        X[feature] = X[feature].cat.codes

X = X.fillna(0)


# ============================================================
# SHAP EXPLAINER
# ============================================================

print(
    f"Calculating SHAP values for "
    f"{sample_size:,} observations..."
)

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_abs_shap": abs(shap_values).mean(axis=0)
    }
)

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

importance_path = (
    REPORTS_DIR
    / "shap_feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=FEATURES
)

shap_df.insert(
    0,
    "date",
    sample["date"].values
)

shap_path = (
    REPORTS_DIR
    / "shap_values.csv"
)

shap_df.to_csv(
    shap_path,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("\nSHAP analysis completed successfully.")

print("\nTop 10 influential features:")

print(
    importance.head(10).to_string(
        index=False
    )
)

print("\nGenerated files:")

print(importance_path)
print(shap_path)