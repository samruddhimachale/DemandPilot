
from pathlib import Path
import json
import os
import urllib.request
import urllib.error

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
API_URL = "https://demandpilot-api.onrender.com"

st.set_page_config(
    page_title="DemandPilot | MLOps Dashboard",
    page_icon="📦",
    layout="wide"
)

# --------------------------------------------------
# PREMIUM DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F5F7FB;
    color: #172B4D;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101F3C 0%, #172E50 100%);
}

[data-testid="stSidebar"] * {
    color: #F4F7FC !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C5D2E5 !important;
}

/* Headings */
h1 {
    color: #142744 !important;
    font-weight: 750 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #1D3557 !important;
    font-weight: 650 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 3px 12px rgba(25, 45, 75, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #17365D !important;
    font-weight: 750;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border-radius: 12px;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: #2563EB;
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.2rem;
    font-weight: 650;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #1D4ED8;
    border: none;
    color: white !important;
}

/* Input fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 8px;
}

/* Sidebar radio navigation */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 10px;
}

/* Dividers */
hr {
    border-color: #E2E8F0;
}

/* Success and info messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: #E6F7EF !important;
    border-radius: 10px;
}

[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #126B45 !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 2. LOAD PROJECT REPORTS
# --------------------------------------------------

@st.cache_data
def load_json(filename):
    path = REPORTS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


@st.cache_data
def load_csv(filename):
    path = REPORTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


xgb_metrics = load_json("xgboost_metrics.json")
baseline_metrics = load_json("baseline_metrics.json")
model_selection = load_json("model_selection.json")
category_data = load_json("category_mappings.json")
mappings = category_data.get("mappings", {})

comparison = load_csv("model_comparison.csv")
monitoring = load_csv("monitoring_metrics.csv")
alerts = load_csv("monitoring_alerts.csv")
shap_data = load_csv("shap_feature_importance.csv")
predictions = load_csv("monitoring_predictions.csv")
error_analysis = load_csv("error_analysis.csv")


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def metric_value(data, key, default=0):
    try:
        return float(data.get(key, default))
    except (ValueError, TypeError):
        return default


def api_health():
    try:
        request = urllib.request.Request(
            f"{API_URL}/health",
            method="GET"
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def call_prediction_api(payload):
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        f"{API_URL}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8")
        return None, f"API error {error.code}: {details}"
    except Exception as error:
        return None, str(error)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📦 DemandPilot")
st.sidebar.caption("Retail Demand & Inventory Forecasting")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Prediction",
        "Model Performance",
        "Monitoring & Drift",
        "Explainability"
    ]
)

health = api_health()

st.sidebar.divider()
st.sidebar.subheader("System Status")

if health:
    st.sidebar.success("API Connected")
    st.sidebar.write(
        f"Model loaded: {'Yes' if health.get('model_loaded') else 'No'}"
    )
    st.sidebar.write(
        f"Mappings loaded: {'Yes' if health.get('mappings_loaded') else 'No'}"
    )
else:
    st.sidebar.error("API Offline")
    st.sidebar.caption("Start FastAPI to enable predictions.")


# --------------------------------------------------
# 5. HEADER
# --------------------------------------------------

st.title("📦 DemandPilot")
st.caption("An MLOps platform for retail demand forecasting and inventory insights.")

st.divider()


# --------------------------------------------------
# 6. OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    st.write(
        "DemandPilot uses machine learning to forecast retail product demand "
        "and support better inventory planning."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Selected Model",
        model_selection.get("selected_model", "XGBoost")
    )

    col2.metric(
        "MAE",
        f"{metric_value(xgb_metrics, 'mae'):.3f}"
    )

    col3.metric(
        "WAPE",
        f"{metric_value(xgb_metrics, 'wape_percent'):.2f}%"
    )

    col4.metric(
        "Validation Rows",
        f"{int(metric_value(xgb_metrics, 'validation_rows')):,}"
    )

    st.subheader("Model Performance Summary")

    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True)

    st.subheader("Model Selection")

    improvement = model_selection.get("improvement", {})

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE Improvement",
        f"{metric_value(improvement, 'mae_percent'):.2f}%"
    )

    c2.metric(
        "RMSE Improvement",
        f"{metric_value(improvement, 'rmse_percent'):.2f}%"
    )

    c3.metric(
        "WAPE Improvement",
        f"{metric_value(improvement, 'wape_percent'):.2f}%"
    )

    st.info(
        "XGBoost was selected using MAE as the primary model-selection metric. "
        "Its sMAPE is higher than the Seasonal Naive baseline, so all metrics "
        "are shown transparently."
    )

    st.subheader("Validation Period")

    st.write(
        f"**{xgb_metrics.get('validation_start', 'N/A')}** "
        f"to **{xgb_metrics.get('validation_end', 'N/A')}**"
    )



# --------------------------------------------------
# 7. LIVE PREDICTION - IMPROVED EXPERIENCE
# --------------------------------------------------

elif page == "Live Prediction":

    st.header("Demand Forecasting")
    st.caption(
        "Generate a demand estimate using your trained XGBoost model."
    )

    if not mappings:
        st.error("Category mappings could not be loaded.")
        st.stop()

    if not health or not health.get("model_loaded") or not health.get("mappings_loaded"):
        st.warning("Prediction API is not ready. Start FastAPI in another terminal.")
        st.stop()

    payload_path = REPORTS_DIR / "api_test_payload.json"

    with open(payload_path, "r", encoding="utf-8") as file:
        defaults = json.load(file)

    # Product and store options
    items = sorted(mappings["item_id"].keys())
    stores = sorted(mappings["store_id"].keys())

    # Use the existing tested payload for initial values
    default_item = defaults["item_id"]
    default_store = defaults["store_id"]

    st.info(
        "Choose a product, store and date. Calendar features are calculated "
        "automatically. Price and demand-history features are supplied below."
    )

    with st.form("improved_prediction_form"):

        # PRODUCT AND STORE
        st.subheader("1. Product and Store")

        col1, col2 = st.columns(2)

        with col1:
            item_id = st.selectbox(
                "Product ID",
                items,
                index=items.index(default_item)
            )

        with col2:
            store_id = st.selectbox(
                "Store",
                stores,
                index=stores.index(default_store)
            )

        # DATE
        st.subheader("2. Forecast Date")

        from datetime import date

        default_date = date(
            int(defaults["year"]),
            int(defaults["month"]),
            int(defaults["day_of_month"])
        )

        forecast_date = st.date_input(
            "Select date",
            value=default_date
        )

        st.caption(
            "Calendar fields such as weekday, month, quarter and year "
            "will be calculated from this date."
        )

        # PRICE
        st.subheader("3. Pricing Information")

        col1, col2 = st.columns(2)

        with col1:
            sell_price = st.number_input(
                "Selling price",
                min_value=0.0,
                value=float(defaults["sell_price"]),
                step=0.10
            )

            price_change = st.number_input(
                "Price change",
                value=float(defaults["price_change"]),
                step=0.10
            )

        with col2:
            price_rolling_mean_7 = st.number_input(
                "7-day average selling price",
                min_value=0.0,
                value=float(defaults["price_rolling_mean_7"]),
                step=0.10
            )

            price_missing = st.checkbox(
                "Price information is missing",
                value=bool(defaults["price_missing"])
            )

        # EVENT AND SNAP
        st.subheader("4. Event and SNAP Information")

        col1, col2 = st.columns(2)

        with col1:
            is_event = st.checkbox(
                "Special event day",
                value=bool(defaults["is_event"])
            )

        with col2:
            snap = st.checkbox(
                "SNAP benefit active",
                value=bool(defaults["snap"])
            )

        # DEMAND HISTORY
        st.subheader("5. Recent Demand History")

        st.caption(
            "Enter historical demand values for the selected product and store."
        )

        with st.expander("Configure demand history", expanded=True):

            col1, col2 = st.columns(2)

            demand_fields = [
                ("lag_1", "Demand 1 day ago"),
                ("lag_7", "Demand 7 days ago"),
                ("lag_14", "Demand 14 days ago"),
                ("lag_28", "Demand 28 days ago"),
                ("rolling_mean_7", "7-day rolling demand average"),
                ("rolling_mean_14", "14-day rolling demand average"),
                ("rolling_mean_28", "28-day rolling demand average"),
                ("rolling_std_7", "7-day rolling demand standard deviation")
            ]

            history = {}

            for index, (field, label) in enumerate(demand_fields):
                with (col1 if index % 2 == 0 else col2):
                    history[field] = st.number_input(
                        label,
                        min_value=0.0,
                        value=float(defaults[field]),
                        step=0.10
                    )

        submitted = st.form_submit_button(
            "Generate Demand Prediction",
            type="primary",
            use_container_width=True
        )

    # BUILD API PAYLOAD
    if submitted:

        # Derive categorical identifiers from the selected product/store
        dept_id = item_id.rsplit("_", 1)[0]
        cat_id = dept_id.split("_")[0]
        state_id = store_id.split("_")[0]

        # Validate derived categories against actual mappings
        valid_categories = (
            dept_id in mappings["dept_id"]
            and cat_id in mappings["cat_id"]
            and state_id in mappings["state_id"]
        )

        if not valid_categories:
            st.error("Could not derive valid product or store categories.")
            st.stop()

        # Calculate calendar features
        start_date = date(2011, 1, 29)
        day_of_week = forecast_date.weekday()

        payload = {
            "item_id": item_id,
            "dept_id": dept_id,
            "cat_id": cat_id,
            "store_id": store_id,
            "state_id": state_id,

            "day_of_week": day_of_week,
            "day_of_month": forecast_date.day,
            "week_of_year": forecast_date.isocalendar().week,
            "month": forecast_date.month,
            "quarter": (forecast_date.month - 1) // 3 + 1,
            "year": forecast_date.year,

            "is_weekend": int(day_of_week >= 5),
            "is_month_start": int(forecast_date.day == 1),
            "is_month_end": int(
                (forecast_date.replace(day=28) +
                 __import__("datetime").timedelta(days=4)).replace(day=1)
                .__sub__(__import__("datetime").timedelta(days=1)).day
                == forecast_date.day
            ),

            "is_event": int(is_event),
            "snap": int(snap),

            "sell_price": float(sell_price),
            "price_missing": int(price_missing),
            "price_change": float(price_change),
            "price_rolling_mean_7": float(price_rolling_mean_7),

            "lag_1": float(history["lag_1"]),
            "lag_7": float(history["lag_7"]),
            "lag_14": float(history["lag_14"]),
            "lag_28": float(history["lag_28"]),
            "rolling_mean_7": float(history["rolling_mean_7"]),
            "rolling_mean_14": float(history["rolling_mean_14"]),
            "rolling_mean_28": float(history["rolling_mean_28"]),
            "rolling_std_7": float(history["rolling_std_7"]),

            "days_since_start": (forecast_date - start_date).days
        }

        result, error = call_prediction_api(payload)

        if error:
            st.error(f"Prediction failed: {error}")

        else:
            st.success("Prediction generated successfully!")

            st.divider()
            st.subheader("Prediction Result")

            predicted = float(result["predicted_demand"])

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Demand",
                f"{predicted:.3f} units"
            )

            col2.metric("Product", item_id)

            col3.metric("Store", store_id)

            st.write(
                f"**Forecast date:** {forecast_date.strftime('%d %B %Y')}"
            )

            st.info(
                "This is a model estimate based on the supplied features. "
                "For a reliable future forecast, provide accurate future "
                "pricing, event, SNAP and historical demand inputs."
            )

            with st.expander("View complete API input"):
                st.json(payload)

# --------------------------------------------------
# 8. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance Comparison")

    if not comparison.empty:

        st.subheader("Evaluation Metrics")

        st.dataframe(comparison, use_container_width=True)

        metric_columns = [
            column for column in ["MAE", "RMSE", "WAPE (%)", "sMAPE (%)"]
            if column in comparison.columns
        ]

        if metric_columns and "Model" in comparison.columns:
            selected_metric = st.selectbox(
                "Choose metric",
                metric_columns
            )

            fig = px.bar(
                comparison,
                x="Model",
                y=selected_metric,
                color="Model",
                title=f"Model Comparison: {selected_metric}",
                text_auto=".3f"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Model comparison report is unavailable.")

    st.subheader("XGBoost Evaluation")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", f"{metric_value(xgb_metrics, 'mae'):.4f}")
    c2.metric("RMSE", f"{metric_value(xgb_metrics, 'rmse'):.4f}")
    c3.metric("sMAPE", f"{metric_value(xgb_metrics, 'smape_percent'):.2f}%")
    c4.metric("WAPE", f"{metric_value(xgb_metrics, 'wape_percent'):.2f}%")


# --------------------------------------------------
# 9. MONITORING AND DRIFT
# --------------------------------------------------

elif page == "Monitoring & Drift":

    st.header("Model Monitoring & Data Drift")

    if not monitoring.empty:

        row = monitoring.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric("Monitoring MAE", f"{float(row['mae']):.4f}")
        c2.metric("Monitoring WAPE", f"{float(row['wape_percent']):.2f}%")
        c3.metric(
            "Zero Demand Rate",
            f"{float(row['zero_demand_rate_percent']):.2f}%"
        )

        st.subheader("Monitoring Metrics")
        st.dataframe(monitoring, use_container_width=True)

    if not alerts.empty:

        st.subheader("Monitoring Alerts")

        st.dataframe(alerts, use_container_width=True)

        if "status" in alerts.columns:
            statuses = alerts["status"].astype(str).str.upper()

            if (statuses == "ALERT").any() or (statuses == "FAIL").any():
                st.error("One or more monitoring checks require attention.")
            else:
                st.success("All recorded monitoring checks are OK.")

    st.subheader("Data Drift Report")

    drift_path = REPORTS_DIR / "data_drift_report.html"

    if drift_path.exists():
        with open(drift_path, "r", encoding="utf-8") as file:
            drift_html = file.read()

        components.html(
            drift_html,
            height=750,
            scrolling=True
        )
    else:
        st.warning("Data drift HTML report not found.")

    if not predictions.empty:
        st.subheader("Monitoring Prediction Samples")
        st.dataframe(
            predictions.head(20),
            use_container_width=True
        )

    if not error_analysis.empty:
        st.subheader("Error Analysis")
        st.dataframe(
            error_analysis.head(20),
            use_container_width=True
        )


# --------------------------------------------------
# 10. EXPLAINABILITY
# --------------------------------------------------

elif page == "Explainability":

    st.header("Model Explainability")
    st.write(
        "SHAP feature importance helps explain which features "
        "influence the XGBoost model's predictions."
    )

    if not shap_data.empty:

        shap_data = shap_data.sort_values(
            "mean_abs_shap",
            ascending=False
        )

        fig = px.bar(
            shap_data.head(15).sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top Features by Mean Absolute SHAP Value",
            labels={
                "mean_abs_shap": "Mean Absolute SHAP Value",
                "feature": "Feature"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("SHAP Feature Importance Data")
        st.dataframe(shap_data, use_container_width=True)

    else:
        st.warning("SHAP feature importance report not found.")


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DemandPilot | Retail Demand and Inventory Forecasting | "
    "MLOps Project"
)

