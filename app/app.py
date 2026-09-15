import sys
from pathlib import Path
import requests

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(ROOT)
)

from services.campaign_service import load_customers
from services.api_client import (
    check_health,
    predict_customer,
    predict_explain_customer,
    get_model_info,
    optimize_campaign_api,
    get_drift_status,
    get_prediction_monitoring,
    get_production_performance,
    API_URL
)

st.set_page_config(
    page_title="Campaign Intelligence",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "Campaign Intelligence Platform"
)

st.caption(
    "AI-powered customer targeting and campaign optimization"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def get_data():

    return load_customers()

customers = get_data()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header(
    "API Status"
)

try:

    health = check_health()

    if health["model_loaded"]:
        st.sidebar.success(
            "ML API: Online"
        )
    else:
        st.sidebar.warning(
            "ML API: Model unavailable"
        )

except Exception:

    st.sidebar.error(
        "ML API: Offline"
    )

try:
    model_info = get_model_info()

    st.sidebar.caption(
        f"Model: {model_info['model_type']}"
    )

    st.sidebar.caption(
        f"Version: {model_info['model_version']}"
    )
except Exception:
    pass

st.sidebar.divider()

st.sidebar.header(
    "Campaign Settings"
)

budget = st.sidebar.number_input(
    "Campaign Budget",
    min_value=1000.0,
    value=100000.0,
    step=5000.0
)

contact_cost = st.sidebar.number_input(
    "Cost per Contact",
    min_value=0.0,
    value=20.0,
    step=5.0
)

conversion_value = st.sidebar.number_input(
    "Value per Conversion",
    min_value=1.0,
    value=1000.0,
    step=100.0
)

strategy = st.sidebar.selectbox(
    "Targeting Strategy",
    [
        "Highest Conversion Probability",
        "Highest Expected Profit",
        "Fatigue-Aware Targeting"
    ]
)

max_contacts = st.sidebar.number_input(
    "Maximum Customers",
    min_value=1,
    value=5000,
    step=500
)


# --------------------------------------------------
# OPTIMIZATION
# --------------------------------------------------

st.header(
    "Campaign Optimizer"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Budget", f"₹{budget:,.0f}")

with col2:
    st.metric("Contact Cost", f"₹{contact_cost:,.0f}")

with col3:
    st.metric("Maximum Contacts", f"{max_contacts:,}")

if st.button("🚀 Optimize Campaign", use_container_width=True):
    try:
        result = optimize_campaign_api(
            budget,
            contact_cost,
            conversion_value,
            strategy,
            max_contacts
        )
        st.session_state["campaign_result"] = result
    except Exception as e:
        st.error(f"Failed to optimize campaign: {e}")

result = st.session_state.get("campaign_result")

if result:
    st.divider()
    st.header("Campaign Recommendation")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers Targeted", result.get("customers_targeted", 0))
    with col2:
        st.metric("Expected Conversions", f"{result.get('expected_conversions', 0):.1f}")
    with col3:
        st.metric("Expected Revenue", f"₹{result.get('expected_revenue', 0):,.0f}")
    with col4:
        st.metric("Expected Profit", f"₹{result.get('expected_profit', 0):,.0f}")

    roi = result.get("roi", 0)
    if "expected_profit" in result and "campaign_cost" in result and result["campaign_cost"] > 0:
        roi = result["expected_profit"] / result["campaign_cost"]
        
    st.metric("Expected ROI", f"{roi:.1%}")

st.divider()

st.header("Strategy Comparison")

experiment_path = (
    ROOT / "data" / "experiments" / "campaign_strategy_results.csv"
)

try:
    experiment_df = pd.read_csv(experiment_path)
    st.dataframe(
        experiment_df,
        use_container_width=True,
        hide_index=True
    )

    chart_data = experiment_df.set_index("strategy")[["profit", "revenue"]]
    st.bar_chart(chart_data)

    best_strategy = (
        experiment_df
        .sort_values("profit", ascending=False)
        .iloc[0]
    )

    st.success(
        f"Best simulated strategy: "
        f"{best_strategy['strategy']} "
        f"with profit of "
        f"₹{best_strategy['profit']:,.0f}"
    )

except FileNotFoundError:
    st.info("Run the campaign experiment to display strategy comparison.")

st.divider()

# --------------------------------------------------
# MODEL HEALTH
# --------------------------------------------------

st.header(
    "Model Health"
)

try:
    performance = get_production_performance()
    
    if performance:
        if performance.get("status") == "insufficient_data":
            st.warning(
                "Production performance is "
                "not available yet because "
                "there are not enough matched "
                "prediction/outcome records."
            )
            st.write(performance)
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "PR-AUC",
                    f"{performance['pr_auc']:.3f}"
                )
            with col2:
                st.metric(
                    "ROC-AUC",
                    f"{performance['roc_auc']:.3f}"
                )
            with col3:
                st.metric(
                    "Brier Score",
                    f"{performance['brier_score']:.3f}"
                )
            with col4:
                st.metric(
                    "Conversion Rate",
                    f"{performance['actual_conversion_rate']:.1%}"
                )

except Exception as e:
    st.error(f"Failed to load production performance: {e}")

try:
    alerts_response = requests.get(f"{API_URL}/monitoring/alerts").json()
    alerts = alerts_response.get("alerts", [])
    
    if alerts:
        st.error("⚠️ Model monitoring alerts detected")
        for alert in alerts:
            st.warning(
                f"{alert['severity']}: {alert['message']}"
            )
    else:
        if performance and performance.get("status") != "insufficient_data":
            st.success("✓ No model degradation alerts")

except Exception as e:
    st.error(f"Failed to load alerts: {e}")

st.divider()

st.subheader("Data Drift")
try:
    drift = get_drift_status()
    
    if (
        drift["numerical_features_with_drift"] == 0
        and
        drift["categorical_features_with_drift"] == 0
    ):
        st.success("No significant feature drift detected.")
    else:
        st.warning(
            "Feature drift detected. "
            "Model review recommended."
        )

    m1, m2 = st.columns(2)
    m1.metric("Numerical Drift", drift["numerical_features_with_drift"])
    m2.metric("Categorical Drift", drift["categorical_features_with_drift"])

except Exception as e:
    st.error(f"Failed to load drift status: {e}")


# --------------------------------------------------
# CUSTOMER EXPLORER
# --------------------------------------------------

st.divider()

st.subheader("Model Intelligence")
st.write("Top Model Drivers")
try:
    importance = pd.read_csv(ROOT / "data" / "experiments" / "shap_feature_importance.csv")
    st.bar_chart(importance.set_index("feature")["mean_abs_shap"])
    st.caption(
        "SHAP values explain how model features "
        "contributed to the prediction. They do not "
        "represent causal effects."
    )
except Exception as e:
    st.info("Global feature importance not available. Generate using the notebook first.")

st.divider()

st.subheader(
    "Customer Explorer"
)

customer_index = st.selectbox(
    "Select Customer",
    customers.index
)

customer = customers.loc[customer_index]

customer_payload = (
    customer
    .drop(
        labels=[
            "y",
            "conversion_probability"
        ],
        errors="ignore"
    )
    .replace({np.nan: None})
    .to_dict()
)

customer_payload["customer_id"] = str(customer_index)

try:
    prediction = predict_explain_customer(
        customer_payload
    )

    probability = prediction[
        "conversion_probability"
    ]
    expected_value = prediction[
        "expected_value"
    ]
    model_version = prediction[
        "model_version"
    ]
    
    st.markdown("### Customer Intelligence")
    m1, m2 = st.columns(2)
    m1.metric("Conversion Probability", f"{probability:.1%}")
    m2.metric("Expected Value", f"₹{expected_value:,.0f}")
    
    st.markdown("#### Why this customer?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Positive factors**")
        for factor in prediction["explanation"]["positive"]:
            st.markdown(f"↑ {factor}")
            
    with c2:
        st.markdown("**Negative factors**")
        for factor in prediction["explanation"]["negative"]:
            st.markdown(f"↓ {factor}")

    st.caption(f"Model: {model_version}")

except Exception as e:
    st.error(f"Failed to get prediction from API: {e}")


st.write(
    "Customer information"
)

customer_info = (
    customer
    .drop(
        labels=[
            "y",
            "conversion_probability"
        ],
        errors="ignore"
    )
    .to_frame("Value")
    .astype(str)
)

st.dataframe(
    customer_info,
    use_container_width=True
)

st.divider()

st.caption(
    "Portfolio demonstration using the UCI Bank Marketing dataset. "
    "Not intended for production banking decisions."
)
