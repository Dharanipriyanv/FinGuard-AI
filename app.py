import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import random
import time as tm

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="💳",
    layout="wide"
)

# -------------------------
# LOAD MODEL + DATA
# -------------------------

model = joblib.load("fraud_model.pkl")
data = pd.read_csv("creditcard.csv")

explainer = shap.TreeExplainer(model)

# -------------------------
# DARK FINTECH THEME
# -------------------------

st.markdown("""
<style>

body {
    background-color:#0e1117;
}

.stButton>button {
    background-color:#00c8ff;
    color:black;
    font-weight:bold;
    border-radius:10px;
}

.stMetric {
    background-color:#1c1f26;
    padding:10px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------

st.title("💳 FinGuard AI")
st.caption("Explainable AI Financial Fraud Detection Platform")

st.divider()

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title("Control Panel")

amount = st.sidebar.slider(
    "Transaction Amount ($)",
    0.0, 50000.0, 500.0
)

time_val = st.sidebar.slider(
    "Transaction Time",
    0.0, 200000.0, 100.0
)

analyze = st.sidebar.button("🔍 Analyze Transaction")

st.sidebar.divider()

attack_mode = st.sidebar.toggle("🚨 Fraud Attack Simulation")

st.sidebar.write("Model: Random Forest")
st.sidebar.write("Dataset: Credit Card Fraud")

# -------------------------
# SESSION STATE
# -------------------------

if "input_data" not in st.session_state:
    st.session_state.input_data = None

# -------------------------
# TABS
# -------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔎 Transaction Analysis",
    "📈 Data Insights",
    "🌍 Global Fraud Map"
])

# -------------------------
# DASHBOARD
# -------------------------

with tab1:

    total = len(data)
    fraud = data["Class"].sum()

    if attack_mode:
        fraud = fraud * 20

    fraud_rate = (fraud / total) * 100

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Transactions", total)
    c2.metric("Fraud Cases", fraud)
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    if attack_mode:
        st.error("🚨 FRAUD ATTACK SIMULATION ACTIVE")

    st.subheader("📊 Fraud Trend Over Time")

    # Create numeric time bins (avoids JSON serialization error)
    data["TimeBin"] = pd.cut(data["Time"], bins=50, labels=False)

    trend = data.groupby("TimeBin")["Class"].sum().reset_index()

    fig_trend = px.line(
        trend,
        x="TimeBin",
        y="Class",
        title="Fraud Activity Trend"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

   
# -------------------------
# TRANSACTION ANALYSIS
# -------------------------

with tab2:

    st.subheader("Transaction Risk Detector")

    col1, col2 = st.columns(2)

    col1.write(f"Transaction Amount: **${amount}**")
    col2.write(f"Transaction Time: **{time_val}**")

    if analyze or True:

        with st.spinner("AI analyzing transaction..."):

            input_data = np.zeros((1, 30))
            input_data[0][0] = time_val
            input_data[0][-1] = amount

            st.session_state.input_data = input_data

            pred = model.predict(input_data)
            prob = model.predict_proba(input_data)

            fraud_prob = prob[0][1] * 100

        st.success("Analysis Complete")

        # FRAUD RISK GAUGE
        colA, colB, colC = st.columns([1,2,1])

        with colB:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob,
                title={'text': "Fraud Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "red"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        # ALERT MESSAGE
        if fraud_prob > 75:

            st.markdown(
                """
                <div style="padding:15px;
                            background-color:#ff4b4b;
                            color:white;
                            border-radius:10px;
                            text-align:center;
                            font-size:20px;">
                🚨 FRAUD ALERT — IMMEDIATE ACTION REQUIRED
                </div>
                """,
                unsafe_allow_html=True
            )

        elif fraud_prob > 40:
            st.warning("⚠ Suspicious Transaction")

        else:
            st.success("✅ Legitimate Transaction")

        st.divider()

        # SHAP EXPLANATION
        st.subheader("🧠 AI Decision Explanation")

        try:

            shap_values = explainer(st.session_state.input_data)

            fig_shap, ax = plt.subplots()

            shap.plots.waterfall(shap_values[0, :, 1], show=False)

            st.pyplot(fig_shap)

        except Exception:

            st.warning("SHAP explanation could not be generated.")
# -------------------------
# DATA INSIGHTS
# -------------------------

with tab3:

    st.subheader("Fraud Distribution")

    fraud_counts = data["Class"].value_counts().reset_index()
    fraud_counts.columns = ["Type", "Count"]

    fraud_counts["Type"] = fraud_counts["Type"].replace({
        0: "Legitimate",
        1: "Fraud"
    })

    fig1 = px.pie(
        fraud_counts,
        names="Type",
        values="Count",
        title="Fraud vs Legitimate Transactions"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Transaction Amount Distribution")

    fig2 = px.histogram(
        data,
        x="Amount",
        nbins=50
    )

    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------

    st.subheader("📊 Model Feature Importance")

    importances = model.feature_importances_

    features = [f"V{i}" for i in range(len(importances))]

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig_imp = px.bar(
        imp_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Influencing Fraud Detection"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------
# GLOBAL FRAUD MAP
# -------------------------

with tab4:

    st.subheader("Global Fraud Monitoring")

    fraud_locations = pd.DataFrame({
        "city": [
            "New York","London","Tokyo",
            "Singapore","Dubai","Mumbai"
        ],
        "lat":[
            40.7128,51.5074,35.6762,
            1.3521,25.2048,19.0760
        ],
        "lon":[
            -74.0060,-0.1278,139.6503,
            103.8198,55.2708,72.8777
        ],
        "fraud_cases":[
            120,90,70,65,50,80
        ]
    })

    fig_map = px.scatter_geo(
        fraud_locations,
        lat="lat",
        lon="lon",
        size="fraud_cases",
        hover_name="city",
        title="Simulated Global Fraud Activity"
    )

    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------
# LIVE FRAUD FEED
# -------------------------

st.divider()
st.subheader("🚨 Live Fraud Monitoring Feed")

cities = ["New York","London","Singapore","Dubai","Tokyo","Mumbai"]

feed = pd.DataFrame({
    "City":[random.choice(cities) for _ in range(5)],
    "Amount":[round(random.uniform(100,5000),2) for _ in range(5)],
    "Status":[random.choice(["Blocked","Under Review","Flagged"]) for _ in range(5)]
})

st.dataframe(feed, use_container_width=True)

# -------------------------
# AI ASSISTANT
# -------------------------

st.divider()

st.subheader("🤖 FinGuard AI Assistant")

user_question = st.text_input("Ask about fraud detection")

if user_question:

    q = user_question.lower()

    if "fraud" in q:
        st.write("Fraud transactions are detected using machine learning models trained on behavioral patterns.")

    elif "model" in q:
        st.write("FinGuard AI uses a Random Forest model trained on anonymized credit card transactions.")

    elif "shap" in q:
        st.write("SHAP explains how each feature influences the AI's decision.")

    elif "dataset" in q:
        st.write("The dataset contains real-world anonymized credit card transactions.")

    else:
        st.write("FinGuard AI analyzes transaction patterns and flags suspicious activities.")