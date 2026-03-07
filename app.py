import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load("fraud_model.pkl")

# Load dataset
data = pd.read_csv("creditcard.csv")

# SHAP explainer
explainer = shap.TreeExplainer(model)

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="💳",
    layout="wide"
)

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

time = st.sidebar.slider(
    "Transaction Time",
    0.0, 200000.0, 100.0
)

analyze = st.sidebar.button("🔍 Analyze Transaction")

st.sidebar.divider()

attack_mode = st.sidebar.toggle("🚨 Fraud Attack Simulation")

st.sidebar.write("Model: Random Forest")
st.sidebar.write("Dataset: Credit Card Fraud")

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

# -------------------------
# TRANSACTION ANALYSIS
# -------------------------

with tab2:

    st.subheader("Transaction Risk Detector")

    col1, col2 = st.columns(2)

    col1.write(f"Transaction Amount: **${amount}**")
    col2.write(f"Transaction Time: **{time}**")

    if analyze:

        with st.spinner("AI analyzing transaction..."):

            input_data = np.zeros((1,30))
            input_data[0][0] = time
            input_data[0][-1] = amount

            pred = model.predict(input_data)
            prob = model.predict_proba(input_data)

            fraud_prob = prob[0][1] * 100

        st.success("Analysis Complete")

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob,
            title={'text': "Fraud Risk %"},
            gauge={
                'axis': {'range':[0,100]},
                'steps':[
                    {'range':[0,40],'color':"green"},
                    {'range':[40,75],'color':"yellow"},
                    {'range':[75,100],'color':"red"}
                ]
            }
        ))

        st.plotly_chart(fig,use_container_width=True)

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

        # -------------------------
        # EXPLAINABLE AI
        # -------------------------

        st.subheader("🧠 AI Decision Explanation")

        shap_values = explainer.shap_values(input_data)

        fig2, ax = plt.subplots()

        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1],
            input_data,
            matplotlib=True,
            show=False
        )

        st.pyplot(fig2)

# -------------------------
# DATA INSIGHTS
# -------------------------

with tab3:

    st.subheader("Fraud Distribution")

    fraud_counts = data["Class"].value_counts().reset_index()
    fraud_counts.columns = ["Type","Count"]

    fraud_counts["Type"] = fraud_counts["Type"].replace({
        0:"Legitimate",
        1:"Fraud"
    })

    fig1 = px.pie(
        fraud_counts,
        names="Type",
        values="Count",
        title="Fraud vs Legitimate Transactions"
    )

    st.plotly_chart(fig1,use_container_width=True)

    st.subheader("Transaction Amount Distribution")

    fig2 = px.histogram(
        data,
        x="Amount",
        nbins=50
    )

    st.plotly_chart(fig2,use_container_width=True)

# -------------------------
# GLOBAL FRAUD MAP
# -------------------------

with tab4:

    st.subheader("Global Fraud Monitoring")

    fraud_locations = pd.DataFrame({
        "city":[
            "New York","London","Tokyo","Singapore","Dubai","Mumbai"
        ],
        "lat":[
            40.7128,51.5074,35.6762,1.3521,25.2048,19.0760
        ],
        "lon":[
            -74.0060,-0.1278,139.6503,103.8198,55.2708,72.8777
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

    st.plotly_chart(fig_map,use_container_width=True)

# -------------------------
# AI ASSISTANT
# -------------------------

st.divider()

st.subheader("🤖 FinGuard AI Assistant")

user_question = st.text_input("Ask about fraud detection")

if user_question:

    if "fraud" in user_question.lower():
        st.write("Fraud transactions are detected using machine learning models trained on transaction patterns.")

    elif "model" in user_question.lower():
        st.write("This system uses a Random Forest machine learning model.")

    elif "dataset" in user_question.lower():
        st.write("The model was trained using a credit card fraud detection dataset.")

    else:
        st.write("FinGuard AI monitors transactions and detects suspicious patterns.")