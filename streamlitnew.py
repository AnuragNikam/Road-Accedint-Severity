import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Accident AI Dashboard", layout="wide")

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
@st.cache_resource
def load_files():
    model = load_model("model.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    return model, scaler, encoders, columns

try:
    model, scaler, encoders, columns = load_files()
except:
    st.error("⚠️ Run training file first!")
    st.stop()

# -----------------------------
# HEADER
# -----------------------------
st.title("🚗 Smart Road Accident Severity System")
st.markdown("### Deep Learning + Interactive Dashboard")

# -----------------------------
# SIDEBAR MENU
# -----------------------------
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Data Insights", "🔮 Prediction"])

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "🏠 Home":
    col1, col2, col3 = st.columns(3)

    col1.metric("Model Type", "Deep Learning")
    col2.metric("Algorithm", "Neural Network")
    col3.metric("Status", "Active ✅")

    st.markdown("---")

    st.subheader("📌 Project Overview")
    st.write("""
    This system predicts road accident severity using a Deep Neural Network.
    
    Features:
    - Real-time prediction
    - Interactive UI
    - Data visualization
    - Scalable deployment
    """)

# -----------------------------
# DATA INSIGHTS PAGE
# -----------------------------
elif menu == "📊 Data Insights":

    df = pd.read_csv("Road Accident Data.csv")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Accident Severity Distribution")
    fig1 = plt.figure()
    df["Accident_Severity"].value_counts().plot(kind="bar")
    st.pyplot(fig1)

    if "Weather_Conditions" in df.columns:
        st.subheader("Weather Impact")
        fig2 = plt.figure()
        df["Weather_Conditions"].value_counts().head(10).plot(kind="bar")
        st.pyplot(fig2)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif menu == "🔮 Prediction":

    st.subheader("Enter Accident Details")

    col1, col2 = st.columns(2)

    user_input = []

    for i, col in enumerate(columns):
        if col in encoders:
            options = list(encoders[col].classes_)

            if i % 2 == 0:
                val = col1.selectbox(col, options)
            else:
                val = col2.selectbox(col, options)

            val = encoders[col].transform([val])[0]

        else:
            if i % 2 == 0:
                val = col1.slider(col, 0, 200, 10)
            else:
                val = col2.slider(col, 0, 200, 10)

        user_input.append(val)

    st.markdown("---")

    if st.button("🚀 Predict Now"):
        data = np.array([user_input])
        data = scaler.transform(data)

        prediction = model.predict(data)
        score = prediction[0][0]

        st.subheader("Prediction Result")

        colA, colB = st.columns(2)

        if score > 0.5:
            colA.error("🚨 High Severity")
        else:
            colA.success("✅ Low Severity")

        colB.metric("Confidence Score", f"{score:.2f}")

        # Progress bar visualization
        st.progress(float(score))

        # Simple gauge-style chart
        fig3 = plt.figure()
        plt.bar(["Risk"], [score])
        plt.ylim(0, 1)
        plt.title("Accident Risk Level")
        st.pyplot(fig3)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("PG Project | Deep Learning + Streamlit Dashboard")
