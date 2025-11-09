# -------------------- FIX FOR STREAMLIT CLOUD --------------------
import sys
import os

# Add repo root to Python path so 'src' folder modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# -------------------- IMPORTS --------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime

# -------------------- HOPSWORKS IMPORT (OPTIONAL) --------------------
try:
    from src.hopsworks_utils import upload_to_hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False

# -------------------- STREAMLIT CLOUD SECRETS --------------------
if HOPSWORKS_AVAILABLE:
    try:
        HOPSWORKS_API_KEY = st.secrets["HOPSWORKS_API_KEY"]
        HOPSWORKS_PROJECT = st.secrets["HOPSWORKS_PROJECT"]
        HOPSWORKS_USERNAME = st.secrets["HOPSWORKS_USERNAME"]
        HOPSWORKS_PASSWORD = st.secrets["HOPSWORKS_PASSWORD"]
    except KeyError:
        st.warning("⚠️ Hopsworks secrets are missing in Streamlit Cloud!")

# -------------------- LOCAL IMPORTS --------------------
from src.utils import load_data
from src.predict import predict_next3days
from src.train_randomforest import train_randomforest
from src.train_xgboost import train_xgboost
from src.train_linear import train_linear

# -------------------- STREAMLIT PAGE CONFIG --------------------
st.set_page_config(page_title="🌍 Pearls AQI Predictor", layout="wide")

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #e3f2fd, #bbdefb); font-family: 'Segoe UI', sans-serif; }
    .banner { background-color:#0d47a1; padding: 18px; border-radius: 10px; text-align: center; color: white; font-size: 28px; font-weight: 700; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
    .subtext { font-size:16px; font-weight:500; color:#e3f2fd; margin-top:-10px; }
    .metric-box { text-align:center; padding:15px; border-radius:10px; background:#fff; box-shadow:0 3px 6px rgba(0,0,0,0.15); }
    h2, h3, h4 { color: #0d47a1 !important; }
    </style>
""", unsafe_allow_html=True)

# --- Banner ---
st.markdown("""
<div class="banner">
    🌍 Pearls AQI Predictor — Karachi
    <div class="subtext">Developed by <b>Mubashir Rehman</b></div>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
df = load_data()
if df.empty:
    st.warning("⚠️ No data found. Please run backfill and fetch scripts first.")
else:
    st.success(f"✅ Loaded {len(df)} AQI records.")
    st.dataframe(df.tail(10))

# --- Buttons Row ---
col1, col2, col3 = st.columns(3)

# 🚀 Train Models
with col1:
    if st.button("🚀 Train Linear Regression"):
        with st.spinner("Training Linear Regression... ⏳"):
            train_linear()
            st.success("✅ Linear Regression trained!")
    if st.button("🚀 Train Random Forest"):
        with st.spinner("Training Random Forest... ⏳"):
            train_randomforest()
            st.success("✅ Random Forest trained!")
    if st.button("🚀 Train XGBoost"):
        with st.spinner("Training XGBoost... ⏳"):
            train_xgboost()
            st.success("✅ XGBoost trained!")

# ☁️ Sync to Hopsworks
with col2:
    if HOPSWORKS_AVAILABLE and 'HOPSWORKS_API_KEY' in locals():
        if st.button("☁️ Sync to Hopsworks"):
            with st.spinner("Uploading data to Hopsworks Feature Store..."):
                try:
                    if df.empty:
                        st.warning("⚠️ No data to upload!")
                    else:
                        upload_to_hopsworks(
                            df,
                            api_key=HOPSWORKS_API_KEY,
                            project=HOPSWORKS_PROJECT,
                            username=HOPSWORKS_USERNAME,
                            password=HOPSWORKS_PASSWORD
                        )
                        st.success(f"✅ Synced {len(df)} records to Hopsworks successfully!")
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")
    else:
        st.info("⚠️ Hopsworks not available in this environment")

# 🔮 Predict 3 Days
with col3:
    if st.button("🔮 Predict Next 3 Days"):
        st.info("Scroll down to view predictions below 👇")

# --- Model Performance ---
st.subheader("📊 Model Performance Comparison")
model_files = {
    "Linear Regression": "models/aqi_linear_regression_model.pkl",
    "Random Forest": "models/aqi_random_forest_model.pkl",
    "XGBoost": "models/aqi_xgboost_model.pkl"
}

metrics = []
for model_name, path in model_files.items():
    if os.path.exists(path):
        model = joblib.load(path)
        X = df[['PM2_5','PM10','NO2','SO2','CO','O3','hour','day','month']].dropna()
        y = df.loc[X.index, 'AQI']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 500)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        metrics.append({"Model": model_name, "RMSE": rmse, "R²": r2})
    else:
        metrics.append({"Model": model_name, "RMSE": None, "R²": None})

metrics_df = pd.DataFrame(metrics)
st.dataframe(metrics_df)

# --- Charts for Model Performance ---
if not metrics_df.empty:
    st.subheader("📈 RMSE Comparison")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(metrics_df["Model"], metrics_df["RMSE"], color=["#1565c0","#ef6c00","#2e7d32"])
    ax.set_ylabel("RMSE")
    st.pyplot(fig)

    st.subheader("💪 R² Score Comparison")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.bar(metrics_df["Model"], metrics_df["R²"], color=["#1565c0","#ef6c00","#2e7d32"])
    ax2.set_ylabel("R²")
    st.pyplot(fig2)

# --- 3-Day Forecast Prediction ---
st.subheader("🔮 Next 3-Day Average AQI (0–500 Scale)")

def categorize_aqi(value):
    if value <= 50: return "Good 😊", "#4CAF50"
    elif value <= 100: return "Moderate 😐", "#FFEB3B"
    elif value <= 150: return "Unhealthy for Sensitive 😷", "#FF9800"
    elif value <= 200: return "Unhealthy 😫", "#F44336"
    elif value <= 300: return "Very Unhealthy 😵", "#9C27B0"
    else: return "Hazardous ☠️", "#6A1B9A"

try:
    df_pred = predict_next3days()
    if df_pred.empty:
        st.info("⚠️ No forecast data available.")
    else:
        df_pred = df_pred.loc[:, ~df_pred.columns.duplicated()]
        for col in df_pred.columns:
            if "Predicted_AQI" in col or col == "Average_Predicted_AQI":
                df_pred[col] = np.clip(df_pred[col], 0, 500)
        if "Average_Predicted_AQI" in df_pred.columns:
            avg_aqi = round(df_pred["Average_Predicted_AQI"].mean())
            category, color = categorize_aqi(avg_aqi)
            st.markdown(f"""
                <div style="background-color:{color}; padding:25px; border-radius:15px; text-align:center; color:white;">
                    <h2>Next 3-Day Average AQI</h2>
                    <h1 style="font-size:50px;">{avg_aqi}</h1>
                    <h3>{category}</h3>
                </div>
            """, unsafe_allow_html=True)
        st.subheader("📈 Active vs Average Predicted AQI")
        plot_cols = ["AQI"]
        if "Average_Predicted_AQI" in df_pred.columns:
            plot_cols.append("Average_Predicted_AQI")
        if len(plot_cols) > 0:
            chart_df = df_pred.set_index("datetime")[plot_cols]
            st.line_chart(chart_df)
        st.subheader("📊 Model-wise Predicted AQI")
        model_cols = [col for col in df_pred.columns if "_Predicted_AQI" in col]
        if len(model_cols) > 0:
            st.line_chart(df_pred.set_index("datetime")[model_cols])
        else:
            st.info("⚠️ No model predictions available to plot.")
except Exception as e:
    st.error(f"Prediction failed: {e}")

# --- Pollutant Trends ---
st.subheader("📉 Pollutant Trends Over Time")
if not df.empty:
    pollutants = ['PM2_5','PM10','NO2','SO2','CO','O3']
    st.line_chart(df.set_index('datetime')[pollutants])

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#0d47a1;'>© 2025 Pearls AQI Project | Developed by Mubashir Rehman</p>", unsafe_allow_html=True)
