# src/predict.py
import os
import joblib
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from src.utils import ensure_dirs

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = "24.8607", "67.0011"

def predict_next3days():
    ensure_dirs()
    
    # Fetch 3-day AQI forecast from OpenWeather
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={API_KEY}"
    resp = requests.get(url, timeout=20)
    data = resp.json()
    
    rows = []
    for rec in data.get("list", []):
        dt = datetime.fromtimestamp(rec["dt"])
        comp = rec.get("components", {})
        aqi_raw = rec.get("main", {}).get("aqi", None)
        # Scale AQI: 1–5 → 0–500
        aqi_scaled = aqi_raw * 100 if aqi_raw is not None else None
        rows.append({
            "datetime": dt,
            "AQI": aqi_scaled,
            "PM2_5": comp.get("pm2_5"),
            "PM10": comp.get("pm10"),
            "NO2": comp.get("no2"),
            "SO2": comp.get("so2"),
            "CO": comp.get("co"),
            "O3": comp.get("o3")
        })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("⚠️ No forecast data available")
        return df
    
    # Feature engineering for time
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    # Load trained models
    MODEL_PATHS = {
        "Linear Regression": "models/aqi_linear_regression_model.pkl",
        "Random Forest": "models/aqi_random_forest_model.pkl",
        "XGBoost": "models/aqi_xgboost_model.pkl"
    }

    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        models[model_name] = joblib.load(model_path)

    # Prepare input features
    X = df[['PM2_5','PM10','NO2','SO2','CO','O3','hour','day','month']]

    # Predict AQI using all models
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X)

    df["Linear_Regression_Predicted_AQI"] = predictions["Linear Regression"]
    df["Random_Forest_Predicted_AQI"] = predictions["Random Forest"]
    df["Xgboost_Predicted_AQI"] = predictions["XGBoost"]

    # Average predicted AQI
    # Scale to 0–500
    df["Linear_Regression_Predicted_AQI"] *= 100
    df["Random_Forest_Predicted_AQI"] *= 100
    df["Xgboost_Predicted_AQI"] *= 100
    df["Average_Predicted_AQI"] = df[
    ["Linear_Regression_Predicted_AQI","Random_Forest_Predicted_AQI","Xgboost_Predicted_AQI"]
    ].mean(axis=1)


    # Display first 5 rows as a check
    print("✅ 3-Day AQI Prediction Complete")
    print(df[["datetime","Average_Predicted_AQI","Linear_Regression_Predicted_AQI",
              "Random_Forest_Predicted_AQI","Xgboost_Predicted_AQI"]].head())

    return df

if __name__ == "__main__":
    predict_next3days()

