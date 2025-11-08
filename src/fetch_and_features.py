# src/fetch_and_features.py
import os, requests, pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from src.utils import ensure_dirs, load_data, save_data
from src.hopsworks_utils import upload_to_hopsworks

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = "24.8607", "67.0011"
URL_FORECAST = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={API_KEY}"

def fetch_next3days_forecast():
    ensure_dirs()
    resp = requests.get(URL_FORECAST, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    now = datetime.now()
    for rec in data.get("list", []):
        dt = datetime.fromtimestamp(rec["dt"])
        if (dt - now).total_seconds() >= 72 * 3600:
            continue
        comp = rec.get("components", {})
        rows.append({
            "datetime": dt,
            "AQI": rec.get("main", {}).get("aqi"),
            "PM2_5": comp.get("pm2_5"),
            "PM10": comp.get("pm10"),
            "NO2": comp.get("no2"),
            "SO2": comp.get("so2"),
            "CO": comp.get("co"),
            "O3": comp.get("o3")
        })

    df_new = pd.DataFrame(rows)
    df_existing = load_data()
    df = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates("datetime").sort_values("datetime")
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["AQI_change_rate"] = df["AQI"].diff().fillna(0)
    save_data(df)
    upload_to_hopsworks(df)
    print("Saved forecast + uploaded to Hopsworks.")

if __name__ == "__main__":
    fetch_next3days_forecast()
