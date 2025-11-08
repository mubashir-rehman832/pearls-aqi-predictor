# src/backfill_history.py
import argparse, time, requests, pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from src.utils import ensure_dirs, load_data, save_data
from src.hopsworks_utils import upload_to_hopsworks
import os

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = "24.8607", "67.0011"

def fetch_history(days=180):  # 6 months (≈180 days)
    ensure_dirs()
    now = int(time.time())
    start = now - days * 24 * 3600
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start}&end={now}&appid={API_KEY}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for rec in data.get("list", []):
        dt = datetime.fromtimestamp(rec["dt"])
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

    df = pd.DataFrame(rows)
    if df.empty:
        print("No historical data found.")
        return

    df_existing = load_data()
    df = pd.concat([df_existing, df], ignore_index=True).drop_duplicates("datetime").sort_values("datetime")
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["AQI_change_rate"] = df["AQI"].diff().fillna(0)

    save_data(df)
    upload_to_hopsworks(df)
    print("Historical data (6 months) saved and uploaded to Hopsworks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    fetch_history(days=args.days)
