# src/utils.py
import os
import pandas as pd

DATA_PATH = os.path.join("data", "karachi_aqi.xlsx")
MODEL_DIR = "models"

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_excel(path, parse_dates=["datetime"])

def save_data(df, path=DATA_PATH):
    df.to_excel(path, index=False)
