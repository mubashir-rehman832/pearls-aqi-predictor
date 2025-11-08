# src/hopsworks_utils.py
import hopsworks
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def connect_hopsworks():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("Missing HOPSWORKS_API_KEY in .env")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    print("Connected to Hopsworks Feature Store")
    return fs

def upload_to_hopsworks(df: pd.DataFrame, feature_group_name="aqi_data", version=1):
    if df.empty:
        print("⚠️ No data to upload.")
        return
    fs = connect_hopsworks()
    try:
        feature_group = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            primary_key=["datetime"],
            description="Air Quality Index dataset for Karachi"
        )
        feature_group.insert(df, write_options={"wait_for_job": False})
        print(f"Uploaded {len(df)} rows to Hopsworks Feature Group: {feature_group_name}")
    except Exception as e:
        print("Upload to Hopsworks failed:", e)
