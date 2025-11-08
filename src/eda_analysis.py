# src/eda_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_data, save_data

def perform_eda():
    print("🔍 Starting Exploratory Data Analysis (EDA)...")

    # Load dataset
    df = load_data()
    if df.empty:
        print("⚠️ No data found. Run fetch/backfill first.")
        return

    print(f"✅ Data Loaded — {df.shape[0]} rows, {df.shape[1]} columns")

    # Convert datetime column if needed
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])

    # Handle duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"🧹 Removed {before - after} duplicate rows")

    # Check and handle missing values
    print("\n📉 Missing Values Before Cleaning:")
    print(df.isnull().sum())

    # Fill missing numeric values with median
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"✅ Filled missing values in '{col}' with median ({median_val:.2f})")

    # Feature engineering for time
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    # AQI change rate
    if "AQI" in df.columns:
        df["AQI_change_rate"] = df["AQI"].diff().fillna(0)

    print("\n✅ Data Cleaning Completed Successfully!")
    print(f"📊 Cleaned Dataset Shape: {df.shape}")

    # Save cleaned dataset back to Excel
    save_data(df)
    print("💾 Cleaned data saved successfully to 'data/karachi_aqi.xlsx'")

    # ===== VISUALIZATIONS =====
    print("\n📈 Generating visualizations...")

    # 1. Correlation Heatmap
    plt.figure(figsize=(9, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of AQI Features")
    plt.tight_layout()
    plt.show()

    # 2. AQI Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(df["AQI"], bins=25, kde=True)
    plt.title("AQI Distribution")
    plt.xlabel("AQI Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # 3. PM2.5 vs AQI Scatter
    if "PM2_5" in df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x="PM2_5", y="AQI", data=df, alpha=0.7)
        plt.title("PM2.5 vs AQI")
        plt.xlabel("PM2.5")
        plt.ylabel("AQI")
        plt.tight_layout()
        plt.show()

    # 4. AQI over Time
    plt.figure(figsize=(10, 5))
    plt.plot(df["datetime"], df["AQI"], label="AQI", color="blue")
    plt.title("AQI Trend Over Time (Karachi)")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("✅ EDA Completed Successfully!")

if __name__ == "__main__":
    perform_eda()
