# src/train_linear.py
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import load_data, ensure_dirs

def train_linear():
    ensure_dirs()
    df = load_data()
    df = df.dropna(subset=['AQI','PM2_5','PM10','NO2','SO2','CO','O3'])
    X = df[['PM2_5','PM10','NO2','SO2','CO','O3','hour','day','month']]
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Linear Regression trained | RMSE={rmse:.3f} | R²={r2:.3f}")

    joblib.dump(model, "models/aqi_linear_regression_model.pkl")

if __name__ == "__main__":
    train_linear()
