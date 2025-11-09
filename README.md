# Air Quality Index (AQI) Predictor

## Overview
The **Air Quality Index (AQI) Predictor** is a Python-based local application that forecasts AQI for the next three days using historical data. The app provides clear visualizations of predicted AQI levels, helping users make informed decisions about outdoor activities and environmental safety.

---

## Features
- Predicts AQI for the next 3 days.
- Displays predictions along with hazard categories (Good, Moderate, Unhealthy, etc.).
- Calculates model performance metrics like **RMSE** and **R² Score**.
- Interactive local GUI using **Streamlit**.

---

## Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/mubashir-rehman832/pearls-aqi-predictor
cd "C:\Users\User\Desktop\internship folder\pearls-aqi-predictor"
```
2.**Create Virtual Environment (venv)**
```bash
python -m venv venv
```
3.**Activate Virtual Environment**
```bash
window:venv\Scripts\activate
macOS/Linux:source venv/bin/activate
```
Using a virtual environment ensures that project dependencies are isolated from other Python projects,avoiding package conflicts.

4.**Install Required Libraries**
```bash
pip install -r requirements.txt
```
<img width="590" height="433" alt="image" src="https://github.com/user-attachments/assets/c577b01f-ae12-4cfe-803b-bff2d026f918" />

5.**Run the App Locally**
```bash
python -m streamlit run app/streamlit_app.py
```
**USAGE**

Open the Streamlit app in your browser.

Input required parameters (if any) or directly view the AQI forecast.

See the predicted AQI values along with hazard categories.

Model performance metrics (RMSE, R²) are displayed for evaluation.

---

## Folder Structure

<img width="615" height="285" alt="image" src="https://github.com/user-attachments/assets/eaacbbef-cd95-4d01-a132-821aa8224ff7" />

---

## Output

<img width="1278" height="245" alt="image" src="https://github.com/user-attachments/assets/643b0f56-5632-4ffb-87c1-c52b2aac1dbd" />

---

## Model Metrics

<img width="1234" height="113" alt="image" src="https://github.com/user-attachments/assets/4148dab4-f8d1-4ff8-9e99-eb9265ea0d11" />

---

## Author

Mubashir Rehman

GitHub: mubashir-rehman832

Contact: rehmanmubashir186@gmail.com




