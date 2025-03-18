from fastapi import FastAPI
import requests
import joblib
import numpy as np
from datetime import datetime, timedelta

app = FastAPI()

# Load ML Model
model = joblib.load("arima_aqi_model.pkl")

# IQAir API Key
API_KEY = "1d08f00c-6de1-4dd7-afb9-0d5a3498d97f"
IQAIR_URL = "https://api.airvisual.com/v2/nearest_city"

API_KEY_CN = "fc52df53c364f2dcdbe16fb65c8efef9c71fe8ea"
AQICN_URL = "http://api.waqi.info/feed/A416860/"

@app.get("/air-quality")
def get_air_quality():
    response = requests.get(f"{IQAIR_URL}?key={API_KEY}")
    return response.json()

@app.get("/predict")
def predict_aqi():
    try:
        # 1️⃣ Fetch latest air quality data from IQAir API
        response = requests.get(f"{IQAIR_URL}?key={API_KEY}")
        response.raise_for_status()
        air_quality_data = response.json()

        # 2️⃣ Extract PM2.5 value from API response
        pm25 = air_quality_data.get("data", {}).get("current", {}).get("pollution", {}).get("aqius")
        if pm25 is None:
            return {"error": "PM2.5 data not available"}

        # 3️⃣ Forecast AQI for the next 5 days using ARIMA
        forecast_steps = 5
        forecast_aqi = model.forecast(steps=forecast_steps)

        # 4️⃣ Create date labels for forecast
        start_date = datetime.today()
        forecast_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, forecast_steps + 1)]

        # 5️⃣ Return actual data + prediction
        return {
            "current_aqi": pm25,
            "forecast": {forecast_dates[i]: round(forecast_aqi[i], 2) for i in range(forecast_steps)}
        }

    except requests.exceptions.RequestException as e:
        return {"error": "Failed to fetch data from IQAir", "details": str(e)}
    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}