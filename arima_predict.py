import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess your AQI_US data
file_path = "/mnt/data/processed_air_quality.csv"
df = pd.read_csv("processed_air_quality.csv")
df['datetime'] = pd.to_datetime(df['date_pm25'] + ' ' + df['time_pm25'])
df.set_index('datetime', inplace=True)

# Aggregate AQI to daily average
df_aqi = df[['AQI_US']].resample('D').mean()

# Train the ARIMA model
model_aqi = ARIMA(df_aqi, order=(5, 1, 0))
model_aqi_fit = model_aqi.fit()

# Save the trained model as a .pkl file
joblib.dump(model_aqi_fit, "arima_aqi_model.pkl")

print("ARIMA Model Saved!")