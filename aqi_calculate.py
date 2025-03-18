import pandas as pd
import numpy as np

# Load dataset
file_path = "C:/Users/farha/Downloads/dataset_clean_Bandung_float_AD.xlsx"  # Update this if needed
df = pd.read_excel(file_path)

# Drop PM1 if present
df_clean = df.drop(columns=["PM1"], errors="ignore").dropna()

# Define AQI breakpoints function
def calculate_aqi(concentration, breakpoints):
    """Calculate AQI given concentration and breakpoints"""
    for (bp_low, bp_high, ilow, ihigh) in breakpoints:
        if bp_low <= concentration <= bp_high:
            return ((ihigh - ilow) / (bp_high - bp_low)) * (concentration - bp_low) + ilow
    return 500  # If above max range

# AQI breakpoints for PM2.5 and PM10
breakpoints_pm25 = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

breakpoints_pm10 = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

# Calculate AQI for PM2.5 and PM10
df_clean["AQI_PM25"] = df_clean["PM25"].apply(lambda x: calculate_aqi(x, breakpoints_pm25))
df_clean["AQI_PM10"] = df_clean["PM10"].apply(lambda x: calculate_aqi(x, breakpoints_pm10))

# Final AQI is the max of PM2.5 AQI and PM10 AQI
df_clean["AQI_US"] = df_clean[["AQI_PM25", "AQI_PM10"]].max(axis=1)

# Save cleaned dataset
df_clean.to_csv("processed_air_quality.csv", index=False)

print("AQI calculation complete. Saved as processed_air_quality.csv")