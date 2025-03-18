import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the dataset
file_path = "F:/PROGRAMMING CUY/API_ANDROID/processed_air_quality.csv"  # Update this if needed
df = pd.read_csv("processed_air_quality.csv")



# Define features (X) and target variable (Y)
X = df[["PM25", "PM10"]]  # Features: PM2.5, PM10
y = df["AQI_US"]  # Target: US AQI calculated from PM2.5

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate model performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf:.2f}")

joblib.dump(rf_model, "model.pkl")
