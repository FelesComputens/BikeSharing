import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the hourly data
hourly_df = pd.read_csv("data/hour.csv")

# Convert dteday to datetime
hourly_df["dteday"] = pd.to_datetime(hourly_df["dteday"])

# Use only the target variable 'cnt' for time series modeling
y = hourly_df["cnt"]

# Time-based split
train_size = int(len(y) * 0.8)  # Use 80% of the data for training
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train the Holt-Winters Exponential Smoothing model
holt_winters_model = ExponentialSmoothing(
    y_train,
    seasonal="add",  # Additive seasonality
    seasonal_periods=24,  # Assuming daily seasonality (24 hours)
)
holt_winters_result = holt_winters_model.fit()

# Make predictions
y_pred = holt_winters_result.forecast(steps=len(y_test))

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Deviation (MAD): {np.mean(np.abs(y_test - y_pred))}")

# Predict future values (example: next 72 hours)
future_predictions = holt_winters_result.forecast(steps=72)
# print(f"Future Predictions: {future_predictions}")

# Plot the future predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(
    range(len(future_predictions)),
    y_test[:72],
    label="Actual",
    color="orange",
    marker="x",
)
plt.plot(
    range(len(future_predictions)),
    future_predictions,
    label="Predicted",
    color="blue",
    marker="o",
)
plt.title("Predictions vs Actual Values")
plt.xlabel("Time (hours)")
plt.ylabel("Bike Rentals")
plt.legend()
plt.grid(True)
plt.show()
