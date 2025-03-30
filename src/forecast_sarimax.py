import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the hourly data
hourly_df = pd.read_csv("data/hour.csv")

# Combine dteday and hr to create a timestamp column
hourly_df["timestamp"] = pd.to_datetime(hourly_df["dteday"]) + pd.to_timedelta(
    hourly_df["hr"], unit="h"
)

# Set the timestamp column as the index and ensure it is a DatetimeIndex
hourly_df = hourly_df.set_index("timestamp")
hourly_df.index = pd.to_datetime(hourly_df.index)

# Ensure the index is a DatetimeIndex to access year, dayofyear, and hour attributes
if not isinstance(hourly_df.index, pd.DatetimeIndex):
    raise TypeError("Index is not a DatetimeIndex. Please check the data format.")

# For SARIMAX we only need the timestamp column and the target variable cnt.
# We drop the unnecessary columns.
hourly_df = hourly_df[["cnt"]]

# Hourly differencing since there is a strong daily pattern
hourly_df["cnt_diff"] = hourly_df["cnt"].diff(periods=24)

# Fill in the values which are NaN due to the differencing
hourly_df["cnt_diff"].fillna(method="backfill", inplace=True)
print(hourly_df.info())
print(hourly_df.head())

# Identify the Seasonal Component
result = seasonal_decompose(hourly_df["cnt"], model="multiplicative", period=12)
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()
residual = result.resid.dropna()

# Plot the decomposed components
plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(hourly_df["cnt"].iloc[:744], label="Original Series")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(trend.iloc[:744], label="Trend")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(seasonal.iloc[:744], label="Seasonal")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(residual.iloc[:744], label="Residuals")
plt.legend()

plt.tight_layout()
plt.show()

# Create an index of year, day, and hour as an exogenous variable for SARIMAX
hourly_df["year_day_hour_index"] = (
    hourly_df.index.year * 10000  # Multiply year by 10000 to give it more weight
    + hourly_df.index.dayofyear * 100  # Multiply day of year by 100
    + hourly_df.index.hour  # Add the hour
)

# SARIMAX_model = pm.auto_arima(
#     hourly_df["cnt"],
#     exogenous=hourly_df[["year_day_hour_index"]],
#     start_p=1,
#     start_q=1,
#     max_p=2,
#     max_q=2,  # Reduce max_p and max_q
#     d=1,
#     max_d=1,  # Reduce differencing
#     start_P=1,
#     start_Q=1,
#     max_P=1,
#     max_Q=1,  # Reduce seasonal parameters
#     D=1,
#     max_D=1,
#     seasonal=True,
#     m=24,  # Focus on daily seasonality
#     trace=True,
#     error_action="ignore",
#     suppress_warnings=True,
#     stepwise=False,  # Disable stepwise search
# )

# Use only data from 2011
# Filter the DataFrame to include only data from 2011
subset_df = hourly_df.loc[hourly_df.index.year == 2012]

# Update the training and testing split
train_size = int(len(subset_df) * 0.8)
train_df = subset_df.iloc[:train_size]
test_df = subset_df.iloc[train_size:]

# Define features and target variable
X_train = train_df[["year_day_hour_index"]]
y_train = train_df["cnt"]
X_test = test_df[["year_day_hour_index"]]
y_test = test_df["cnt"]

# Train the SARIMAX model
sarimax_model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 1, 1),  # Example SARIMA order (p, d, q)
    seasonal_order=(1, 1, 1, 24),  # Example seasonal order (P, D, Q, s)
)
sarimax_result = sarimax_model.fit(disp=False)

# Make predictions on the test set
y_pred = sarimax_result.predict(start=train_size, end=len(subset_df) - 1, exog=X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mad = np.mean(np.abs(y_test - y_pred))

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Deviation (MAD): {mad}")

# Forecast the first 72 hours of the test set
forecast_72h = sarimax_result.predict(start=train_size, end=train_size + 71, exog=X_test.iloc[:72])

# Plot the forecast vs actual values
plt.figure(figsize=(12, 6))
plt.plot(range(len(forecast_72h)), forecast_72h, label="Predicted", color="blue", marker="o")
plt.plot(
    range(len(forecast_72h)), y_test.iloc[:72].values, label="Actual", color="orange", marker="x"
)
plt.title("Forecast vs Actual Values (First 72 Hours)")
plt.xlabel("Time (hours)")
plt.ylabel("Bike Rentals")
plt.legend()
plt.grid(True)
plt.show()
