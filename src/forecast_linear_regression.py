import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression  # Import LinearRegression

# Load the hourly data
hourly_df = pd.read_csv('data/hour.csv')

# Convert dteday to datetime
hourly_df['dteday'] = pd.to_datetime(hourly_df['dteday'])

# The features 
# Feature engineering: Create time-based features
# The features hr (hour), mnth (month), yr (year), and weekday (day of the week) are already present in the dataset.
# Only day of the month needs to be created
hourly_df['day'] = hourly_df['dteday'].dt.day

# Create lag features from weathersit, temp, hum, windspeed, and the target variable cnt
lag_features = ['weathersit', 'temp', 'hum', 'windspeed', 'cnt']
lag_hours = 72  # Lag by 72 hours

for feature in lag_features:
    hourly_df[f'{feature}_lag{lag_hours}'] = hourly_df[feature].shift(lag_hours)

# Drop rows with NaN values caused by lagging
hourly_df = hourly_df.dropna()

# Define features and target variable
# Features are: hr, day, mnth, yr, weekday, holiday, 'season', and the lagged features.
# Target feature is cnt, the number of bike rentals.
features = ['hr', 'day', 'mnth', 'yr', 'weekday', "workingday", 'holiday', 'season'] + [f'{feature}_lag{lag_hours}' for feature in lag_features]
X = hourly_df[features]
y = hourly_df['cnt']

# Time-based split
train_size = int(len(X) * 0.8)  # Use 80% of the data for training
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Ensure indices are aligned after splitting
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Initialize and train the Linear Regression model
model = LinearRegression()  # Updated model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Deviation (MAD): {np.mean(np.abs(y_test - y_pred))}")

# Predict future values (example: next 72 hours)
if len(X_test) >= 72:
    future_data = X_test.iloc[:72]
    future_predictions = model.predict(future_data)
    # print(f"Future Predictions: {future_predictions}")

    # Plot the future predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(len(future_predictions)),
        future_predictions,
        label="Predicted",
        color="blue",
        marker="o",
    )
    plt.plot(
        range(len(future_predictions)),
        y_test.iloc[:72].values,
        label="Actual",
        color="orange",
        marker="x",
    )
    plt.title("Future Predictions vs Actual Values")
    plt.xlabel("Time (hours)")
    plt.ylabel("Bike Rentals")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Not enough data in X_test for future predictions.")

