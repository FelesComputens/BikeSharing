import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the hourly data
hourly_df = pd.read_csv('data/hour.csv')

# Convert dteday to datetime
hourly_df['dteday'] = pd.to_datetime(hourly_df['dteday'])

# The features 
# Feature engineering: Create time-based features
# The features hr (hour), mnth (month), yr (year), and weekday (day of the week) are already present in the dataset.
# Only day of the month needs to be created
hourly_df['day'] = hourly_df['dteday'].dt.day

# Create lag features from weatersit, temp, hum, windspeed, and the target variable cnt
lag_features = ['weathersit', 'temp', 'hum', 'windspeed', 'cnt']
lag_hours = 72  # Lag by 72 hours

for feature in lag_features:
    hourly_df[f'{feature}_lag{lag_hours}'] = hourly_df[feature].shift(lag_hours)

# Drop rows with NaN values caused by lagging
hourly_df = hourly_df.dropna()

# Convert weathersit_lag{lag_hours} to an integer since it is categorical.
hourly_df[f'weathersit_lag{lag_hours}'] = hourly_df[f'weathersit_lag{lag_hours}'].astype(int)

# Define categorical features
# categorical_features = ['holiday', 'season', 'weekday', f'weathersit_lag{lag_hours}']
categorical_features = ['holiday', 'season', 'weekday', f'weathersit_lag{lag_hours}']

# Define features and target variable
# Features are: hr, day, mnth, yr, weekday, holiday, 'season', and the lagged features.
# Target feature is cnt, the number of bike rentals.
features = ['hr', 'day', 'mnth', 'yr', 'weekday', 'holiday', 'season'] + [f'{feature}_lag{lag_hours}' for feature in lag_features]
X = hourly_df[features]
y = hourly_df['cnt']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the CatBoost model
model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
model.fit(X_train, y_train, cat_features=categorical_features)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

# Predict future values (example: next 72 hours)
if len(X_test) >= 72:
    future_data = X_test.iloc[:72]
    future_predictions = model.predict(future_data)
    print(f"Future Predictions: {future_predictions}")

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