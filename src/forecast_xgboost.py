"""
This script loads the hourly Bike Sharing dataset and trains an XGBoost model to predict the number
of bike rentals (cnt).
The methods for loading, preprocessing and splitting the data have been encapsulated in functions.
The sames has been done for the methods for model training, evaluation and visualizing the
predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from the specified file path.

    Args:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df: pd.DataFrame, lag_hours: int = 72) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the data by creating lagged features and removing NaN values.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        lag_hours (int): The number of hours to lag the features. Default is 72.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and the target
        variable Series.
    """
    hourly_df = df.copy()

    # Create calendarical features
    # The features hr (hour), mnth (month), yr (year), and weekday (day of the week) are already
    # present in the dataset.
    # Only day of the month needs to be created from the column dteday.
    hourly_df["dteday"] = pd.to_datetime(df["dteday"])
    hourly_df["day"] = hourly_df["dteday"].dt.day

    # Create lag features:
    # These are created from the weather-realted features weathersit, temp, hum, windspeed, and the
    # target variable cnt
    lag_features = ["weathersit", "temp", "hum", "windspeed", "cnt"]

    for feature in lag_features:
        hourly_df[f"{feature}_lag{lag_hours}"] = hourly_df[feature].shift(lag_hours)

    # Drop rows with NaN values caused by lagging
    hourly_df = hourly_df.dropna()

    # Define features and target variable
    # Features are: hr, day, mnth, yr, weekday, holiday, season, and the lagged features.
    # Target feature is cnt, the number of bike rentals.
    features = ["hr", "day", "mnth", "yr", "weekday", "holiday", "season"] + [
        f"{feature}_lag{lag_hours}" for feature in lag_features
    ]
    X = hourly_df[features]
    y = hourly_df["cnt"]

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, train_size: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets based on a time-based split.
    A time-based split upholds the time series nature of the data.

    Args:
        X (pd.DataFrame): The features DataFrame.
        y (pd.Series): The target variable Series.
        train_size (float): The proportion of the data to use for training. Default is 0.8.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training
        features DataFrame, testing features DataFrame, training target Series, and testing target Series.
    """
    train_size = int(len(X) * train_size)  # Use 80% of the data for training
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Ensure indices are aligned after splitting
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Train an XGBoost model on the training data.

    Args:
        X_train (pd.DataFrame): The training features DataFrame.
        y_train (pd.Series): The training target Series.

    Returns:
        XGBRegressor: The trained XGBoost model.
    """
    model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float, float, float]:
    """
    Evaluate the trained model on the testing data.

    Args:
        model (XGBRegressor): The trained XGBoost model.
        X_test (pd.DataFrame): The testing features DataFrame.
        y_test (pd.Series): The testing target Series.

    Returns:
        tuple[float, float, float]: A tuple containing the RMSE, MAE, and R-squared values.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rmse, mae, r2


def visualize_prediction(
    model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series, pred_length: int = 72
) -> None:
    """
    Visualize the prediction of the model.

    Args:
        model (XGBRegressor): The trained XGBoost model.
        X_test (pd.DataFrame): The testing features DataFrame.
        y_test (pd.Series): The testing target Series.
        pred_length (int): The number of hours to predict. Default is 72.
    """
    if len(X_test) >= pred_length:  # Check whether we have enough data
        future_data = X_test.iloc[:pred_length]
        future_predictions = model.predict(future_data)

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
            y_test.iloc[:pred_length].values,
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
    else:  # If we have not enough data, print a warning.
        print("Not enough data in X_test for future predictions.")


if __name__ == "__main__":
    file_path = "data/hour.csv"  # Path to the dataset
    pred_hours = 72  # Number of hours to predict. Is also used for lagging the features.

    print("Loading data...")
    data = load_data(file_path)

    print("Preprocessing data...")
    X, y = preprocess_data(data, lag_hours=pred_hours)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_size=0.8)

    print("Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train)

    print("Evaluating model:")
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared: {r2}")

    print(f"Visualizing predictions for the next {pred_hours} hours...")
    visualize_prediction(model, X_test, y_test, pred_length=pred_hours)
    print("Done.")
