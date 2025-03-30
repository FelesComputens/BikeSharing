from forecast_xgboost import (
    evaluate_model,
    load_data,
    preprocess_data,
    split_data,
    train_xgboost_model,
    visualize_prediction_vs_actual,
)


def main():
    file_path = "data/hour.csv"
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
    rmse, mae, r2, mad = evaluate_model(model, X_test, y_test)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared: {r2}")
    print(f"Mean Absolute Deviation (MAD): {mad}")

    # The model now has been created. The following code is only needed for evaluating it.
    print(f"Visualizing predictions for the first {pred_hours} of the test set...")
    visualize_prediction_vs_actual(model, X_test, y_test, pred_length=pred_hours)
    print("Done.")


if __name__ == "__main__":
    main()
