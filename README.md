# Bike Sharing XGBoost Forecasting

This project aims to predict the number of bike rentals using an XGBoost model. It utilizes 
historical bike sharing data to train the model and evaluate its performance.

## Project Structure

```
BikeSharing
├── src
│   ├── exploratory_analysis.py # Script for quick overview over data and plots 
│   ├── forecast_xgboost.py   # Contains functions for data handling and model operations
│   └── main.py               # Entry point for the project, orchestrates the workflow
├── data                      # Directory for storing the dataset
│   ├── day.csv               # Contains the daily bike rental data (not needed for prediction)
│   ├── hour.csv              # Contains the hourly bike rental data
│   └── Readme.txt            # Description of the dataset, this comes from the original source
├── requirements.txt          # Lists the project dependencies
└── README.md                 # Documentation for the project (this file)
```

## Setup

1. unpack the .zip archive
   ```
   cd BikeSharing
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that the dataset is placed in the `data` directory. The expected file is `hour.csv`.

## Running the Project

To run the project, execute the following command:
```
python src/main.py
```

This will load the data, preprocess it, train the XGBoost model, evaluate its performance, and 
visualize the predictions for the next 72 hours. You can use main.py as orientation on how to use 
the methods.

## Dependencies

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost

Make sure to install these libraries using the `requirements.txt` file provided.