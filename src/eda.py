import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

pd.options.display.max_columns=99

# Load the hourly data
hourly_df = pd.read_csv('data/hour.csv')
print(hourly_df.info())
print(hourly_df.describe())
print(hourly_df.head())
print(hourly_df.shape)

daily_df = pd.read_csv('data/day.csv')

# Convert dteday to datetime for proper plotting
daily_df['dteday'] = pd.to_datetime(daily_df['dteday'])

# Plot daily bike rentals
plt.figure(figsize=(10, 6))
plt.plot(daily_df['dteday'], daily_df['cnt'], label='# Bike Rentals', color='blue')
plt.xlabel('Date')
plt.ylabel('Bike Rentals')
plt.title('Daily Bike Rentals Over Time')
plt.legend()
plt.grid(True)
plt.show()


# Plot daily temperature
plt.figure(figsize=(10, 6))
plt.plot(daily_df['dteday'], daily_df['temp'], label='Temperature in normalized Celsius', color='blue')
plt.xlabel('Date')
plt.ylabel('Temperature (normalized)')
plt.title('Daily Temperature Over Time')
plt.legend()
plt.grid(True)
plt.show()
