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

# Plot daily humidity
plt.figure(figsize=(10, 6))
plt.plot(daily_df['dteday'], daily_df['hum'], label='Humidity in Normalized Percent', color='blue')
plt.xlabel('Date')
plt.ylabel('Humidity in % (normalized)')
plt.title('Daily Humidity Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot daily windspeed
plt.figure(figsize=(10, 6))
plt.plot(daily_df['dteday'], daily_df['windspeed'], label='Normalized Windspeed', color='blue')
plt.xlabel('Date')
plt.ylabel('Windspeed (normalized)')
plt.title('Daily Windspeed Over Time')
plt.legend()
plt.grid(True)
plt.show()


# Plot monthly registered vs. casual users as a bar chart
# Aggregate the user numbers monthly for plotting
monthly_df = daily_df.resample('M', on='dteday').sum()
plt.figure(figsize=(10, 6))
width = 0.4  # Width of the bars
x = np.arange(len(monthly_df.index))
plt.bar(x - width/2, monthly_df['registered'], width=width, label='Registered Users', color='blue')
plt.bar(x + width/2, monthly_df['casual'], width=width, label='Casual Users', color='orange')
plt.xlabel('Month')
plt.ylabel('Users')
plt.title('Monthly Registered vs Casual Users')
plt.xticks(x, monthly_df.index.strftime('%Y-%m'), rotation=45)  # Format month labels
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()