import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_columns = 99

# Load the hourly data
hourly_df = pd.read_csv("data/hour.csv")
print(hourly_df.info())
print(hourly_df.describe())
print(hourly_df.head())
print(hourly_df.shape)

# Load the daily data.
# It is used to plot different features of the dataset over time.
daily_df = pd.read_csv("data/day.csv")

# Convert dteday to datetime for proper plotting
daily_df["dteday"] = pd.to_datetime(daily_df["dteday"])
hourly_df["dteday"] = pd.to_datetime(hourly_df["dteday"])

# Plot daily bike rentals
plt.figure(figsize=(10, 6))
plt.plot(daily_df["dteday"], daily_df["cnt"], label="# Bike Rentals", color="blue")
plt.xlabel("Date")
plt.ylabel("Bike Rentals")
plt.title("Daily Bike Rentals Over Time")
plt.legend()
plt.grid(True)
plt.show()


# Plot daily temperature
plt.figure(figsize=(10, 6))
plt.plot(
    daily_df["dteday"], daily_df["temp"], label="Temperature in normalized Celsius", color="blue"
)
plt.xlabel("Date")
plt.ylabel("Temperature (normalized)")
plt.title("Daily Temperature Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot daily humidity
plt.figure(figsize=(10, 6))
plt.plot(daily_df["dteday"], daily_df["hum"], label="Humidity in Normalized Percent", color="blue")
plt.xlabel("Date")
plt.ylabel("Humidity in % (normalized)")
plt.title("Daily Humidity Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot daily windspeed
plt.figure(figsize=(10, 6))
plt.plot(daily_df["dteday"], daily_df["windspeed"], label="Normalized Windspeed", color="blue")
plt.xlabel("Date")
plt.ylabel("Windspeed (normalized)")
plt.title("Daily Windspeed Over Time")
plt.legend()
plt.grid(True)
plt.show()


# Plot monthly registered vs. casual users as a bar chart
# Aggregate the user numbers monthly for plotting
monthly_df = daily_df.resample("ME", on="dteday").sum()
plt.figure(figsize=(10, 6))
width = 0.4  # Width of the bars
x = np.arange(len(monthly_df.index))
plt.bar(
    x - width / 2, monthly_df["registered"], width=width, label="Registered Users", color="blue"
)
plt.bar(x + width / 2, monthly_df["casual"], width=width, label="Casual Users", color="orange")
plt.xlabel("Month")
plt.ylabel("Users")
plt.title("Monthly Registered vs Casual Users")
plt.xticks(x, monthly_df.index.strftime("%Y-%m"), rotation=45)  # Format month labels
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()


# Plot daily bike rentals for May 2012
may_df = daily_df[(daily_df["dteday"] >= "2012-05-01") & (daily_df["dteday"] <= "2012-05-31")]
plt.figure(figsize=(15, 6))
plt.plot(may_df["dteday"], may_df["cnt"], label="# Bike Rentals", color="green")

# Add weather situation as annotations
weather_labels = {1: "Clear", 2: "Mist", 3: "Light Snow/Rain", 4: "Heavy Snow/Rain"}
for i, row in may_df.iterrows():
    plt.text(
        row["dteday"],
        row["cnt"] + 50,  # Offset for better visibility
        weather_labels.get(row["weathersit"], "Unknown"),
        fontsize=8,
        ha="center",
        color="black",
    )
plt.xlabel("Date")
plt.ylabel("Bike Rentals")
plt.title("Daily Bike Rentals (May 2012)")
plt.legend()
plt.grid(True)
plt.show()


# Plot daily bike rentals for August 2012
august_df = daily_df[(daily_df["dteday"] >= "2012-08-01") & (daily_df["dteday"] <= "2012-08-31")]
plt.figure(figsize=(15, 6))
plt.plot(august_df["dteday"], august_df["cnt"], label="# Bike Rentals", color="green")
# Add weather situation as annotations
weather_labels = {1: "Clear", 2: "Mist", 3: "Light Snow/Rain", 4: "Heavy Snow/Rain"}
for i, row in august_df.iterrows():
    plt.text(
        row["dteday"],
        row["cnt"] + 50,  # Offset for better visibility
        weather_labels.get(row["weathersit"], "Unknown"),
        fontsize=8,
        ha="center",
        color="black",
    )
plt.xlabel("Date")
plt.ylabel("Bike Rentals")
plt.title("Daily Bike Rentals (August 2012)")
plt.legend()
plt.grid(True)
plt.show()


# Plot hourly rentals for September 2 to 8, 2012
september_week_hourly_df = hourly_df[
    (hourly_df["dteday"] >= "2012-09-02") & (hourly_df["dteday"] <= "2012-09-08")
]

plt.figure(figsize=(15, 6))
plt.plot(
    september_week_hourly_df["dteday"] + pd.to_timedelta(september_week_hourly_df["hr"], unit="h"),
    september_week_hourly_df["cnt"],
    label="# Hourly Rentals",
    color="purple",
)
plt.xlabel("Date and Hour")
plt.ylabel("Bike Rentals")
plt.title("Hourly Bike Rentals (September 2 to 8, 2012)")
plt.legend()
plt.grid(True)
plt.show()
