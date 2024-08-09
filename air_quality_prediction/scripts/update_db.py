import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define your local time zone (e.g., for Abu Dhabi)
local_tz = pytz.timezone('Asia/Dubai')

# Database connection
def connect_to_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

# Convert UNIX timestamp to nearest hour in local time
def unix_to_local_nearest_hour(unix_timestamp):
    utc_dt = datetime.utcfromtimestamp(unix_timestamp).replace(tzinfo=pytz.utc)
    local_dt = utc_dt.astimezone(local_tz)
    # Round to the nearest hour
    if local_dt.minute >= 30:
        local_dt = local_dt + timedelta(hours=1)
    local_dt = local_dt.replace(minute=0, second=0, microsecond=0)
    return local_dt.strftime('%Y-%m-%d %H:%M:%S')

# Convert forecast_pm25 timestamp to nearest hour
def round_to_nearest_hour(timestamp):
    if isinstance(timestamp, pd.Timestamp):
        dt = timestamp
    else:
        dt = pd.Timestamp(timestamp)
    
    # Round to the nearest hour using pandas' functionality
    dt = dt.round('H')
    return dt.strftime('%Y-%m-%d %H:%M:%S')


# Fetch forecast data from forecast_pm25 table
def fetch_forecast_data():
    conn = connect_to_db()
    cursor = conn.cursor()

    query = """
    SELECT timestamp, forecasted_data
    FROM home_assistant.forecast_pm25
    """
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    forecast_data = pd.DataFrame(results, columns=['timestamp', 'forecasted_data'])
    forecast_data['timestamp'] = forecast_data['timestamp'].apply(round_to_nearest_hour)

    return forecast_data


# Fetch and average sensor data based on rounded timestamps
def fetch_and_average_sensor_data(metadata_ids):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Dictionary to store data by rounded timestamps
    data_by_timestamp = {}

    for metadata_id in metadata_ids:
        query = f"""
        SELECT sensor_statistics_hourly.average, sensor_statistics_coorelation.timestamp as unix_timestamp
        FROM sensor_statistics_hourly
        JOIN sensor_statistics_coorelation ON sensor_statistics_hourly.update_time_id = sensor_statistics_coorelation.update_time_id
        WHERE sensor_statistics_hourly.metadata_id = {metadata_id}
        ORDER BY sensor_statistics_coorelation.timestamp DESC
        LIMIT 10000
        """
        cursor.execute(query)
        results = cursor.fetchall()
        for result in results:
            average = float(result[0])
            rounded_timestamp = unix_to_local_nearest_hour(result[1])

            if rounded_timestamp not in data_by_timestamp:
                data_by_timestamp[rounded_timestamp] = []
            data_by_timestamp[rounded_timestamp].append(average)

    cursor.close()
    conn.close()

    # Calculate averages for each timestamp
    averaged_data = pd.DataFrame(
        [(ts, np.mean(values)) for ts, values in data_by_timestamp.items()],
        columns=['timestamp', 'sensor_average']
    )
    
    return averaged_data

# Plot the forecasted data and the sensor average
def plot_forecast_and_sensor_data(forecast_data, averaged_data):
    # Merge the forecast data and averaged sensor data on the timestamp
    merged_data = pd.merge(forecast_data, averaged_data, on='timestamp', how='inner')

    plt.figure(figsize=(14, 7))
    plt.plot(merged_data['timestamp'], merged_data['forecasted_data'], label='Forecasted Data', color='blue', marker='o')
    plt.plot(merged_data['timestamp'], merged_data['sensor_average'], label='Sensor Average', color='red', marker='x')
    
    plt.xlabel('Timestamp')
    plt.ylabel('PM2.5 Value')
    plt.title('Forecasted Data vs Sensor Average')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    metadata_ids = [86, 74, 35]

    # Fetch and process data
    forecast_data = fetch_forecast_data()
    averaged_data = fetch_and_average_sensor_data(metadata_ids)

    # Plot the data
    plot_forecast_and_sensor_data(forecast_data, averaged_data)

if __name__ == "__main__":
    main()
