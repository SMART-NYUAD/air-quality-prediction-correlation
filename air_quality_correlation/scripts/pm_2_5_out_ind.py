import numpy as np
import mysql.connector
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pytz
from dotenv import load_dotenv

# Import the scraper function
from web_scraper_cities import run_scraper

# Define your local time zone (e.g., for Abu Dhabi)
local_tz = pytz.timezone('Asia/Dubai')

# Load environment variables from .env file
load_dotenv()

# Database connection
def connect_to_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

# Convert UNIX timestamp to local time
def unix_to_local(unix_timestamp):
    utc_dt = datetime.utcfromtimestamp(unix_timestamp).replace(tzinfo=pytz.utc)
    local_dt = utc_dt.astimezone(local_tz)
    return local_dt.strftime('%Y-%m-%d %H:%M:%S')

# Fetch the last N data points for multiple sensors and average them
def fetch_and_average_data(sensor_ids, n_points=1000):
    conn = connect_to_db()
    cursor = conn.cursor()

    def fetch_sensor_data(metadata_id):
        query = f"""
        SELECT sensor_statistics_hourly.average, sensor_statistics_coorelation.timestamp as unix_timestamp
        FROM sensor_statistics_hourly
        JOIN sensor_statistics_coorelation ON sensor_statistics_hourly.update_time_id = sensor_statistics_coorelation.update_time_id
        WHERE sensor_statistics_hourly.metadata_id = {metadata_id}
        ORDER BY sensor_statistics_coorelation.timestamp DESC
        LIMIT {n_points}
        """
        cursor.execute(query)
        results = cursor.fetchall()
        if results:
            data = [float(result[0]) for result in results]
            timestamps = [unix_to_local(result[1]) for result in results]
        else:
            data = []
            timestamps = []
            print(f"No data returned for metadata_id: {metadata_id}")
        return data, timestamps

    # Fetch data for each sensor and store in a list
    data_frames = []
    for sensor_id in sensor_ids:
        data, timestamps = fetch_sensor_data(sensor_id)
        df = pd.DataFrame({
            f'sensor_{sensor_id}': data,
            'timestamp': timestamps
        })
        data_frames.append(df)

    # Merge data from all sensors on timestamp
    all_data = data_frames[0]
    for df in data_frames[1:]:
        all_data = pd.merge(all_data, df, on='timestamp', how='inner')

    # Calculate the average of the sensor readings
    all_data['average'] = all_data[[f'sensor_{sensor_id}' for sensor_id in sensor_ids]].mean(axis=1)

    return all_data[['average', 'timestamp']]

# Function to calculate correlation and visualize data
def calculate_and_visualize_correlation(sensor_dict, csv_file, n_points=1000):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Create a DataFrame for all sensor averages
    data_frames = []
    for sensor_name, sensor_ids in sensor_dict.items():
        if sensor_name != 'Outdoor_PM2.5':  # Skip the outdoor_pm2.5 since it's from the CSV
            df = fetch_and_average_data(sensor_ids, n_points=n_points)
            df.rename(columns={'average': sensor_name}, inplace=True)
            data_frames.append(df)

    # Merge all data frames on timestamp
    all_data = data_frames[0]
    for df in data_frames[1:]:
        all_data = pd.merge(all_data, df, on='timestamp', how='inner')

    # Read outdoor PM2.5, pm10_raw, and co2 data from CSV
    def read_csv_data(filename):
        df = pd.read_csv(filename, delimiter=',', parse_dates=['timestamp'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S.%f%z'))
        
        # Convert the timestamps to the correct time zone
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Dubai').dt.tz_localize(None)
        
        df.sort_values('timestamp', ascending=False, inplace=True)
        if df.empty:
            print("CSV file is empty or not loaded properly.")
        return df[['pm2.5_raw', 'pm10_raw', 'co2']], df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values

    outdoor_data, outdoor_timestamps = read_csv_data(csv_file)

    # Create DataFrame for outdoor data (pm2.5, pm10_raw, co2)
    outdoor_df = pd.DataFrame({
        'timestamp': outdoor_timestamps,
        'Outdoor_PM2.5': outdoor_data['pm2.5_raw'],
    })

    # Merge indoor data and outdoor data based on timestamps
    all_data = pd.merge(all_data, outdoor_df, on='timestamp', how='inner')

    # Ensure the data is sorted
    all_data.sort_values('timestamp', inplace=True)

    # Calculate the correlation matrix
    correlation_matrix = all_data.drop(columns=['timestamp']).corr()

    # Print the correlation matrix
    print('Correlation Matrix:')
    print(correlation_matrix)

    # Plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix for All Variables')
    plt.show()

# Main function
def main():
    # Run the scraper and get the latest CSV file
    latest_csv_file = run_scraper()
    if latest_csv_file is None:
        print("Failed to download the latest CSV file. Exiting.")
        return

    # Debug: Print the latest CSV file name
    print("Using latest CSV file:", latest_csv_file)

    # Sensor names and their corresponding metadata_ids
    sensor_dict = {
        'Indoor_PM2.5': [86, 74, 35],
        'Indoor_Humidity': [91, 79, 40],
        'Indoor_Temperature': [90, 78, 39],
        'Outdoor_PM2.5': None  # Placeholder, since it's from the CSV
    }
    csv_file = os.path.join('outdoor_pm25_dataset', latest_csv_file)
    n_points = 1000  # Adjust this value to experiment with different amounts of data

    calculate_and_visualize_correlation(sensor_dict, csv_file, n_points=n_points)

# Run the main function
if __name__ == "__main__":
    main()
