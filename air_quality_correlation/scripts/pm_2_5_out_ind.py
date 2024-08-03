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

# Fetch the last N data points
def fetch_last_n_points(metadata_ids, csv_file, n_points=100):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Data retrieval for multiple sensors
    def fetch_sensor_data(metadata_ids):
        data = []
        timestamps = []
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
            if results:
                data.extend([float(result[0]) for result in results])
                timestamps.extend([unix_to_local(result[1]) for result in results])
            else:
                print("No data returned for metadata_id:", metadata_id)
        return np.array(data), np.array(timestamps)

    # Read outdoor PM2.5 data from CSV
    def read_csv_data(filename):
        df = pd.read_csv(filename, delimiter=',', parse_dates=['timestamp'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S.%f%z'))
        
        # Convert the timestamps to the correct time zone
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Dubai').dt.tz_localize(None)
        
        df.sort_values('timestamp', ascending=False, inplace=True)
        if df.empty:
            print("CSV file is empty or not loaded properly.")
        return df['pm2.5_raw'].values, df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values

    indoor_data, indoor_timestamps = fetch_sensor_data(metadata_ids)
    outdoor_data, outdoor_timestamps = read_csv_data(csv_file)

    # Merge indoor and outdoor data based on timestamps
    all_data = pd.merge(pd.DataFrame({'timestamp': indoor_timestamps, 'indoor_pm2.5': indoor_data}),
                        pd.DataFrame({'timestamp': outdoor_timestamps, 'outdoor_pm2.5': outdoor_data}),
                        on='timestamp', how='inner')

    # Ensure the data is sorted
    all_data.sort_values('timestamp', inplace=True)

    # Return the merged data and timestamps
    return all_data[['indoor_pm2.5', 'outdoor_pm2.5']], all_data['timestamp']

# Function to calculate correlation and visualize data
def calculate_and_visualize_correlation(metadata_ids, csv_file, n_points=100):
    # Fetch the data
    data, timestamps = fetch_last_n_points(metadata_ids, csv_file, n_points=n_points)

    # Calculate correlation
    correlation = data.corr().iloc[0, 1]
    print(f'Correlation between indoor and outdoor PM2.5: {correlation}')

    # Plotting the data
    plt.figure(figsize=(10, 5))
    plt.scatter(data['outdoor_pm2.5'], data['indoor_pm2.5'], alpha=0.5)
    plt.title('Scatter plot of Indoor vs Outdoor PM2.5')
    plt.xlabel('Outdoor PM2.5')
    plt.ylabel('Indoor PM2.5')
    plt.grid(True)
    plt.show()

    # Plotting the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
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

    metadata_ids = [86, 74, 35]
    csv_file = os.path.join('outdoor_pm25_dataset', latest_csv_file)
    n_points = 1000  # You can adjust this value to experiment with different amounts of data

    calculate_and_visualize_correlation(metadata_ids, csv_file, n_points=n_points)

# Run the main function
if __name__ == "__main__":
    main()
