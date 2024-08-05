import numpy as np
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import schedule
import time
import os
import pytz
from dotenv import load_dotenv
import shutil

# Import the scraper function
from web_scraper_cities import run_scraper

# Load the model and scaler
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

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

# Function to clear the outdoor_pm25_dataset folder
def clear_dataset_folder():
    folder = 'outdoor_pm25_dataset'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

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
                timestamps.extend([datetime.fromtimestamp(result[1]).strftime('%Y-%m-%d %H:%M:%S') for result in results])
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

    # Debug: Print the last 100 timestamps from indoor and outdoor data before merging
    print("Last 100 indoor timestamps:", indoor_timestamps[:100])
    print("Last 100 outdoor timestamps:", outdoor_timestamps[:100], outdoor_data[:100])

    # Merge indoor and outdoor data based on timestamps
    all_data = pd.merge(pd.DataFrame({'timestamp': indoor_timestamps, 'indoor_pm2.5': indoor_data}),
                        pd.DataFrame({'timestamp': outdoor_timestamps, 'outdoor_pm2.5': outdoor_data}),
                        on='timestamp', how='inner')

    # Ensure the data is sorted
    all_data.sort_values('timestamp', inplace=True)

    # Debug: Print the last 100 timestamps
    print("Last 100 merged timestamps:", all_data['timestamp'].values[-100:])

    return all_data[['indoor_pm2.5', 'outdoor_pm2.5']].values[-n_points:], all_data['timestamp'].values[-n_points:]

# Upload forecasted and actual data to the database
def upload_to_db(timestamp, forecasted, actual=None):
    conn = connect_to_db()
    cursor = conn.cursor()

    if actual is None:
        query = "INSERT INTO forecast_pm25 (timestamp, forecasted_data) VALUES (%s, %s)"
        # Convert numpy.float32 to Python float
        forecasted = float(forecasted) if forecasted is not None else None
        cursor.execute(query, (timestamp, forecasted))
    else:
        query = "UPDATE forecast_pm25 SET actual_data = %s WHERE timestamp = %s"
        # Convert numpy.float32 to Python float for actual data
        actual = float(actual) if actual is not None else None
        cursor.execute(query, (actual, timestamp))

    conn.commit()
    cursor.close()
    conn.close()

# Main function to run every 6 hours
def forecast_pm25():
    # Clear the dataset folder
    clear_dataset_folder()

    # Run the scraper and get the latest CSV file
    latest_csv_file = run_scraper()
    if latest_csv_file is None:
        print("Failed to download the latest CSV file. Exiting.")
        return

    # Debug: Print the latest CSV file name
    print("Using latest CSV file:", latest_csv_file)

    metadata_ids = [86, 74, 35]
    csv_file = os.path.join('outdoor_pm25_dataset', latest_csv_file)
    n_points = 2000  # You can adjust this value to experiment with different amounts of data

    last_n_data, last_n_timestamps = fetch_last_n_points(metadata_ids, csv_file, n_points=n_points)

    # Scale the data
    last_n_scaled = scaler.transform(last_n_data)

    # Prepare the data for the LSTM model
    X_input = np.reshape(last_n_scaled, (1, n_points, 2))

    # Get the current time
    current_time = datetime.now()

    # Forecast the next 2 hours from the current time
    future_timestamps = [current_time + timedelta(hours=i) for i in range(1, 3)]
    predictions = []

    for _ in range(2):
        pred = model.predict(X_input)
        predictions.append(pred[0, 0])
        # Update X_input for the next prediction
        X_input = np.roll(X_input, -1, axis=1)
        # Use the last outdoor data point, modify if you have updated outdoor data
        new_outdoor_data = X_input[0, -1, 1]  
        X_input[0, -1, :] = [pred[0, 0], new_outdoor_data]

    # Inverse transform the predictions
    indoor_pm25_scaler = MinMaxScaler()
    indoor_pm25_scaler.min_, indoor_pm25_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    predictions_inversed = indoor_pm25_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Upload forecasted data to the database
    for i, future_timestamp in enumerate(future_timestamps):
        upload_to_db(future_timestamp.strftime('%Y-%m-%d %H:%M:%S'), predictions_inversed[i][0])

    # Plot the forecasted values
    plt.figure(figsize=(10, 5))
    plt.plot(future_timestamps, predictions_inversed, marker='o', label='Forecasted PM2.5')
    plt.title('Forecast for the Next 2 Hours')
    plt.xlabel('Timestamp')
    plt.ylabel('PM2.5 Concentration')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to upload actual data to the database
def upload_actual_data():
    metadata_ids = [86, 74, 35]
    # Use the latest CSV file for actual data
    latest_csv_file = sorted([f for f in os.listdir('outdoor_pm25_dataset') if f.endswith('.csv')])[-1]
    csv_file = os.path.join('outdoor_pm25_dataset', latest_csv_file)
    n_points = 2  # Fetch the last 2 hours of actual data

    # Fetch and average the data for the last 2 hours
    conn = connect_to_db()
    cursor = conn.cursor()
    data_points = []

    for metadata_id in metadata_ids:
        query = f"""
        SELECT sensor_statistics_hourly.average, sensor_statistics_coorelation.timestamp as unix_timestamp
        FROM sensor_statistics_hourly
        JOIN sensor_statistics_coorelation ON sensor_statistics_hourly.update_time_id = sensor_statistics_coorelation.update_time_id
        WHERE sensor_statistics_hourly.metadata_id = {metadata_id}
        ORDER BY sensor_statistics_coorelation.timestamp DESC
        LIMIT 2
        """
        cursor.execute(query)
        results = cursor.fetchall()
        data_points.extend(results)

    if not data_points:
        print("No data returned for metadata IDs")
        return

    timestamps = [datetime.fromtimestamp(result[1]).strftime('%Y-%m-%d %H:%M:%S') for result in data_points]
    averaged_data = np.mean([float(result[0]) for result in data_points])

    # Upload actual data to the database
    for timestamp in timestamps:
        upload_to_db(timestamp, None, averaged_data)

# Schedule the tasks
forecast_pm25()
schedule.every(2).hours.do(forecast_pm25)
schedule.every(2).hours.at(":30").do(upload_actual_data)  # Adjust the timing as needed

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)
