import numpy as np
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import schedule
import time
import os
import pytz
from dotenv import load_dotenv
import shutil

local_tz = pytz.timezone('Asia/Dubai')

# Convert UNIX timestamp to local time
def unix_to_local(unix_timestamp):
    utc_dt = datetime.utcfromtimestamp(unix_timestamp).replace(tzinfo=pytz.utc)
    local_dt = utc_dt.astimezone(local_tz)
    return local_dt.strftime('%Y-%m-%d %H:%M:%S')

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


# Fetch the last N data points for indoor and outdoor data
def fetch_last_n_points(metadata_ids, n_points=100):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Fetch indoor data
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
            LIMIT {n_points}
            """
            cursor.execute(query)
            results = cursor.fetchall()
            if results:
                data.extend([float(result[0]) for result in results])
                timestamps.extend([unix_to_local(result[1]) for result in results])
            else:
                print("No data returned for metadata_id:", metadata_id)
        return np.array(data), np.array(timestamps)

    # Fetch outdoor data
    def fetch_outdoor_data():
        data = []
        timestamps = []
        query = f"""
        SELECT cities_data.pm2_5_raw, sensor_statistics_coorelation.timestamp as unix_timestamp
        FROM cities_data
        JOIN sensor_statistics_coorelation ON cities_data.update_time_id = sensor_statistics_coorelation.update_time_id
        WHERE location = "Outdoor"
        ORDER BY sensor_statistics_coorelation.timestamp DESC
        LIMIT {n_points}
        """
        cursor.execute(query)
        results = cursor.fetchall()
        data.extend([float(result[0]) for result in results])
        timestamps.extend([unix_to_local(result[1]) for result in results])
        return np.array(data), np.array(timestamps)

    indoor_data, indoor_timestamps = fetch_sensor_data(metadata_ids)
    outdoor_data, outdoor_timestamps = fetch_outdoor_data()
    print("Sample indoor timestamps:", indoor_timestamps[:10])
    print("Sample outdoor timestamps:", outdoor_timestamps[:10])
    cursor.close()
    conn.close()

    # Merge indoor and outdoor data based on timestamps
    all_data = pd.merge(pd.DataFrame({'timestamp': indoor_timestamps, 'indoor.pm2.5': indoor_data}),
                        pd.DataFrame({'timestamp': outdoor_timestamps, 'outdoor.pm2.5': outdoor_data}),
                        on='timestamp', how='inner')

    # Ensure the data is sorted
    all_data.sort_values('timestamp', inplace=True)

    return all_data[['indoor.pm2.5', 'outdoor.pm2.5']].values[-n_points:], all_data['timestamp'].values[-n_points:]

# Get the latest timestamp from merged indoor and outdoor data
def get_latest_timestamp(metadata_ids, n_points=1):
    metadata_ids = [86, 74, 35]
    n_points = 96  # Number of data points to fetch for the model input
    _, last_timestamps = fetch_last_n_points(metadata_ids, n_points=n_points)
    
    # Ensure there are timestamps in the merged data
    if last_timestamps.size > 0:
        # The last timestamp after sorting in fetch_last_n_points is the latest timestamp
        last_timestamp = last_timestamps[-1]
        return datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S')
    else:
        print("No data available to determine latest timestamp.")
        return None


# Upload forecasted data to the database with overwrite capability
def upload_to_db(timestamp, forecasted):
    conn = connect_to_db()
    cursor = conn.cursor()

    query = """
    INSERT INTO forecast_pm25 (timestamp, forecasted_data) 
    VALUES (%s, %s) 
    ON DUPLICATE KEY UPDATE forecasted_data = VALUES(forecasted_data)
    """
    cursor.execute(query, (timestamp, float(forecasted)))
    conn.commit()
    cursor.close()
    conn.close()

# Main function to run every 24 hours
def forecast_pm25():
    
    metadata_ids = [86, 74, 35]
    n_points = 96  # Number of data points to fetch for the model input

    # Fetch last n data points from the merged indoor and outdoor data
    last_n_data, _ = fetch_last_n_points(metadata_ids, n_points=n_points)

    # Scale the data
    last_n_scaled = scaler.transform(last_n_data)

    # Prepare the data for the LSTM model
    X_input = np.reshape(last_n_scaled, (1, n_points, 2))

    # Determine forecast starting timestamp after merging
    last_timestamp = get_latest_timestamp(metadata_ids, n_points=1)
    if last_timestamp is None:
        print("Failed to retrieve the latest timestamp. Exiting forecast.")
        return

    future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, 73)]
    predictions = model.predict(X_input)[0]  # Predict for the next 72 steps
    
    # Inverse transform the predictions
    indoor_pm25_scaler = MinMaxScaler()
    indoor_pm25_scaler.min_, indoor_pm25_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    predictions_inversed = indoor_pm25_scaler.inverse_transform(predictions.reshape(-1, 1))
    predictions_inversed = np.maximum(predictions_inversed, 0)

    # Upload forecasted data to the database with overwrite for existing timestamps
    for i, future_timestamp in enumerate(future_timestamps):
        upload_to_db(future_timestamp.strftime('%Y-%m-%d %H:%M:%S'), predictions_inversed[i][0])


# Schedule the forecast task to run every 24 hours
forecast_pm25()
schedule.every().day.at("00:01").do(forecast_pm25)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)
