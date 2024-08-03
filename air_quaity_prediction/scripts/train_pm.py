import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
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

# Data retrieval for multiple sensors
def fetch_sensor_data(metadata_ids):
    conn = connect_to_db()
    cursor = conn.cursor()
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

    cursor.close()
    conn.close()
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

def main():
    # Run the scraper and get the latest CSV file
    latest_csv_file = run_scraper()
    if latest_csv_file is None:
        print("Failed to download the latest CSV file. Exiting.")
        return

    # Debug: Print the latest CSV file name
    print("Using latest CSV file:", latest_csv_file)

    metadata_ids = [86, 74, 35]
    indoor_data, indoor_timestamps = fetch_sensor_data(metadata_ids)
    csv_file = os.path.join('outdoor_pm25_dataset', latest_csv_file)
    outdoor_data, outdoor_timestamps = read_csv_data(csv_file)

    # Debug before merging
    print("Sample indoor timestamps:", indoor_timestamps[:5])
    print("Sample outdoor timestamps:", outdoor_timestamps[:5])

    # Merge indoor and outdoor data based on timestamps
    all_data = pd.merge(pd.DataFrame({'timestamp': indoor_timestamps, 'indoor_pm2.5': indoor_data}),
                        pd.DataFrame({'timestamp': outdoor_timestamps, 'outdoor_pm2.5': outdoor_data}),
                        on='timestamp', how='inner')

    if all_data.empty:
        print("No overlapping data found. Please check the timestamp alignment.")
    else:
        print("Data successfully merged. Proceeding with scaling and model training...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(all_data[['indoor_pm2.5', 'outdoor_pm2.5']])

        # Verify scaling
        print("Scaling min:", scaler.data_min_)
        print("Scaling max:", scaler.data_max_)

        # Prepare data for LSTM
        def create_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), :]
                X.append(a)
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)

        look_back = 7000
        X, y = create_dataset(data_scaled, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

        # Splitting data into training and testing without shuffling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        timestamps_test = all_data['timestamp'].values[look_back+1:][len(X_train):]  # Ensure timestamps align with test data

        # Sort timestamps in ascending order
        timestamps_test = timestamps_test[::-1]

        # RNN Model
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 2)))  # Adjust input shape for two features
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=80, batch_size=1, verbose=2)

        # Save the model and other important variables
        model.save('lstm_model.h5')
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        np.save('timestamps_test.npy', timestamps_test)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)

        print("Model and data saved successfully.")

if __name__ == "__main__":
    main()
