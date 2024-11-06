import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, ReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import pytz
from dotenv import load_dotenv
from tensorflow.keras.optimizers import Adam


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

def fetch_outdoor_data():
    conn = connect_to_db()
    cursor = conn.cursor()
    data = []
    timestamps = []
    
    query = f"""
    SELECT cities_data.pm2_5_raw, sensor_statistics_coorelation.timestamp as unix_timestamp
    FROM cities_data
    JOIN sensor_statistics_coorelation ON cities_data.update_time_id = sensor_statistics_coorelation.update_time_id
    WHERE location = "Outdoor"
    ORDER BY sensor_statistics_coorelation.timestamp DESC
    LIMIT 10000
    """
    cursor.execute(query)
    results = cursor.fetchall()
    data.extend([float(result[0]) for result in results])
    timestamps.extend([unix_to_local(result[1]) for result in results])
    cursor.close()
    conn.close()
    return np.array(data), np.array(timestamps)

def main():
    # Debug: Print the message indicating the start of the process
    print("Fetching outdoor data from database...")
    
    metadata_ids = [86, 74, 35]  # Define metadata IDs for indoor data
    indoor_data, indoor_timestamps = fetch_sensor_data(metadata_ids)
    
    # Use fetch_outdoor_data instead of reading from CSV
    outdoor_data, outdoor_timestamps = fetch_outdoor_data()

    # Debug before merging
    print("Sample indoor timestamps:", indoor_timestamps[:5])
    print("Sample outdoor timestamps:", outdoor_timestamps[:5])

    # Merge indoor and outdoor data based on timestamps
    all_data = pd.merge(pd.DataFrame({'timestamp': indoor_timestamps, 'indoor.pm2.5': indoor_data}),
                        pd.DataFrame({'timestamp': outdoor_timestamps, 'outdoor.pm2.5': outdoor_data}),
                        on='timestamp', how='inner')

    if all_data.empty:
        print("No overlapping data found. Please check the timestamp alignment.")
    else:
        print("Data successfully merged. Proceeding with scaling and model training...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(all_data[['indoor.pm2.5', 'outdoor.pm2.5']])

        # Verify scaling
        print("Scaling min:", scaler.data_min_)
        print("Scaling max:", scaler.data_max_)

        # Prepare data for LSTM
        def create_dataset(dataset, look_back=72, forecast_horizon=2):
            X, Y = [], []
            for i in range(len(dataset) - look_back - forecast_horizon):
                X.append(dataset[i:(i + look_back), :])
                Y.append(dataset[(i + look_back):(i + look_back + forecast_horizon), 0])
            return np.array(X), np.array(Y)

        look_back = 48
        forecast_horizon = 72
        X, y = create_dataset(data_scaled, look_back, forecast_horizon)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

        # Splitting data into training and testing without shuffling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        timestamps_test = all_data['timestamp'].values[look_back + forecast_horizon:][len(X_train):]  # Ensure timestamps align with test data

        # Sort timestamps in ascending order
        timestamps_test = timestamps_test[::-1]

        # Model with CNN and Bidirectional LSTM
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, 2)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(LSTM(73, return_sequences=True)))
        model.add(Dropout(0.29))
        model.add(Bidirectional(LSTM(73)))
        model.add(Dense(forecast_horizon))
        model.add(ReLU()) 

        learning_rate = 1e-5
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=6, verbose=1)

        # Save the model and other important variables
        model.save('lstm_model.h5')
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        np.save('timestamps_test.npy', timestamps_test)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)

        print("Model and data saved successfully.")

if __name__ == "__main__":
    main()
