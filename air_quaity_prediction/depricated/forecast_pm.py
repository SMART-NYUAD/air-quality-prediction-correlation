import numpy as np
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import os

# Load the model and scaler
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the last N datapoints to start the forecasting
def fetch_last_n_points(metadata_ids, csv_file, n_points=100):
    # Database connection
    def connect_to_db():
        return mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )

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
                timestamps.extend([datetime.fromtimestamp(result[1]).strftime('%Y-%m-%d %H:%M:%S') for result in results])
            else:
                print("No data returned for metadata_id:", metadata_id)

        cursor.close()
        conn.close()
        return np.array(data), np.array(timestamps)

    # Read outdoor PM2.5 data from CSV
    def read_csv_data(filename):
        df = pd.read_csv(filename, delimiter=';', parse_dates=['timestamp'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S.%f%z'))
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)  # Remove timezone to align with SQL data
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
    # Get the last N data points
    return all_data[['indoor_pm2.5', 'outdoor_pm2.5']].values[-n_points:], all_data['timestamp'].values[-n_points:]

metadata_ids = [86, 74, 35]
csv_file = 'outdoor_pm25.csv'
n_points = 7000  # You can adjust this value to experiment with different amounts of data
last_n_data, last_n_timestamps = fetch_last_n_points(metadata_ids, csv_file, n_points=n_points)

# Scale the data
last_n_scaled = scaler.transform(last_n_data)

# Prepare the data for the LSTM model
X_input = np.reshape(last_n_scaled, (1, n_points, 2))

# Get the current time
current_time = datetime.now()

# Forecast the next 6 hours from the current time
future_timestamps = [current_time + timedelta(hours=i) for i in range(1, 7)]
predictions = []

for _ in range(6):
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

# Plot the forecasted values
plt.figure(figsize=(10, 5))
plt.plot(future_timestamps, predictions_inversed, marker='o', label='Forecasted PM2.5')
plt.title('Forecast for the Next 6 Hours')
plt.xlabel('Timestamp')
plt.ylabel('PM2.5 Concentration')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
