import mysql.connector
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import os

# Database connection
def connect_to_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

# Data retrieval for the last 5 hours
def fetch_last_5_hours():
    conn = connect_to_db()
    cursor = conn.cursor()
    query = """
    SELECT sensor_statistics_hourly.average, sensor_statistics_coorelation.timestamp
    FROM sensor_statistics_hourly
    JOIN sensor_statistics_coorelation ON sensor_statistics_hourly.update_time_id = sensor_statistics_coorelation.update_time_id
    WHERE sensor_statistics_hourly.metadata_id = 86
    ORDER BY sensor_statistics_coorelation.timestamp DESC
    LIMIT 5
    """
    cursor.execute(query)
    results = cursor.fetchall()
    data = np.array([float(result[0]) for result in results][::-1])  # Reverse to chronological order
    timestamps = np.array([datetime.fromtimestamp(result[1]) for result in results][::-1])
    cursor.close()
    conn.close()
    return data, timestamps

# Load the model and the scaler
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Get the last 5 hours of data
data, last_timestamps = fetch_last_5_hours()

# Scale the data
data_scaled = scaler.transform(data.reshape(-1, 1))

# Prepare initial input for the model
X_input = np.reshape(data_scaled, (1, 5, 1))  # Reshape to the form the model expects

# Generate future timestamps
last_known_time = last_timestamps[-1]
future_timestamps = [last_known_time + timedelta(hours=i+1) for i in range(12)]

# Predict the next 12 hours iteratively
predictions = []
for _ in range(12):
    pred = model.predict(X_input)
    predictions.append(scaler.inverse_transform(pred)[0, 0])  # Inverse transform and store the predicted value
    # Update X_input for the next prediction
    X_input = np.roll(X_input, -1, axis=1)
    X_input[0, -1, 0] = pred  # Insert the new prediction at the end of the array

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(future_timestamps, predictions, marker='o', label='Forecasted PM 2.5')
plt.title('Forecast for the Next 12 Hours')
plt.xlabel('Timestamp')
plt.ylabel('PM 2.5')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
