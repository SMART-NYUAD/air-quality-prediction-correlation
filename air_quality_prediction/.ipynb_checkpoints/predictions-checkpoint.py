import mysql.connector
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Database connection
def connect_to_db():
    return mysql.connector.connect(
        host='192.168.50.200',
        user='juanD',
        password='JuanD1_SMART@NYUAD',
        database='home_assistant'
    )

# Data retrieval
def fetch_sensor_data():
    conn = connect_to_db()
    cursor = conn.cursor()
    query = """
    SELECT sensor_statistics_hourly.average, sensor_statistics_coorelation.timestamp
    FROM sensor_statistics_hourly
    JOIN sensor_statistics_coorelation ON sensor_statistics_hourly.update_time_id = sensor_statistics_coorelation.update_time_id
    WHERE sensor_statistics_hourly.metadata_id = 86
    ORDER BY sensor_statistics_coorelation.timestamp
    LIMIT 200
    """
    cursor.execute(query)
    results = cursor.fetchall()
    data = np.array([float(result[0]) for result in results])
    timestamps = np.array([datetime.fromtimestamp(result[1]).strftime('%Y-%m-%d %H:%M:%S') for result in results])
    cursor.close()
    conn.close()
    return data, timestamps

# Load and scale data
data, timestamps = fetch_sensor_data()
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Prepare data for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5
X, y = create_dataset(data_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Splitting data into training and testing
indices = np.arange(X.shape[0])
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.2, random_state=42)
timestamps_test = timestamps[look_back+1:][idx_test]  # Ensure timestamps align with test data

# RNN Model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=1, verbose=2)

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform([y_test])

# Calculate RMSE
from sklearn.metrics import mean_squared_error
testScore = mean_squared_error(y_test[0], predictions[:,0], squared=False)

# Sort the timestamps and corresponding predictions for plotting
sorted_indices = np.argsort(timestamps_test)
sorted_timestamps = timestamps_test[sorted_indices]
sorted_predictions = predictions[sorted_indices, 0]
sorted_actual = y_test[0][sorted_indices]

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(sorted_timestamps, sorted_actual, label='Actual PM 2.5')
plt.plot(sorted_timestamps, sorted_predictions, label='Predicted PM 2.5', alpha=0.7)
plt.xticks(rotation=45)
plt.title('PM2.5 Forecast with RNN')
plt.xlabel('Timestamps')
plt.ylabel('PM 2.5')
plt.legend()
plt.tight_layout()
plt.show()

print(f'Test Score: {testScore} RMSE')
