import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, ReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import time
import os
import pytz
from dotenv import load_dotenv

# Define your local time zone
local_tz = pytz.timezone('Asia/Dubai')
load_dotenv()

# Database connection and data fetching
def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        if conn.is_connected():
            return conn
        else:
            print("Database connection failed.")
            return None
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None


def unix_to_local(unix_timestamp):
    utc_dt = datetime.utcfromtimestamp(unix_timestamp).replace(tzinfo=pytz.utc)
    local_dt = utc_dt.astimezone(local_tz)
    return local_dt.strftime('%Y-%m-%d %H:%M:%S')

def fetch_sensor_data(metadata_ids):
    conn = connect_to_db()
    cursor = conn.cursor()
    data, timestamps = [], []
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
    data, timestamps = [], []
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

# Dataset preparation and evaluation
def create_dataset(dataset, look_back, forecast_horizon=2):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[(i + look_back):(i + look_back + forecast_horizon), 0])
    return np.array(X), np.array(Y)

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def log_results(config, metrics):
    with open("model_performance_log.txt", "a") as log_file:
        log_file.write(f"Configuration: {config}\nMetrics: {metrics}\n\n")

def walk_forward_validation(data, config):
    look_back = config['look_back']
    forecast_horizon = config['forecast_horizon']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    dropout_rate = config['dropout_rate']
    lstm_units = config['lstm_units']
    optimizer_choice = config['optimizer']
    
    optimizer = Adam(learning_rate=learning_rate) if optimizer_choice == 'adam' else RMSprop(learning_rate=learning_rate)

    # Prepare walk-forward splits
    n_splits = 2  # Number of cross-validation splits
    split_size = len(data) // n_splits
    all_split_metrics = []

    for split in range(n_splits):
        train_data = data[:split_size * (split + 1)]
        test_data = data[split_size * (split + 1):split_size * (split + 2)]

        if test_data.size == 0:
            break  # No more data to test

        X_train, y_train = create_dataset(train_data, look_back, forecast_horizon)
        X_test, y_test = create_dataset(test_data, look_back, forecast_horizon)

        # Model architecture based on current configuration
        model = Sequential([
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, 2)),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Dropout(dropout_rate),
            Bidirectional(LSTM(lstm_units)),
            Dense(forecast_horizon),
            ReLU()
        ])
        
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Training
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Prediction and evaluation
        y_pred = model.predict(X_test)
        rmse, mae, r2 = evaluate_model(y_test, y_pred)

        # Log each split result
        split_metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        all_split_metrics.append(split_metrics)

    # Calculate average metrics across all splits
    avg_rmse = np.mean([m['RMSE'] for m in all_split_metrics])
    avg_mae = np.mean([m['MAE'] for m in all_split_metrics])
    avg_r2 = np.mean([m['R2'] for m in all_split_metrics])
    final_metrics = {'Avg RMSE': avg_rmse, 'Avg MAE': avg_mae, 'Avg R2': avg_r2}

    log_results(config, final_metrics)
    return avg_rmse

# Bayesian Optimization
search_space = [
    Integer(48, 200, name='look_back'),
    Integer(30, 150, name='epochs'),
    Integer(1, 8, name='batch_size'),
    Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
    Real(0.1, 0.5, name='dropout_rate'),
    Integer(50, 300, name='lstm_units'),
    Categorical(['adam', 'rmsprop'], name='optimizer')
]

@use_named_args(search_space)
def objective(**params):
    config = {
        'look_back': params['look_back'],
        'forecast_horizon': 72,
        'epochs': params['epochs'],
        'batch_size': params['batch_size'],
        'learning_rate': params['learning_rate'],
        'dropout_rate': params['dropout_rate'],
        'lstm_units': params['lstm_units'],
        'optimizer': params['optimizer']
    }
    
    results = walk_forward_validation(data_scaled, config)
    return results  # Minimize RMSE

def main():
    print("Fetching and preparing data...")
    metadata_ids = [86, 74, 35]
    indoor_data, indoor_timestamps = fetch_sensor_data(metadata_ids)
    outdoor_data, outdoor_timestamps = fetch_outdoor_data()

    all_data = pd.merge(pd.DataFrame({'timestamp': indoor_timestamps, 'indoor.pm2.5': indoor_data}),
                        pd.DataFrame({'timestamp': outdoor_timestamps, 'outdoor.pm2.5': outdoor_data}),
                        on='timestamp', how='inner')

    if all_data.empty:
        print("No overlapping data found. Please check the timestamp alignment.")
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        global data_scaled
        data_scaled = scaler.fit_transform(all_data[['indoor.pm2.5', 'outdoor.pm2.5']])

        result = gp_minimize(objective, search_space, n_calls=20, random_state=0)
        best_params = result.x
        best_score = result.fun
        print("Best parameters:", best_params)
        print("Best RMSE:", best_score)

if __name__ == "__main__":
    main()
