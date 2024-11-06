import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, ReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import time
import pickle
import os
import pytz
from dotenv import load_dotenv

# Define your local time zone
local_tz = pytz.timezone('Asia/Dubai')
load_dotenv()

def connect_to_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

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
    n_splits = 5  # Number of cross-validation splits
    split_size = len(data) // n_splits
    all_results = []

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
        start_time = time.time()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        training_time = time.time() - start_time

        # Prediction and evaluation
        y_pred = model.predict(X_test)
        rmse, mae, r2 = evaluate_model(y_test, y_pred)

        # Log each split result
        split_metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Training Time': training_time}
        config_metrics = {**config, **split_metrics}  # Combine configuration and metrics
        all_results.append(config_metrics)
        print(f"Split {split + 1}/{n_splits} - Metrics: {split_metrics}")
        log_results(config, split_metrics)  # Log results per split

    return all_results

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
        data_scaled = scaler.fit_transform(all_data[['indoor.pm2.5', 'outdoor.pm2.5']])

        # Define configurations for hyperparameter tuning
        configs = [
            # Baseline configurations
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 60, 'batch_size': 1, 'learning_rate': 0.001, 
            'dropout_rate': 0.3, 'lstm_units': 100, 'optimizer': 'adam'},
            {'look_back': 120, 'forecast_horizon': 72, 'epochs': 50, 'batch_size': 2, 'learning_rate': 0.0005, 
            'dropout_rate': 0.4, 'lstm_units': 150, 'optimizer': 'rmsprop'},
            
            # Variations on look_back
            {'look_back': 72, 'forecast_horizon': 72, 'epochs': 40, 'batch_size': 1, 'learning_rate': 0.001, 
            'dropout_rate': 0.2, 'lstm_units': 80, 'optimizer': 'adam'},
            {'look_back': 144, 'forecast_horizon': 72, 'epochs': 80, 'batch_size': 4, 'learning_rate': 0.0005, 
            'dropout_rate': 0.5, 'lstm_units': 200, 'optimizer': 'adam'},

            # Variations on learning rate
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 100, 'batch_size': 1, 'learning_rate': 0.01, 
            'dropout_rate': 0.3, 'lstm_units': 120, 'optimizer': 'adam'},
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 70, 'batch_size': 1, 'learning_rate': 0.0001, 
            'dropout_rate': 0.3, 'lstm_units': 100, 'optimizer': 'rmsprop'},
            
            # Variations on dropout rate and batch size
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 60, 'batch_size': 8, 'learning_rate': 0.001, 
            'dropout_rate': 0.1, 'lstm_units': 100, 'optimizer': 'adam'},
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 60, 'batch_size': 16, 'learning_rate': 0.001, 
            'dropout_rate': 0.5, 'lstm_units': 100, 'optimizer': 'adam'},
            
            # LSTM units variation (for network depth and capacity)
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 50, 'batch_size': 1, 'learning_rate': 0.001, 
            'dropout_rate': 0.3, 'lstm_units': 50, 'optimizer': 'adam'},
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 50, 'batch_size': 1, 'learning_rate': 0.001, 
            'dropout_rate': 0.3, 'lstm_units': 200, 'optimizer': 'adam'},

            # High epochs with different configurations
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 150, 'batch_size': 2, 'learning_rate': 0.001, 
            'dropout_rate': 0.3, 'lstm_units': 150, 'optimizer': 'adam'},
            {'look_back': 96, 'forecast_horizon': 72, 'epochs': 200, 'batch_size': 4, 'learning_rate': 0.0005, 
            'dropout_rate': 0.4, 'lstm_units': 120, 'optimizer': 'rmsprop'},
            
            # Low and high look_back extremes
            {'look_back': 48, 'forecast_horizon': 72, 'epochs': 30, 'batch_size': 8, 'learning_rate': 0.005, 
            'dropout_rate': 0.1, 'lstm_units': 50, 'optimizer': 'adam'},
            {'look_back': 200, 'forecast_horizon': 72, 'epochs': 100, 'batch_size': 1, 'learning_rate': 0.0002, 
            'dropout_rate': 0.6, 'lstm_units': 250, 'optimizer': 'rmsprop'}
        ]



        for config in configs:
            print(f"Evaluating configuration: {config}")
            results = walk_forward_validation(data_scaled, config)
            print(f"Results for configuration {config}:", results)

if __name__ == "__main__":
    main()
