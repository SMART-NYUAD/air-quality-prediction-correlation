import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import pickle

# Load model, scaler, and data
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
timestamps_test = np.load('timestamps_test.npy', allow_pickle=True)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Correct approach to handle the inversion of single feature prediction
# Extract the scaling factors for the first feature assumed to be PM2.5
feature_index = 0  # adjust this if your PM2.5 feature is in a different position
scale = scaler.scale_[feature_index]
min_ = scaler.min_[feature_index]

# Make predictions
predictions = model.predict(X_test)

# Manually inverse transform the predictions
predictions_inversed = predictions * scale + min_

# Manually inverse transform the y_test values
y_test_inversed = y_test * scale + min_

# Create DataFrame for easier manipulation
df = pd.DataFrame({
    'Timestamp': timestamps_test,
    'Actual': y_test_inversed.flatten(),
    'Predicted': predictions_inversed.flatten()
})

# Align data: rounding timestamps to the nearest hour (change as needed)
df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.round('H')

# Resample or aggregate data to ensure alignment
df_resampled = df.groupby('Timestamp').mean().reset_index()

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(df_resampled['Timestamp'], df_resampled['Actual'], label='Actual PM2.5', marker='o')
plt.plot(df_resampled['Timestamp'], df_resampled['Predicted'], label='Predicted PM2.5', alpha=0.7, marker='o')
plt.title('Comparison of Actual and Predicted PM2.5 Values')
plt.xlabel('Timestamp')
plt.ylabel('PM2.5 Concentration')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(df_resampled['Actual'], df_resampled['Predicted']))
print(f"RMSE: {rmse}")

# Distribution of prediction errors
errors = np.abs(df_resampled['Actual'] - df_resampled['Predicted'])
plt.figure(figsize=(10, 5))
sns.kdeplot(errors, fill=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Absolute Error of Predicted PM 2.5')
plt.ylabel('Density')
plt.show()
