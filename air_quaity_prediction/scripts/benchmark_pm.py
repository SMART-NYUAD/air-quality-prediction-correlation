import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the model and scaler
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the test data and timestamps
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
timestamps_test = np.load('timestamps_test.npy', allow_pickle=True)  # Ensure timestamps are loaded correctly

# Generate predictions
predictions = model.predict(X_test)

# Create a new scaler for the indoor PM2.5 feature
indoor_pm25_scaler = MinMaxScaler()
indoor_pm25_scaler.min_, indoor_pm25_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

# Inverse transform the predictions and actual values using the new scaler
predictions_inversed = indoor_pm25_scaler.inverse_transform(predictions)
y_test_inversed = indoor_pm25_scaler.inverse_transform(y_test.reshape(-1, 1))

# Ensure no negative values (optional debugging)
print("Predicted values (inverse transformed):", predictions_inversed[:10])
print("Actual values (inverse transformed):", y_test_inversed[:10])

# Calculate errors
errors = predictions_inversed - y_test_inversed

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test_inversed, predictions_inversed))
mae = mean_absolute_error(y_test_inversed, predictions_inversed)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Plotting the results: Actual vs Predicted
plt.figure(figsize=(15, 7))
plt.plot(timestamps_test, y_test_inversed, label='Actual PM2.5', marker='o', linestyle='-', color='blue')
plt.plot(timestamps_test, predictions_inversed, label='Predicted PM2.5', marker='x', linestyle='--', color='red')
plt.title('Comparison of Actual and Predicted PM2.5 Values')
plt.xlabel('Timestamp')
plt.ylabel('PM2.5 Concentration')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Plotting the distribution of errors with density
plt.figure(figsize=(10, 5))
sns.kdeplot(errors.flatten(), fill=True, color="red")
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error (Predicted - Actual)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test_inversed, predictions_inversed, alpha=0.5)
plt.plot([y_test_inversed.min(), y_test_inversed.max()], [y_test_inversed.min(), y_test_inversed.max()], 'k--', lw=2)
plt.title('Actual vs Predicted PM2.5')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.grid(True)
plt.show()
