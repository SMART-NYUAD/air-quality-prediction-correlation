import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Load saved model and data
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
timestamps_test = np.load('timestamps_test.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform([y_test])

# Calculate RMSE
testScore = mean_squared_error(y_test[0], predictions[:,0], squared=False)

# Calculate absolute errors
errors = np.abs(y_test[0] - predictions[:,0])

# Sort the timestamps and corresponding predictions for plotting
sorted_indices = np.argsort(timestamps_test)
sorted_timestamps = timestamps_test[sorted_indices]
sorted_predictions = predictions[sorted_indices, 0]
sorted_actual = y_test[0][sorted_indices]

# Plotting results of predictions
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

# Plotting distribution of prediction errors
plt.figure(figsize=(10, 5))
sns.kdeplot(errors, fill=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Absolute Error of Predicted PM 2.5')
plt.ylabel('Density')
plt.show()

print(f'Test Score: {testScore} RMSE')
