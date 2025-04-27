import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load Dataset (Replace with your actual CSV file)
data_path = "C:\svm model\preprocessed_wind_energy_data.csv"
df = pd.read_csv(data_path)

# Ensure dataset has expected columns
target_column = "Power"
feature_columns = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                   'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
                   'winddirection_100m', 'windgusts_10m']

# Normalize data for LSTM (Required)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[feature_columns + [target_column]])

# Split into Train & Test
train_size = int(len(df) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

X_test = test_data[:, :-1]  # Features
y_test = test_data[:, -1]   # Actual Power Output

# Reshape for LSTM [Samples, TimeSteps, Features]
X_test_LSTM = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 🔹 Load LSTM Model
lstm_model = tf.keras.models.load_model("C:\\LSTM2\\wind_energy_forecasting_model2.h5")

# 🔹 Load SVR Model
with open("C:\\uday\\svr_wind_model.pkl", "rb") as f:
    svr_model = pickle.load(f)

# 🔹 Make Predictions
lstm_predictions = lstm_model.predict(X_test_LSTM)
svr_predictions = svr_model.predict(X_test)

# Rescale Predictions (Inverse Transform)
lstm_predictions = scaler.inverse_transform(
    np.hstack((X_test, lstm_predictions.reshape(-1, 1)))
)[:, -1]

svr_predictions = scaler.inverse_transform(
    np.hstack((X_test, svr_predictions.reshape(-1, 1)))
)[:, -1]

y_test_actual = scaler.inverse_transform(
    np.hstack((X_test, y_test.reshape(-1, 1)))
)[:, -1]

# 🔹 Evaluate Performance
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📌 Evaluation for {model_name}:")
    print(f"🔸 MAE:  {mae:.4f}")
    print(f"🔸 RMSE: {rmse:.4f}")
    print(f"🔸 R² Score: {r2:.4f}")

# Compare Models
evaluate_model(y_test_actual, lstm_predictions, "LSTM Model")
evaluate_model(y_test_actual, svr_predictions, "SVR Model")
