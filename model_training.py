import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv(r"C:\wind_2\Location1.csv")  # Add 'r' before the string
  # Replace with actual dataset path

# Convert Time column to datetime if needed
df["Time"] = pd.to_datetime(df["Time"])
df = df.set_index("Time")

# Define features and target
X = df.drop(columns=["Power"])
y = df["Power"]

# Time-based train-test split (80-20)
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train SVR model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train, y_train)

# Predictions
y_pred = svr_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE: {mae}, MSE: {mse}")

# Save model
joblib.dump(svr_model, "svr_wind_model.pkl")
