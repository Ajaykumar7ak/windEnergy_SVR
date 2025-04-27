import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the trained model
model = joblib.load("C:\\uday\\svr_wind_model.pkl")

# Load dataset
df = pd.read_csv(r"C:\wind_2\Location1.csv")

# Convert Time column to datetime
df["Time"] = pd.to_datetime(df["Time"])
df = df.set_index("Time")

# Define features and target
X_test = df.drop(columns=["Power"])
y_test = df["Power"]

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}, MSE: {mse}, R² Score: {r2}")
