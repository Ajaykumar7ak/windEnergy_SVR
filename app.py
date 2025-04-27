import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np
import plotly.express as px

# Load the trained model
model = joblib.load("C:\\uday\\svr_wind_model.pkl")

# Function to fetch Open-Meteo data
def fetch_weather_data():
    url = "https://api.open-meteo.com/v1/forecast?latitude=9.57&longitude=77.78&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m&forecast_days=2&temperature_unit=celsius&windspeed_unit=kmh&precipitation_unit=mm"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['hourly'])
    return df

# Streamlit UI
st.set_page_config(page_title="Wind Energy Forecasting", page_icon="🌬️", layout="wide")
st.title("🌬️ Wind Energy Forecasting App")

# Input mode selection
st.sidebar.title("⚙️ Settings")
input_mode = st.sidebar.radio("Select Input Mode:", ("Live Open-Meteo Data", "Manual Input"))

if input_mode == "Live Open-Meteo Data":
    st.subheader("📍 Live & 2-Day Wind Energy Prediction for Kalasalingam University")

    # Fetch and predict
    live_data = fetch_weather_data()
    X_live = live_data.drop(columns=["time", "hour", "day_of_week"], errors='ignore')

    live_predictions = model.predict(X_live)
    live_data["Predicted Power"] = live_predictions

    # Display current wind energy production
    current_power = live_data.iloc[0]["Predicted Power"]
    st.metric(label="⚡ Current Wind Power Output", value=f"{current_power:.2f} kW")

    # Display forecast table
    st.write("🔍 **2-Day Forecast Data**")
    st.dataframe(live_data)

    # Line Chart
    fig = px.line(live_data, x=live_data.index, y="Predicted Power", 
                  title="Predicted Wind Power Over Time", 
                  labels={"Predicted Power": "Power Output (kW)"})
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.subheader("📊 Statistical Insights")
    st.write("**Mean Predicted Power:**", f"{np.mean(live_data['Predicted Power']):.2f} kW")
    st.write("**Max Predicted Power:**", f"{np.max(live_data['Predicted Power']):.2f} kW")
    st.write("**Min Predicted Power:**", f"{np.min(live_data['Predicted Power']):.2f} kW")

else:
    st.subheader("✍️ Manual Wind Condition Input for 48 Hours Prediction")

    # Manual Input Fields
    col1, col2 = st.columns(2)

    with col1:
        temperature_2m = st.number_input("🌡️ Temperature at 2m (°C)", min_value=-30.0, max_value=50.0, value=25.0)
        relativehumidity_2m = st.number_input("💧 Relative Humidity at 2m (%)", min_value=0.0, max_value=100.0, value=50.0)
        dewpoint_2m = st.number_input("💧 Dew Point at 2m (°C)", min_value=-30.0, max_value=50.0, value=18.0)

    with col2:
        windspeed_10m = st.number_input("🌬️ Wind Speed at 10m (km/h)", min_value=0.0, max_value=150.0, value=15.0)
        windspeed_100m = st.number_input("🌬️ Wind Speed at 100m (km/h)", min_value=0.0, max_value=150.0, value=25.0)
        windgusts_10m = st.number_input("🌪️ Wind Gusts at 10m (km/h)", min_value=0.0, max_value=200.0, value=30.0)

    winddirection_10m = st.number_input("🧭 Wind Direction at 10m (°)", min_value=0.0, max_value=360.0, value=180.0)
    winddirection_100m = st.number_input("🧭 Wind Direction at 100m (°)", min_value=0.0, max_value=360.0, value=190.0)

    # Predict Button
    if st.button("🔮 Predict 48-Hour Wind Power Forecast"):
        # Create 48-hour manual data with small variations
        base_features = np.array([temperature_2m, relativehumidity_2m, dewpoint_2m,
                                  windspeed_10m, windspeed_100m, 
                                  winddirection_10m, winddirection_100m, windgusts_10m])

        np.random.seed(42)  # For reproducibility
        variations = np.random.normal(loc=0.0, scale=0.05, size=(48, len(base_features)))  # ±5% noise
        input_48h = base_features * (1 + variations)

        # Prediction
        manual_predictions = model.predict(input_48h)

        # Create DataFrame
        hours = pd.date_range(start=pd.Timestamp.now(), periods=48, freq="H")
        manual_df = pd.DataFrame({
            "Hour": hours,
            "Predicted Power (kW)": manual_predictions
        })

        # Display table
        st.write("🔍 **48-Hour Manual Forecast Data**")
        st.dataframe(manual_df)

        # Line Chart
        fig = px.line(manual_df, x="Hour", y="Predicted Power (kW)", 
                      title="Predicted Wind Power Over 48 Hours (Manual Input)", 
                      labels={"Predicted Power (kW)": "Power Output (kW)"})
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        st.subheader("📊 Statistical Insights (Manual)")
        st.write(f"**Mean Predicted Power:** {np.mean(manual_df['Predicted Power (kW)']):.2f} kW")
        st.write(f"**Max Predicted Power:** {np.max(manual_df['Predicted Power (kW)']):.2f} kW")
        st.write(f"**Min Predicted Power:** {np.min(manual_df['Predicted Power (kW)']):.2f} kW")

# Footer
st.markdown("---")
st.markdown("### ℹ️ About the Model")
st.markdown("This application uses a **Support Vector Machine (SVM) regression model** trained on historical wind data to predict wind energy production.")
st.markdown("**Weather Data Source:** Open-Meteo API 🌍 (for live mode)")
st.markdown("**Location:** Kalasalingam University 📍")
st.markdown("Developed by team - 08  | exsel 2nd Year** 🚀")
