import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load Data
df = pd.read_csv(r"C:\Users\hi\Documents\NLP\Time_Series_Analysis\Data_Sets\wp_log_peyton_manning.csv")
df['ds'] = pd.to_datetime(df['ds'])
df.set_index('ds', inplace=True)

# Load ARIMA Model
with open("ML_Model/arima_model.pkl", "rb") as f:
    arima_model = pickle.load(f)

# Sidebar settings
st.title("ARIMA Time Series Forecasting")
forecast_steps = st.sidebar.slider("Select Forecast Steps", 1, 60, 30)

# Forecast using ARIMA
forecast = arima_model.forecast(steps=forecast_steps)

# Evaluate Model
test_size = int(len(df) * 0.2)
test = df[-test_size:]
mse_arima = mean_squared_error(test['y'][:forecast_steps], forecast[:test_size])
rmse_arima = np.sqrt(mse_arima)

# Plot Results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index[-100:], df['y'][-100:], label="Actual Data", color="blue")
future_dates = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='D')[1:]
ax.plot(future_dates, forecast, label="ARIMA Forecast", color="green", linestyle="--")
ax.legend()
ax.grid(True)
ax.set_title("ARIMA Model Forecast")
st.pyplot(fig)

# Display RMSE
st.sidebar.subheader("Model Performance")
st.sidebar.write(f"**ARIMA RMSE:** {rmse_arima:.4f}")