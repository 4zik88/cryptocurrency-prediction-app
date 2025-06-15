import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from data_loader import DataLoader
from predictor import LSTMPredictor
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Page config
st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def get_data_loader():
    try:
        return DataLoader()
    except Exception as e:
        st.error(f"Failed to initialize DataLoader: {str(e)}")
        st.stop()

st.session_state.data_loader = get_data_loader()

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")

# Dynamic Symbol selection
if 'available_pairs' not in st.session_state:
    st.session_state.available_pairs = st.session_state.data_loader.get_available_pairs()

symbol = st.sidebar.selectbox(
    "Select Cryptocurrency",
    st.session_state.available_pairs if st.session_state.available_pairs else ["BTCUSDT"]
)

if st.sidebar.button("🔄 Refresh Pairs"):
    st.session_state.available_pairs = st.session_state.data_loader.get_available_pairs()
    st.rerun()

# Prediction Horizon Selection
time_horizons = {
    "1 Hour": 1,
    "4 Hours": 4,
    "8 Hours": 8,
    "1 Day (24H)": 24,
}
selected_horizon_label = st.sidebar.selectbox("Select Prediction Horizon", list(time_horizons.keys()))
n_future_steps = time_horizons[selected_horizon_label]

# Model instantiation based on horizon
@st.cache_resource
def get_predictor(horizon):
    return LSTMPredictor(n_future_steps=horizon)

predictor = get_predictor(n_future_steps)

# Other parameters
lookback_days = st.sidebar.slider("Historical Data (days)", min_value=30, max_value=365, value=180)
threshold = st.sidebar.slider("Signal Threshold (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

# --- Main Content ---
st.title(f"{symbol} Price Prediction for the Next {selected_horizon_label}")

def generate_trading_signal(current_price, predicted_sequence, threshold):
    """Generate Long/Short/Hold signal based on the predicted price trajectory."""
    max_predicted_price = np.max(predicted_sequence)
    min_predicted_price = np.min(predicted_sequence)
    
    upside_potential = (max_predicted_price - current_price) / current_price * 100
    downside_risk = (min_predicted_price - current_price) / current_price * 100

    if upside_potential > threshold:
        return "Long", f"Predicted peak gain of {upside_potential:.2f}%"
    elif downside_risk < -threshold:
        return "Short", f"Predicted potential drop of {abs(downside_risk):.2f}%"
    else:
        return "Hold", "No significant price movement predicted."

try:
    # 1. Fetch and prepare data
    with st.spinner(f"Fetching data and preparing for {selected_horizon_label} forecast..."):
        df = st.session_state.data_loader.fetch_historical_data(symbol=symbol, lookback_days=lookback_days)
        if df.empty:
            st.error("Failed to fetch data. Please check symbol or network connection.")
            st.stop()
        
        df = st.session_state.data_loader.add_technical_indicators(df)
        X_train, y_train, X_test, y_test, original_y_test = st.session_state.data_loader.prepare_data(df, n_future_steps=n_future_steps)

        if X_train is None:
            st.error("Insufficient data for prediction. Try increasing the historical data period.")
            st.stop()

    # 2. Load or train model
    if not predictor.load_model():
        with st.spinner(f"Training model for {selected_horizon_label} horizon... This might take a moment."):
            predictor.train(X_train, y_train)
    
    # 3. Generate and process predictions
    with st.spinner("Generating predictions..."):
        last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
        predicted_scaled_sequence = predictor.predict(last_sequence)[0]
        predicted_prices = st.session_state.data_loader.inverse_transform_price(predicted_scaled_sequence)
    
    current_price = df['close'].iloc[-1]
    
    # --- Display Metrics and Signals ---
    signal, signal_reason = generate_trading_signal(current_price, predicted_prices, threshold)
    
    st.header("Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:,.2f}")
    col2.metric(f"Predicted Low ({selected_horizon_label})", f"${np.min(predicted_prices):,.2f}", delta_color="inverse")
    col3.metric(f"Predicted High ({selected_horizon_label})", f"${np.max(predicted_prices):,.2f}")
    
    st.subheader("Trading Signal")
    if signal == "Long":
        st.success(f"**Go Long** 🟢")
    elif signal == "Short":
        st.error(f"**Go Short** 🔴")
    else:
        st.info(f"**Hold/Neutral** ⚪")
    st.caption(signal_reason)

    # --- Plotting ---
    st.header("Price Forecast Chart")
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'], name="Historical Price", line=dict(color='royalblue')
    ))

    # Predicted sequence
    future_dates = pd.to_datetime([df.index[-1] + timedelta(hours=i+1) for i in range(n_future_steps)])
    fig.add_trace(go.Scatter(
        x=future_dates, y=predicted_prices, name="Predicted Price Path", line=dict(color='tomato', dash='dash')
    ))

    fig.update_layout(
        title=f"{symbol} Price Forecast for the Next {selected_horizon_label}",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        template='plotly_dark',
        legend=dict(x=0, y=1, traceorder="normal")
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    logging.error(f"Application error: {str(e)}")
    st.error(f"An error occurred: {str(e)}")
    st.error("Please try refreshing the page or selecting different parameters.") 