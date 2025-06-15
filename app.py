import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from data_loader import DataLoader
from predictor import LSTMPredictor
from translations import get_text, init_language, get_current_language, set_language
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize language
init_language()
current_lang = get_current_language()

# Page config
st.set_page_config(
    page_title=get_text("page_title", current_lang),
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def get_data_loader():
    try:
        return DataLoader()
    except Exception as e:
        st.error(f"{get_text('failed_initialize_dataloader', current_lang)} {str(e)}")
        st.stop()

st.session_state.data_loader = get_data_loader()

# --- Sidebar Configuration ---
st.sidebar.title(get_text("configuration", current_lang))

# Language switcher
languages = {"English": "en", "Українська": "uk"}
selected_language = st.sidebar.selectbox(
    get_text("language", current_lang),
    list(languages.keys()),
    index=0 if current_lang == "en" else 1
)

if languages[selected_language] != current_lang:
    set_language(languages[selected_language])
    st.rerun()

# Update current language after potential change
current_lang = get_current_language()

# Dynamic Symbol selection
if 'available_pairs' not in st.session_state:
    st.session_state.available_pairs = st.session_state.data_loader.get_available_pairs()

symbol = st.sidebar.selectbox(
    get_text("select_cryptocurrency", current_lang),
    st.session_state.available_pairs if st.session_state.available_pairs else ["BTCUSDT"]
)

if st.sidebar.button(get_text("refresh_pairs", current_lang)):
    st.session_state.available_pairs = st.session_state.data_loader.get_available_pairs()
    st.rerun()

# Prediction Horizon Selection
time_horizons = {
    get_text("1_hour", current_lang): 1,
    get_text("4_hours", current_lang): 4,
    get_text("8_hours", current_lang): 8,
    get_text("1_day", current_lang): 24,
}
selected_horizon_label = st.sidebar.selectbox(get_text("select_prediction_horizon", current_lang), list(time_horizons.keys()))
n_future_steps = time_horizons[selected_horizon_label]

# Model instantiation based on horizon
@st.cache_resource
def get_predictor(horizon):
    return LSTMPredictor(n_future_steps=horizon)

predictor = get_predictor(n_future_steps)

# Other parameters
lookback_days = st.sidebar.slider(get_text("historical_data_days", current_lang), min_value=30, max_value=365, value=180)
threshold = st.sidebar.slider(get_text("signal_threshold", current_lang), min_value=0.1, max_value=10.0, value=2.0, step=0.1)

# --- Main Content ---
st.title(f"{symbol} {get_text('price_prediction_title', current_lang)} {selected_horizon_label}")

def generate_trading_signal(current_price, predicted_sequence, threshold, lang):
    """Generate Long/Short/Hold signal based on the predicted price trajectory."""
    max_predicted_price = np.max(predicted_sequence)
    min_predicted_price = np.min(predicted_sequence)
    
    upside_potential = (max_predicted_price - current_price) / current_price * 100
    downside_risk = (min_predicted_price - current_price) / current_price * 100

    if upside_potential > threshold:
        return "Long", f"{get_text('predicted_peak_gain', lang)} {upside_potential:.2f}%"
    elif downside_risk < -threshold:
        return "Short", f"{get_text('predicted_potential_drop', lang)} {abs(downside_risk):.2f}%"
    else:
        return "Hold", get_text("no_significant_movement", lang)

try:
    # 1. Fetch and prepare data
    with st.spinner(get_text("fetching_data", current_lang, selected_horizon_label)):
        df = st.session_state.data_loader.fetch_historical_data(symbol=symbol, lookback_days=lookback_days)
        if df.empty:
            st.error(get_text("failed_fetch_data", current_lang))
            st.stop()
        
        df = st.session_state.data_loader.add_technical_indicators(df)
        X_train, y_train, X_test, y_test, original_y_test = st.session_state.data_loader.prepare_data(df, n_future_steps=n_future_steps)

        if X_train is None:
            st.error(get_text("insufficient_data", current_lang))
            st.stop()

    # 2. Load or train model
    if not predictor.load_model():
        with st.spinner(get_text("training_model", current_lang, selected_horizon_label)):
            predictor.train(X_train, y_train)
    
    # 3. Generate and process predictions
    with st.spinner(get_text("generating_predictions", current_lang)):
        last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
        predicted_scaled_sequence = predictor.predict(last_sequence)[0]
        predicted_prices = st.session_state.data_loader.inverse_transform_price(predicted_scaled_sequence)
    
    current_price = df['close'].iloc[-1]
    
    # --- Display Metrics and Signals ---
    signal, signal_reason = generate_trading_signal(current_price, predicted_prices, threshold, current_lang)
    
    st.header(get_text("prediction_summary", current_lang))
    col1, col2, col3 = st.columns(3)
    col1.metric(get_text("current_price", current_lang), f"${current_price:,.2f}")
    col2.metric(f"{get_text('predicted_low', current_lang)} ({selected_horizon_label})", f"${np.min(predicted_prices):,.2f}", delta_color="inverse")
    col3.metric(f"{get_text('predicted_high', current_lang)} ({selected_horizon_label})", f"${np.max(predicted_prices):,.2f}")
    
    st.subheader(get_text("trading_signal", current_lang))
    if signal == "Long":
        st.success(get_text("go_long", current_lang))
    elif signal == "Short":
        st.error(get_text("go_short", current_lang))
    else:
        st.info(get_text("hold_neutral", current_lang))
    st.caption(signal_reason)

    # --- Plotting ---
    st.header(get_text("price_forecast_chart", current_lang))
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'], name=get_text("historical_price", current_lang), line=dict(color='royalblue')
    ))

    # Predicted sequence
    future_dates = pd.to_datetime([df.index[-1] + timedelta(hours=i+1) for i in range(n_future_steps)])
    fig.add_trace(go.Scatter(
        x=future_dates, y=predicted_prices, name=get_text("predicted_price_path", current_lang), line=dict(color='tomato', dash='dash')
    ))

    fig.update_layout(
        title=f"{symbol} {get_text('price_forecast_for', current_lang)} {selected_horizon_label}",
        xaxis_title=get_text("date", current_lang),
        yaxis_title=get_text("price_usdt", current_lang),
        template='plotly_dark',
        legend=dict(x=0, y=1, traceorder="normal")
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    logging.error(f"Application error: {str(e)}")
    st.error(f"{get_text('error_occurred', current_lang)} {str(e)}")
    st.error(get_text("try_refresh", current_lang)) 