import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from data_loader import DataLoader
from futures_data_loader import FuturesDataLoader
from predictor import LSTMPredictor
from futures_predictor import FuturesLSTMPredictor
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
def get_spot_data_loader():
    try:
        return DataLoader()
    except Exception as e:
        st.error(f"{get_text('failed_initialize_dataloader', current_lang)} {str(e)}")
        st.stop()

@st.cache_resource
def get_futures_data_loader():
    try:
        return FuturesDataLoader()
    except Exception as e:
        st.error(f"{get_text('failed_initialize_dataloader', current_lang)} {str(e)}")
        st.stop()

# --- State Management ---
def initialize_state():
    """Initialize session state for filters if they don't exist."""
    if 'filters_initialized' not in st.session_state:
        st.session_state.filters_initialized = True
        # Store language-independent values
        st.session_state.market_type = "spot"
        st.session_state.symbol = "BTCUSDT"
        st.session_state.n_future_steps = 1
        st.session_state.lookback_days = 365
        st.session_state.threshold = 2.0

def reset_filters():
    """Reset all filter values to defaults."""
    st.session_state.market_type = "spot"
    st.session_state.symbol = "BTCUSDT"
    st.session_state.n_future_steps = 1
    st.session_state.lookback_days = 365
    st.session_state.threshold = 2.0
    st.rerun()

# Initialize state at the beginning of the script
initialize_state()

# --- Sidebar Configuration ---
st.sidebar.title(get_text("configuration", current_lang))

# Language switcher
languages = {"English": "en", "Українська": "uk"}
selected_language_label = st.sidebar.selectbox(
    get_text("language", current_lang),
    list(languages.keys()),
    index=0 if current_lang == "en" else 1
)
if languages[selected_language_label] != current_lang:
    set_language(languages[selected_language_label])
    st.rerun()

# Market Type Selection
market_types_map = {
    get_text("spot_market", current_lang): "spot",
    get_text("futures_market", current_lang): "futures"
}
market_display_options = list(market_types_map.keys())
market_internal_values = list(market_types_map.values())

# Find current market index
try:
    current_market_index = market_internal_values.index(st.session_state.market_type)
except (ValueError, KeyError):
    current_market_index = 0
    st.session_state.market_type = "spot"

selected_market_display = st.sidebar.selectbox(
    get_text("select_market_type", current_lang),
    market_display_options,
    index=current_market_index
)
st.session_state.market_type = market_types_map[selected_market_display]

# Initialize appropriate data loader based on market type
data_loader = get_spot_data_loader() if st.session_state.market_type == "spot" else get_futures_data_loader()

# Dynamic Symbol selection based on market type
cache_key = f'available_pairs_{st.session_state.market_type}'
if cache_key not in st.session_state:
    with st.spinner("Fetching available pairs..."):
        if st.session_state.market_type == "spot":
            st.session_state[cache_key] = data_loader.get_available_pairs()
        else:
            st.session_state[cache_key] = data_loader.get_futures_pairs()

available_symbols = st.session_state.get(cache_key, ["BTCUSDT"])

# Ensure stored symbol is valid for current market type
if st.session_state.symbol not in available_symbols:
    st.session_state.symbol = available_symbols[0] if available_symbols else "BTCUSDT"

# Find current symbol index
try:
    current_symbol_index = available_symbols.index(st.session_state.symbol)
except (ValueError, KeyError):
    current_symbol_index = 0
    st.session_state.symbol = available_symbols[0] if available_symbols else "BTCUSDT"

symbol = st.sidebar.selectbox(
    get_text("select_cryptocurrency", current_lang),
    available_symbols,
    index=current_symbol_index
)
st.session_state.symbol = symbol

if st.sidebar.button(get_text("refresh_pairs", current_lang)):
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    st.rerun()

# Prediction Horizon Selection
time_horizons_map = {
    get_text("1_hour", current_lang): 1,
    get_text("4_hours", current_lang): 4,
    get_text("8_hours", current_lang): 8,
    get_text("12_hours", current_lang): 12,
    get_text("1_day", current_lang): 24,
}
horizon_display_options = list(time_horizons_map.keys())
horizon_internal_values = list(time_horizons_map.values())

# Find current horizon index
try:
    current_horizon_index = horizon_internal_values.index(st.session_state.n_future_steps)
except (ValueError, KeyError):
    current_horizon_index = 0
    st.session_state.n_future_steps = 1

selected_horizon_display = st.sidebar.selectbox(
    get_text("select_prediction_horizon", current_lang), 
    horizon_display_options,
    index=current_horizon_index
)
st.session_state.n_future_steps = time_horizons_map[selected_horizon_display]

# Other parameters with state preservation
lookback_days = st.sidebar.slider(
    get_text("historical_data_days", current_lang), 
    min_value=30, max_value=365, 
    value=st.session_state.get('lookback_days', 365)
)
st.session_state.lookback_days = lookback_days

threshold = st.sidebar.slider(
    get_text("signal_threshold", current_lang), 
    min_value=0.1, max_value=10.0, 
    value=st.session_state.get('threshold', 2.0), 
    step=0.1
)
st.session_state.threshold = threshold

# Add a reset button
st.sidebar.button(get_text("reset_filters", current_lang), on_click=reset_filters, use_container_width=True)

# Extract values for use in the app
market_type = st.session_state.market_type
n_future_steps = st.session_state.n_future_steps

# Model instantiation based on market type and horizon
@st.cache_resource
def get_spot_predictor(horizon):
    return LSTMPredictor(n_future_steps=horizon)

@st.cache_resource
def get_futures_predictor(horizon):
    return FuturesLSTMPredictor(n_future_steps=horizon)

if market_type == "spot":
    predictor = get_spot_predictor(n_future_steps)
else:
    predictor = get_futures_predictor(n_future_steps)

# --- Main Content ---
title_key = "futures_prediction_title" if market_type == "futures" else "price_prediction_title"
st.title(f"{symbol} {get_text(title_key, current_lang)} {selected_horizon_display}")

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
    # 1. Fetch and prepare data based on market type
    with st.spinner(get_text("fetching_data", current_lang, selected_horizon_display)):
        if market_type == "spot":
            df = data_loader.fetch_historical_data(symbol=symbol, lookback_days=lookback_days)
            if df.empty:
                st.error(get_text("failed_fetch_data", current_lang))
                st.stop()
            
            df = data_loader.add_technical_indicators(df)
            X_train, y_train, X_test, y_test, original_y_test = data_loader.prepare_data(df, n_future_steps=n_future_steps)
        else:
            df = data_loader.fetch_futures_data(symbol=symbol, lookback_days=lookback_days)
            if df.empty:
                st.error(get_text("failed_fetch_data", current_lang))
                st.stop()
            
            df = data_loader.add_futures_indicators(df)
            X_train, y_train, X_test, y_test, original_y_test = data_loader.prepare_futures_data(df, n_future_steps=n_future_steps)

        if X_train is None:
            st.error(get_text("insufficient_data", current_lang))
            st.stop()

    # 2. Load or train model
    n_features = X_train.shape[2]
    if not predictor.load_model(n_features):
        with st.spinner(get_text("training_model", current_lang, selected_horizon_display)):
            predictor.train(X_train, y_train)
    
    # 3. Generate and process predictions
    with st.spinner(get_text("generating_predictions", current_lang)):
        last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
        predicted_scaled_sequence = predictor.predict(last_sequence)[0]
        predicted_prices = data_loader.inverse_transform_price(predicted_scaled_sequence)
    
    current_price = df['close'].iloc[-1]
    
    # --- Display Metrics and Signals ---
    signal, signal_reason = generate_trading_signal(current_price, predicted_prices, threshold, current_lang)
    
    # Different layouts for spot vs futures
    if market_type == "futures":
        st.header(get_text("futures_metrics", current_lang))
        
        # Calculate futures-specific metrics
        futures_metrics = data_loader.calculate_futures_metrics(df, current_price, predicted_prices)
        
        # Main forecast metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Current Price", f"${current_price:,.4f}")
        
        predicted_final = predicted_prices[-1]
        change_final = (predicted_final - current_price) / current_price * 100
        col2.metric(f"🎯 Exact Forecast ({selected_horizon_display})", 
                   f"${predicted_final:,.4f}", 
                   f"{change_final:+.2f}%")
        
        predicted_avg = np.mean(predicted_prices)
        change_avg = (predicted_avg - current_price) / current_price * 100
        col3.metric(f"📊 Average Price ({selected_horizon_display})", 
                   f"${predicted_avg:,.4f}", 
                   f"{change_avg:+.2f}%")
        
        # Detailed step-by-step forecast
        st.subheader("📈 Step-by-Step Forecast")
        
        if n_future_steps > 1:
            forecast_df = pd.DataFrame({
                'Hour': [f"Hour {i+1}" for i in range(n_future_steps)],
                'Predicted Price': [f"${price:,.4f}" for price in predicted_prices],
                'Change from Current': [f"{((price - current_price) / current_price * 100):+.2f}%" for price in predicted_prices],
                'Price Movement': [
                    "📈 UP" if price > current_price else "📉 DOWN" if price < current_price else "➡️ FLAT"
                    for price in predicted_prices
                ]
            })
            
            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"🎯 **Exact Forecast for next {selected_horizon_display}:** ${predicted_final:,.4f} ({change_final:+.2f}%)")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        predicted_low = np.min(predicted_prices)
        change_low = (predicted_low - current_price) / current_price * 100
        col1.metric("📉 Minimum Expected", f"${predicted_low:,.4f}", f"{change_low:+.2f}%")
        
        predicted_high = np.max(predicted_prices)
        change_high = (predicted_high - current_price) / current_price * 100
        col2.metric("📈 Maximum Expected", f"${predicted_high:,.4f}", f"{change_high:+.2f}%")
        
        price_range = predicted_high - predicted_low
        range_pct = (price_range / current_price) * 100
        col3.metric("📏 Price Range", f"${price_range:,.4f}", f"{range_pct:.2f}%")
        
        volatility = np.std(predicted_prices)
        vol_pct = (volatility / current_price) * 100
        col4.metric("⚡ Volatility", f"${volatility:,.4f}", f"{vol_pct:.2f}%")
        
        # Additional futures metrics
        if 'risk_reward_ratio' in futures_metrics:
            st.subheader("🎯 Futures-Specific Metrics")
            col5, col6, col7, col8 = st.columns(4)
            col5.metric(get_text("risk_reward_ratio", current_lang), f"{futures_metrics['risk_reward_ratio']:.2f}")
            
            if 'atr' in futures_metrics:
                col6.metric(get_text("atr", current_lang), f"${futures_metrics['atr']:.2f}")
            
            if 'volatility' in futures_metrics:
                col7.metric(get_text("volatility", current_lang), f"{futures_metrics['volatility']:.4f}")
            
            if 'volume_ratio' in futures_metrics:
                col8.metric(get_text("volume_ratio", current_lang), f"{futures_metrics['volume_ratio']:.2f}")
    else:
        st.header(get_text("prediction_summary", current_lang))
        
        # Main forecast metrics for spot
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Current Price", f"${current_price:,.4f}")
        
        predicted_final = predicted_prices[-1]
        change_final = (predicted_final - current_price) / current_price * 100
        col2.metric(f"🎯 Exact Forecast ({selected_horizon_display})", 
                   f"${predicted_final:,.4f}", 
                   f"{change_final:+.2f}%")
        
        predicted_avg = np.mean(predicted_prices)
        change_avg = (predicted_avg - current_price) / current_price * 100
        col3.metric(f"📊 Average Price ({selected_horizon_display})", 
                   f"${predicted_avg:,.4f}", 
                   f"{change_avg:+.2f}%")
        
        # Detailed step-by-step forecast
        st.subheader("📈 Step-by-Step Forecast")
        
        if n_future_steps > 1:
            forecast_df = pd.DataFrame({
                'Hour': [f"Hour {i+1}" for i in range(n_future_steps)],
                'Predicted Price': [f"${price:,.4f}" for price in predicted_prices],
                'Change from Current': [f"{((price - current_price) / current_price * 100):+.2f}%" for price in predicted_prices],
                'Price Movement': [
                    "📈 UP" if price > current_price else "📉 DOWN" if price < current_price else "➡️ FLAT"
                    for price in predicted_prices
                ]
            })
            
            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"🎯 **Exact Forecast for next {selected_horizon_display}:** ${predicted_final:,.4f} ({change_final:+.2f}%)")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        predicted_low = np.min(predicted_prices)
        change_low = (predicted_low - current_price) / current_price * 100
        col1.metric("📉 Minimum Expected", f"${predicted_low:,.4f}", f"{change_low:+.2f}%")
        
        predicted_high = np.max(predicted_prices)
        change_high = (predicted_high - current_price) / current_price * 100
        col2.metric("📈 Maximum Expected", f"${predicted_high:,.4f}", f"{change_high:+.2f}%")
        
        price_range = predicted_high - predicted_low
        range_pct = (price_range / current_price) * 100
        col3.metric("📏 Price Range", f"${price_range:,.4f}", f"{range_pct:.2f}%")
        
        volatility = np.std(predicted_prices)
        vol_pct = (volatility / current_price) * 100
        col4.metric("⚡ Volatility", f"${volatility:,.4f}", f"{vol_pct:.2f}%")
    
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

    # Add additional indicators for futures
    if market_type == "futures" and 'bb_upper' in df.columns:
        # Add Bollinger Bands for futures
        fig.add_trace(go.Scatter(
            x=df.index[-100:], y=df['bb_upper'].iloc[-100:], 
            name="Bollinger Upper", line=dict(color='gray', dash='dot'), opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=df.index[-100:], y=df['bb_lower'].iloc[-100:], 
            name="Bollinger Lower", line=dict(color='gray', dash='dot'), opacity=0.5,
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
        ))

    chart_title = f"{symbol} {get_text('price_forecast_for', current_lang)} {selected_horizon_display}"
    if market_type == "futures":
        chart_title += " (Futures)"
    
    fig.update_layout(
        title=chart_title,
        xaxis_title=get_text("date", current_lang),
        yaxis_title=get_text("price_usdt", current_lang),
        template='plotly_dark',
        legend=dict(x=0, y=1, traceorder="normal")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Additional technical indicator charts
    recent_data = df.iloc[-100:]  # Last 100 data points
    
    # MACD Indicator
    if 'macd' in df.columns:
        st.subheader("📊 MACD Indicator")
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['macd'], 
            name="MACD", line=dict(color='blue')
        ))
        fig_macd.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['macd_signal'], 
            name="Signal", line=dict(color='red')
        ))
        fig_macd.add_trace(go.Bar(
            x=recent_data.index, y=recent_data['macd_histogram'], 
            name="Histogram", opacity=0.6
        ))
        
        fig_macd.update_layout(
            title="MACD Analysis",
            xaxis_title=get_text("date", current_lang),
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Stochastic Oscillator
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        st.subheader("📈 Stochastic Oscillator")
        fig_stoch = go.Figure()
        
        fig_stoch.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['stoch_k'], 
            name="%K (Fast)", line=dict(color='blue', width=2)
        ))
        fig_stoch.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['stoch_d'], 
            name="%D (Slow)", line=dict(color='red', width=2)
        ))
        
        # Add overbought/oversold levels
        fig_stoch.add_hline(y=80, line_dash="dash", line_color="gray", 
                           annotation_text="Overbought (80)")
        fig_stoch.add_hline(y=20, line_dash="dash", line_color="gray", 
                           annotation_text="Oversold (20)")
        fig_stoch.add_hline(y=50, line_dash="dot", line_color="lightgray", 
                           annotation_text="Midline (50)", opacity=0.5)
        
        # Fill areas for overbought/oversold zones
        fig_stoch.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.1, 
                           annotation_text="Overbought Zone", annotation_position="top left")
        fig_stoch.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.1, 
                           annotation_text="Oversold Zone", annotation_position="bottom left")
        
        fig_stoch.update_layout(
            title="Stochastic Oscillator (%K and %D)",
            xaxis_title=get_text("date", current_lang),
            yaxis_title="Value (%)",
            template='plotly_dark',
            height=400,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_stoch, use_container_width=True)
        
        # Add interpretation
        latest_k = recent_data['stoch_k'].iloc[-1]
        latest_d = recent_data['stoch_d'].iloc[-1]
        
        if latest_k > 80 and latest_d > 80:
            st.warning("⚠️ **Signal**: Asset in overbought zone. Possible price decline.")
        elif latest_k < 20 and latest_d < 20:
            st.success("💡 **Signal**: Asset in oversold zone. Possible price bounce.")
        elif latest_k > latest_d and latest_k > 50:
            st.info("📈 **Signal**: %K above %D and above 50 - bullish signal.")
        elif latest_k < latest_d and latest_k < 50:
            st.info("📉 **Signal**: %K below %D and below 50 - bearish signal.")
        else:
            st.info("➡️ **Status**: Neutral stochastic oscillator readings.")
        
        st.caption(f"Current values: %K = {latest_k:.2f}, %D = {latest_d:.2f}")

except Exception as e:
    logging.error(f"Application error: {str(e)}")
    st.error(f"{get_text('error_occurred', current_lang)} {str(e)}")
    st.error(get_text("try_refresh", current_lang)) 