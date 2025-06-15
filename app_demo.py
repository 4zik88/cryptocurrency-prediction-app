import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import numpy as np
from translations import get_text, init_language, get_current_language, set_language

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

# Demo mode warning
st.warning("🚧 **DEMO MODE** - This is a demonstration version without live API data. The full version with real-time predictions requires TensorFlow and API credentials.")

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

# Market Type Selection
market_types = {
    get_text("spot_market", current_lang): "spot",
    get_text("futures_market", current_lang): "futures"
}
selected_market_type = st.sidebar.selectbox(
    get_text("select_market_type", current_lang),
    list(market_types.keys())
)
market_type = market_types[selected_market_type]

# Demo cryptocurrency selection
demo_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
symbol = st.sidebar.selectbox(
    get_text("select_cryptocurrency", current_lang),
    demo_symbols
)

# Prediction Horizon Selection
time_horizons = {
    get_text("1_hour", current_lang): 1,
    get_text("4_hours", current_lang): 4,
    get_text("8_hours", current_lang): 8,
    get_text("1_day", current_lang): 24,
}
selected_horizon_label = st.sidebar.selectbox(get_text("select_prediction_horizon", current_lang), list(time_horizons.keys()))
n_future_steps = time_horizons[selected_horizon_label]

# Other parameters
lookback_days = st.sidebar.slider(get_text("historical_data_days", current_lang), min_value=30, max_value=365, value=180)
threshold = st.sidebar.slider(get_text("signal_threshold", current_lang), min_value=0.1, max_value=10.0, value=2.0, step=0.1)

# --- Main Content ---
title_key = "futures_prediction_title" if market_type == "futures" else "price_prediction_title"
st.title(f"{symbol} {get_text(title_key, current_lang)} {selected_horizon_label}")

# Generate demo data
@st.cache_data
def generate_demo_data(symbol, days=180):
    """Generate realistic demo cryptocurrency data."""
    np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
    
    # Base prices for different cryptocurrencies
    base_prices = {
        "BTCUSDT": 45000,
        "ETHUSDT": 2800,
        "SOLUSDT": 95,
        "BNBUSDT": 320,
        "ADAUSDT": 0.45
    }
    
    base_price = base_prices.get(symbol, 1000)
    
    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='H')
    
    # Generate realistic price movement
    returns = np.random.normal(0, 0.02, days)  # 2% volatility
    returns[0] = 0
    
    # Add some trend and cycles
    trend = np.linspace(-0.1, 0.1, days)
    cycle = 0.05 * np.sin(np.linspace(0, 4*np.pi, days))
    
    cumulative_returns = np.cumsum(returns + trend + cycle)
    prices = base_price * np.exp(cumulative_returns)
    
    # Generate OHLCV data
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, days))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, days))
    df['volume'] = np.random.uniform(1000000, 5000000, days)
    
    return df

def add_demo_indicators(df, market_type):
    """Add technical indicators to demo data."""
    # Basic indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = 50 + 30 * np.sin(np.linspace(0, 10*np.pi, len(df)))  # Demo RSI
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['price_change'].rolling(20).std()
    
    if market_type == "futures":
        # Additional futures indicators
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std_dev = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

def generate_demo_predictions(current_price, n_steps, market_type):
    """Generate realistic demo predictions."""
    np.random.seed(42)  # Consistent predictions
    
    # Generate prediction path
    volatility = 0.01 if market_type == "spot" else 0.015
    returns = np.random.normal(0, volatility, n_steps)
    
    # Add slight upward bias for demo
    trend = np.linspace(0, 0.02, n_steps)
    
    cumulative_returns = np.cumsum(returns + trend)
    predictions = current_price * np.exp(cumulative_returns)
    
    return predictions

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

# Generate demo data
with st.spinner("Generating demo data..."):
    df = generate_demo_data(symbol, lookback_days * 24)  # Convert days to hours
    df = add_demo_indicators(df, market_type)
    
    current_price = df['close'].iloc[-1]
    predicted_prices = generate_demo_predictions(current_price, n_future_steps, market_type)

# --- Display Metrics and Signals ---
signal, signal_reason = generate_trading_signal(current_price, predicted_prices, threshold, current_lang)

# Different layouts for spot vs futures
if market_type == "futures":
    st.header(get_text("futures_metrics", current_lang))
    
    # Calculate demo futures metrics
    futures_metrics = {
        'current_price': current_price,
        'predicted_min': np.min(predicted_prices),
        'predicted_max': np.max(predicted_prices),
        'risk_reward_ratio': 1.5,  # Demo value
        'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 100,
        'volatility': df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.02,
        'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.2
    }
    
    # Display futures metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(get_text("current_price", current_lang), f"${current_price:,.2f}")
    col2.metric(f"{get_text('predicted_low', current_lang)} ({selected_horizon_label})", f"${np.min(predicted_prices):,.2f}", delta_color="inverse")
    col3.metric(f"{get_text('predicted_high', current_lang)} ({selected_horizon_label})", f"${np.max(predicted_prices):,.2f}")
    col4.metric(get_text("risk_reward_ratio", current_lang), f"{futures_metrics['risk_reward_ratio']:.2f}")
    
    # Second row of metrics for futures
    col5, col6, col7, col8 = st.columns(4)
    col5.metric(get_text("atr", current_lang), f"${futures_metrics['atr']:.2f}")
    col6.metric(get_text("volatility", current_lang), f"{futures_metrics['volatility']:.4f}")
    col7.metric(get_text("volume_ratio", current_lang), f"{futures_metrics['volume_ratio']:.2f}")
else:
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

# Historical data (last 100 points for better visualization)
recent_df = df.iloc[-100:]
fig.add_trace(go.Scatter(
    x=recent_df.index, y=recent_df['close'], 
    name=get_text("historical_price", current_lang), 
    line=dict(color='royalblue')
))

# Predicted sequence
future_dates = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=n_future_steps, freq='H')
fig.add_trace(go.Scatter(
    x=future_dates, y=predicted_prices, 
    name=get_text("predicted_price_path", current_lang), 
    line=dict(color='tomato', dash='dash')
))

# Add additional indicators for futures
if market_type == "futures" and 'bb_upper' in df.columns:
    # Add Bollinger Bands for futures
    fig.add_trace(go.Scatter(
        x=recent_df.index, y=recent_df['bb_upper'], 
        name="Bollinger Upper", line=dict(color='gray', dash='dot'), opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=recent_df.index, y=recent_df['bb_lower'], 
        name="Bollinger Lower", line=dict(color='gray', dash='dot'), opacity=0.5,
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
    ))

chart_title = f"{symbol} {get_text('price_forecast_for', current_lang)} {selected_horizon_label}"
if market_type == "futures":
    chart_title += " (Futures Demo)"
else:
    chart_title += " (Spot Demo)"

fig.update_layout(
    title=chart_title,
    xaxis_title=get_text("date", current_lang),
    yaxis_title=get_text("price_usdt", current_lang),
    template='plotly_dark',
    legend=dict(x=0, y=1, traceorder="normal")
)
st.plotly_chart(fig, use_container_width=True)

# Additional futures-specific charts
if market_type == "futures" and 'macd' in df.columns:
    st.subheader("MACD Indicator (Futures Demo)")
    fig_macd = go.Figure()
    
    recent_data = df.iloc[-100:]  # Last 100 data points
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
        title="MACD Analysis (Demo)",
        xaxis_title=get_text("date", current_lang),
        template='plotly_dark'
    )
    st.plotly_chart(fig_macd, use_container_width=True)

# Information about the full version
st.info("""
🚀 **Want the full version with real-time data?**

The complete application includes:
- Real-time data from Bybit API
- Advanced LSTM neural network predictions
- Live market analysis
- Actual trading signals

To deploy the full version, you'll need:
1. Bybit API credentials
2. A server with TensorFlow support
3. Use the Docker deployment option from the repository
""")

# Footer
st.markdown("---")
st.markdown("📊 **Demo Version** - Built with Streamlit | 🌐 **Multi-language Support** | 🔗 **Open Source**") 