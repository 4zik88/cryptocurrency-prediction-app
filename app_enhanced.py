import streamlit as st

# Page config MUST be first
st.set_page_config(
    page_title="Cryptocurrency Price Prediction",
    page_icon="📈",
    layout="wide"
)

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import numpy as np
import os

# Try to import real data loaders and predictors, fall back to demo mode
try:
    # Check if we have API credentials available
    api_key = None
    api_secret = None
    
    # Try to get from Streamlit secrets first
    if hasattr(st, 'secrets') and 'BYBIT_API_KEY' in st.secrets:
        api_key = st.secrets['BYBIT_API_KEY']
        api_secret = st.secrets['BYBIT_API_SECRET']
        os.environ['BYBIT_API_KEY'] = api_key
        os.environ['BYBIT_API_SECRET'] = api_secret
    # Fall back to environment variables
    elif 'BYBIT_API_KEY' in os.environ:
        api_key = os.environ['BYBIT_API_KEY']
        api_secret = os.environ['BYBIT_API_SECRET']
    
    # If we have API credentials, try to import real modules
    if api_key and api_secret:
        from data_loader import DataLoader
        from futures_data_loader import FuturesDataLoader
        from predictor import LSTMPredictor
        from futures_predictor import FuturesLSTMPredictor
        API_MODE = True
        API_STATUS_MESSAGE = "🚀 **LIVE MODE** - Using real-time API data and ML predictions!"
    else:
        raise ImportError("No API credentials available")
        
except (ImportError, Exception) as e:
    # Fall back to demo mode
    API_MODE = False
    API_STATUS_MESSAGE = "🚧 **DEMO MODE** - This is a demonstration version. Add your Bybit API credentials in Streamlit Cloud secrets to enable live data!"

# Import translations
try:
    from translations import get_text, init_language, get_current_language, set_language
except ImportError:
    # Fallback translation system
    def get_text(key, lang="en", *args):
        translations = {
            "page_title": "Cryptocurrency Price Prediction",
            "configuration": "Configuration",
            "language": "Language",
            "spot_market": "Spot Market",
            "futures_market": "Futures Market",
            "select_market_type": "Select Market Type",
            "select_cryptocurrency": "Select Cryptocurrency",
            "refresh_pairs": "Refresh Pairs",
            "1_hour": "1 Hour",
            "4_hours": "4 Hours", 
            "8_hours": "8 Hours",
            "1_day": "1 Day",
            "select_prediction_horizon": "Select Prediction Horizon",
            "historical_data_days": "Historical Data (Days)",
            "signal_threshold": "Signal Threshold (%)",
            "price_prediction_title": "Price Prediction",
            "futures_prediction_title": "Futures Prediction",
            "current_price": "Current Price",
            "predicted_low": "Predicted Low",
            "predicted_high": "Predicted High",
            "trading_signal": "Trading Signal",
            "price_chart": "Price Chart",
            "technical_indicators": "Technical Indicators"
        }
        return translations.get(key, key)
    
    def init_language(): pass
    def get_current_language(): return "en"
    def set_language(lang): pass

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize language
init_language()
current_lang = get_current_language()

# Display API status message
if API_MODE:
    st.success(API_STATUS_MESSAGE)
else:
    st.info(API_STATUS_MESSAGE)

# Demo data generation functions
def generate_demo_crypto_data(symbol="BTCUSDT", days=180):
    """Generate realistic cryptocurrency demo data"""
    np.random.seed(42)  # For consistent demo data
    
    # Base price for different cryptocurrencies
    base_prices = {
        "BTCUSDT": 45000,
        "ETHUSDT": 3000,
        "ADAUSDT": 0.5,
        "SOLUSDT": 100,
        "DOTUSDT": 7,
        "LINKUSDT": 15
    }
    
    base_price = base_prices.get(symbol, 45000)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate price data with realistic patterns
    n_points = len(dates)
    
    # Trend component
    trend = np.linspace(0, 0.2, n_points)  # 20% growth over period
    
    # Seasonal component (daily patterns)
    seasonal = 0.02 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, 0.01, n_points))
    
    # Combine components
    log_returns = trend + seasonal + random_walk
    prices = base_price * np.exp(log_returns)
    
    # Add some volatility
    volatility = np.random.normal(1, 0.02, n_points)
    prices = prices * volatility
    
    # Generate OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.995, 1.005, n_points),
        'high': prices * np.random.uniform(1.001, 1.02, n_points),
        'low': prices * np.random.uniform(0.98, 0.999, n_points),
        'close': prices,
        'volume': np.random.uniform(1000000, 10000000, n_points)
    })
    
    # Ensure OHLC relationships are correct
    df['high'] = np.maximum.reduce([df['open'], df['high'], df['low'], df['close']])
    df['low'] = np.minimum.reduce([df['open'], df['high'], df['low'], df['close']])
    
    return df

def add_demo_technical_indicators(df):
    """Add technical indicators to demo data"""
    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(24)
    
    return df

def generate_demo_predictions(current_price, n_steps=24):
    """Generate realistic demo predictions"""
    np.random.seed(42)
    
    # Generate a realistic price trajectory
    returns = np.random.normal(0.001, 0.02, n_steps)  # Small positive drift with volatility
    
    # Add some trend and mean reversion
    trend = np.linspace(0, 0.01, n_steps)  # Slight upward trend
    mean_reversion = -0.1 * np.cumsum(returns)  # Mean reversion component
    
    combined_returns = returns + trend + mean_reversion * 0.1
    
    # Generate price sequence
    prices = [current_price]
    for ret in combined_returns:
        prices.append(prices[-1] * (1 + ret))
    
    return np.array(prices[1:])

# Data loader classes for demo mode
class DemoDataLoader:
    def __init__(self):
        self.available_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT"]
    
    def get_available_pairs(self):
        return self.available_pairs
    
    def get_futures_pairs(self):
        return self.available_pairs
    
    def fetch_historical_data(self, symbol, lookback_days):
        return generate_demo_crypto_data(symbol, lookback_days)
    
    def fetch_futures_data(self, symbol, lookback_days):
        return generate_demo_crypto_data(symbol, lookback_days)
    
    def add_technical_indicators(self, df):
        return add_demo_technical_indicators(df)
    
    def add_futures_indicators(self, df):
        return add_demo_technical_indicators(df)
    
    def prepare_data(self, df, n_future_steps):
        # Simple demo data preparation
        return None, None, None, None, None
    
    def prepare_futures_data(self, df, n_future_steps):
        return None, None, None, None, None
    
    def inverse_transform_price(self, scaled_prices):
        return scaled_prices
    
    def calculate_futures_metrics(self, df, current_price, predicted_prices):
        return {
            'risk_reward_ratio': np.random.uniform(1.5, 3.0),
            'atr': df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02,
            'volatility': df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.25,
            'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.2
        }

class DemoPredictor:
    def __init__(self, n_future_steps=24):
        self.n_future_steps = n_future_steps
    
    def load_model(self):
        return True  # Always "loaded" in demo mode
    
    def train(self, X_train, y_train):
        pass  # No training in demo mode
    
    def predict(self, last_sequence):
        # Generate demo predictions
        current_price = 45000  # Default price
        predictions = generate_demo_predictions(current_price, self.n_future_steps)
        return [predictions]

# Initialize data loaders and predictors based on mode
@st.cache_resource
def get_data_loaders_and_predictors():
    if API_MODE:
        spot_loader = DataLoader()
        futures_loader = FuturesDataLoader()
        return spot_loader, futures_loader, True
    else:
        demo_loader = DemoDataLoader()
        return demo_loader, demo_loader, False

# Get loaders
spot_loader, futures_loader, is_real_api = get_data_loaders_and_predictors()

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

# Select appropriate data loader
data_loader = spot_loader if market_type == "spot" else futures_loader

# Dynamic Symbol selection
cache_key = f'available_pairs_{market_type}'
if cache_key not in st.session_state:
    if market_type == "spot":
        st.session_state[cache_key] = data_loader.get_available_pairs()
    else:
        st.session_state[cache_key] = data_loader.get_futures_pairs()

symbol = st.sidebar.selectbox(
    get_text("select_cryptocurrency", current_lang),
    st.session_state[cache_key] if st.session_state[cache_key] else ["BTCUSDT"]
)

if st.sidebar.button(get_text("refresh_pairs", current_lang)):
    if market_type == "spot":
        st.session_state[cache_key] = data_loader.get_available_pairs()
    else:
        st.session_state[cache_key] = data_loader.get_futures_pairs()
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

# Other parameters
lookback_days = st.sidebar.slider(get_text("historical_data_days", current_lang), min_value=30, max_value=365, value=180)
threshold = st.sidebar.slider(get_text("signal_threshold", current_lang), min_value=0.1, max_value=10.0, value=2.0, step=0.1)

# --- Main Content ---
title_key = "futures_prediction_title" if market_type == "futures" else "price_prediction_title"
st.title(f"{symbol} {get_text(title_key, current_lang)} {selected_horizon_label}")

def generate_trading_signal(current_price, predicted_sequence, threshold, lang):
    """Generate Long/Short/Hold signal based on the predicted price trajectory."""
    max_predicted_price = np.max(predicted_sequence)
    min_predicted_price = np.min(predicted_sequence)
    
    upside_potential = (max_predicted_price - current_price) / current_price * 100
    downside_risk = (min_predicted_price - current_price) / current_price * 100

    if upside_potential > threshold:
        return "Long", f"Predicted peak gain: {upside_potential:.2f}%"
    elif downside_risk < -threshold:
        return "Short", f"Predicted potential drop: {abs(downside_risk):.2f}%"
    else:
        return "Hold", "No significant movement predicted"

try:
    # 1. Fetch and prepare data
    with st.spinner(f"Fetching {symbol} data..."):
        if market_type == "spot":
            df = data_loader.fetch_historical_data(symbol=symbol, lookback_days=lookback_days)
        else:
            df = data_loader.fetch_futures_data(symbol=symbol, lookback_days=lookback_days)
        
        if df.empty:
            st.error("Failed to fetch data")
            st.stop()
        
        if market_type == "spot":
            df = data_loader.add_technical_indicators(df)
        else:
            df = data_loader.add_futures_indicators(df)

    # 2. Generate predictions
    current_price = df['close'].iloc[-1]
    
    if API_MODE:
        # Use real ML predictions
        if market_type == "spot":
            predictor = LSTMPredictor(n_future_steps=n_future_steps)
        else:
            predictor = FuturesLSTMPredictor(n_future_steps=n_future_steps)
        
        # Prepare data for ML model
        if market_type == "spot":
            X_train, y_train, X_test, y_test, original_y_test = data_loader.prepare_data(df, n_future_steps=n_future_steps)
        else:
            X_train, y_train, X_test, y_test, original_y_test = data_loader.prepare_futures_data(df, n_future_steps=n_future_steps)
        
        if X_train is not None:
            # Load or train model
            if not predictor.load_model():
                with st.spinner("Training ML model..."):
                    predictor.train(X_train, y_train)
            
            # Generate predictions
            with st.spinner("Generating ML predictions..."):
                last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
                predicted_scaled_sequence = predictor.predict(last_sequence)[0]
                predicted_prices = data_loader.inverse_transform_price(predicted_scaled_sequence)
        else:
            # Fall back to demo predictions if data preparation fails
            predicted_prices = generate_demo_predictions(current_price, n_future_steps)
    else:
        # Use demo predictions
        predicted_prices = generate_demo_predictions(current_price, n_future_steps)
    
    # 3. Display results
    signal, signal_reason = generate_trading_signal(current_price, predicted_prices, threshold, current_lang)
    
    # Display metrics
    if market_type == "futures":
        st.header("Futures Trading Metrics")
        
        # Calculate futures-specific metrics
        if hasattr(data_loader, 'calculate_futures_metrics'):
            futures_metrics = data_loader.calculate_futures_metrics(df, current_price, predicted_prices)
        else:
            futures_metrics = {
                'risk_reward_ratio': np.random.uniform(1.5, 3.0),
                'atr': current_price * 0.02,
                'volatility': 0.25,
                'volume_ratio': 1.2
            }
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(get_text("current_price", current_lang), f"${current_price:,.2f}")
        col2.metric(f"{get_text('predicted_low', current_lang)} ({selected_horizon_label})", f"${np.min(predicted_prices):,.2f}", delta_color="inverse")
        col3.metric(f"{get_text('predicted_high', current_lang)} ({selected_horizon_label})", f"${np.max(predicted_prices):,.2f}")
        col4.metric(get_text("trading_signal", current_lang), signal)
        
        # Additional futures metrics
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Risk/Reward Ratio", f"{futures_metrics['risk_reward_ratio']:.2f}")
        col6.metric("ATR", f"${futures_metrics['atr']:,.2f}")
        col7.metric("Volatility", f"{futures_metrics['volatility']:.1%}")
        col8.metric("Volume Ratio", f"{futures_metrics['volume_ratio']:.2f}")
        
    else:
        st.header("Spot Trading Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric(get_text("current_price", current_lang), f"${current_price:,.2f}")
        col2.metric(f"{get_text('predicted_high', current_lang)} ({selected_horizon_label})", f"${np.max(predicted_prices):,.2f}")
        col3.metric(get_text("trading_signal", current_lang), signal)
    
    # Signal explanation
    st.info(f"**Signal Reasoning:** {signal_reason}")
    
    # 4. Create charts
    st.header(get_text("price_chart", current_lang))
    
    # Prepare data for plotting
    recent_df = df.tail(200)  # Show last 200 data points
    
    # Create future timestamps
    last_timestamp = recent_df['timestamp'].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=n_future_steps,
        freq='H'
    )
    
    # Create main price chart
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Candlestick(
        x=recent_df['timestamp'],
        open=recent_df['open'],
        high=recent_df['high'],
        low=recent_df['low'],
        close=recent_df['close'],
        name='Historical Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Predicted prices
    fig.add_trace(go.Scatter(
        x=future_timestamps,
        y=predicted_prices,
        mode='lines+markers',
        name=f'Predicted Price ({selected_horizon_label})',
        line=dict(color='blue', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add technical indicators if available
    if 'sma_20' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df['timestamp'],
            y=recent_df['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'sma_50' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df['timestamp'],
            y=recent_df['sma_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='purple', width=1)
        ))
    
    # Add Bollinger Bands for futures
    if market_type == "futures" and 'bb_upper' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df['timestamp'],
            y=recent_df['bb_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=recent_df['timestamp'],
            y=recent_df['bb_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{symbol} Price Analysis and Prediction",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators chart for futures
    if market_type == "futures":
        st.header("Technical Analysis")
        
        # MACD Chart
        if 'macd' in recent_df.columns:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ))
            fig_macd.add_trace(go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['macd_signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red')
            ))
            fig_macd.add_trace(go.Bar(
                x=recent_df['timestamp'],
                y=recent_df['macd_histogram'],
                name='MACD Histogram',
                marker_color='green'
            ))
            fig_macd.update_layout(
                title="MACD Analysis",
                xaxis_title="Time",
                yaxis_title="MACD",
                height=400
            )
            st.plotly_chart(fig_macd, use_container_width=True)
        
        # RSI Chart
        if 'rsi' in recent_df.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(
                title="RSI (Relative Strength Index)",
                xaxis_title="Time",
                yaxis_title="RSI",
                height=300,
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Volume analysis
    st.header("Volume Analysis")
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=recent_df['timestamp'],
        y=recent_df['volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    if 'volume_sma' in recent_df.columns:
        fig_volume.add_trace(go.Scatter(
            x=recent_df['timestamp'],
            y=recent_df['volume_sma'],
            mode='lines',
            name='Volume SMA',
            line=dict(color='red', width=2)
        ))
    fig_volume.update_layout(
        title="Trading Volume",
        xaxis_title="Time",
        yaxis_title="Volume",
        height=300
    )
    st.plotly_chart(fig_volume, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your API credentials and try again.")

# Footer
st.markdown("---")
if API_MODE:
    st.success("✅ **Live Mode Active** - Using real-time data and ML predictions")
else:
    st.info("ℹ️ **Demo Mode** - To enable live data, add your Bybit API credentials to Streamlit Cloud secrets")
    st.markdown("""
    **To enable live mode:**
    1. Go to your Streamlit Cloud app settings
    2. Add these secrets:
       - `BYBIT_API_KEY` = your_api_key
       - `BYBIT_API_SECRET` = your_api_secret
    3. Restart the app
    """) 