import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from data_loader import DataLoader
from futures_data_loader import FuturesDataLoader
from predictor import LSTMPredictor
from futures_predictor import FuturesLSTMPredictor
from cryptocompare_data_loader import CryptoCompareDataLoader
from trading_pattern_analyzer import TradingPatternAnalyzer
import os
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

@st.cache_resource
def get_cryptocompare_data_loader():
    """Get cached CryptoCompare data loader for all cryptocurrencies"""
    try:
        # Get API key from environment
        api_key = os.getenv('CRYPTOCOMPARE_API_KEY')
        if not api_key:
            # Try Streamlit secrets
            try:
                api_key = st.secrets.get('CRYPTOCOMPARE_API_KEY', '')
            except:
                pass
        
        return CryptoCompareDataLoader(api_key=api_key)
    except Exception as e:
        st.warning(f"CryptoCompare data loader failed to initialize: {str(e)}")
        return None

@st.cache_resource
def get_trading_pattern_analyzer():
    """Get cached trading pattern analyzer"""
    try:
        return TradingPatternAnalyzer()
    except Exception as e:
        st.warning(f"Trading pattern analyzer failed to initialize: {str(e)}")
        return None

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
            
            # Enhance data with trading pattern features for better predictions
            pattern_analyzer = get_trading_pattern_analyzer()
            if pattern_analyzer:
                try:
                    df = pattern_analyzer.enhance_prediction_features(df.copy())
                    logging.info("Enhanced prediction features added to spot dataset")
                except Exception as e:
                    logging.warning(f"Could not enhance spot prediction features: {str(e)}")
            
            X_train, y_train, X_test, y_test, original_y_test = data_loader.prepare_data(df, n_future_steps=n_future_steps)
        else:
            df = data_loader.fetch_futures_data(symbol=symbol, lookback_days=lookback_days)
            if df.empty:
                st.error(get_text("failed_fetch_data", current_lang))
                st.stop()
            
            df = data_loader.add_futures_indicators(df)
            
            # Enhance data with trading pattern features for better predictions
            pattern_analyzer = get_trading_pattern_analyzer()
            if pattern_analyzer:
                try:
                    df = pattern_analyzer.enhance_prediction_features(df.copy())
                    logging.info("Enhanced prediction features added to futures dataset")
                except Exception as e:
                    logging.warning(f"Could not enhance futures prediction features: {str(e)}")
            
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
    
    # 4. Enhance predictions with CryptoCompare data (for ALL cryptocurrencies)
    cryptocompare_loader = get_cryptocompare_data_loader()
    cryptocompare_enhancement = {}
    if cryptocompare_loader:
        with st.spinner("Enhancing predictions with CryptoCompare market intelligence..."):
            current_market_data = {'current_price': current_price}
            cryptocompare_enhancement = cryptocompare_loader.enhance_prediction_with_cryptocompare_data(
                symbol, predicted_prices, current_market_data
            )
    
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
    
    # --- Enhanced Trading Pattern Analysis ---
    st.header("🔬 Advanced Trading Pattern Analysis")
    
    # Initialize trading pattern analyzer
    pattern_analyzer = get_trading_pattern_analyzer()
    
    if pattern_analyzer:
        with st.spinner("Analyzing trading patterns..."):
            try:
                # Detect price jumps
                jump_analysis = pattern_analyzer.detect_price_jumps(df, 
                                                                  jump_threshold=threshold, 
                                                                  min_volume_ratio=1.5)
                
                # Identify support/resistance levels
                sr_analysis = pattern_analyzer.identify_support_resistance_levels(df)
                
                # Analyze intraday patterns
                intraday_analysis = pattern_analyzer.analyze_intraday_patterns(df)
                
                # Calculate market microstructure features
                microstructure_features = pattern_analyzer.calculate_market_microstructure_features(df)
                
                # Generate trading insights
                trading_insights = pattern_analyzer.generate_trading_insights(
                    df, jump_analysis, sr_analysis, intraday_analysis, microstructure_features
                )
                
                # Display jump analysis
                if jump_analysis:
                    st.subheader("📊 Price Jump Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("⬆️ Upward Jumps", jump_analysis.get('jump_up_count', 0))
                    col2.metric("⬇️ Downward Jumps", jump_analysis.get('jump_down_count', 0))
                    col3.metric("📈 Avg Jump Size (Up)", f"{jump_analysis.get('avg_jump_up_size', 0):.2f}%")
                    col4.metric("📉 Avg Jump Size (Down)", f"{jump_analysis.get('avg_jump_down_size', 0):.2f}%")
                    
                    if jump_analysis.get('post_jump_analysis'):
                        with st.expander("🔍 Post-Jump Behavior Analysis"):
                            for period, data in jump_analysis['post_jump_analysis'].items():
                                st.write(f"**{period.replace('_', ' ').title()}:**")
                                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                                pcol1.metric("Avg Return After Up Jump", f"{data['avg_return_after_jump_up']:.2f}%")
                                pcol2.metric("Avg Return After Down Jump", f"{data['avg_return_after_jump_down']:.2f}%")
                                pcol3.metric("Up Jump Reversal Rate", f"{data['reversal_probability_up']*100:.1f}%")
                                pcol4.metric("Down Jump Reversal Rate", f"{data['reversal_probability_down']*100:.1f}%")
                
                # Display support/resistance analysis
                if sr_analysis and sr_analysis.get('levels'):
                    st.subheader("🎯 Support & Resistance Levels")
                    
                    # Show nearest levels
                    col1, col2 = st.columns(2)
                    
                    if sr_analysis.get('nearest_support'):
                        support = sr_analysis['nearest_support']
                        col1.metric("🟢 Nearest Support", 
                                   f"${support['price']:,.2f}", 
                                   f"-{support['distance_pct']:.2f}% ({support['strength']} touches)")
                    
                    if sr_analysis.get('nearest_resistance'):
                        resistance = sr_analysis['nearest_resistance']
                        col2.metric("🔴 Nearest Resistance", 
                                   f"${resistance['price']:,.2f}", 
                                   f"+{resistance['distance_pct']:.2f}% ({resistance['strength']} touches)")
                    
                    # Show all significant levels
                    with st.expander("📋 All Significant Levels"):
                        levels_df = pd.DataFrame(sr_analysis['levels'])
                        if not levels_df.empty:
                            levels_df['price'] = levels_df['price'].apply(lambda x: f"${x:,.2f}")
                            levels_df['distance_pct'] = levels_df['distance_pct'].apply(lambda x: f"{x:.2f}%")
                            st.dataframe(levels_df, use_container_width=True)
                
                # Display intraday patterns
                if intraday_analysis:
                    st.subheader("⏰ Intraday Trading Patterns")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("📈 Peak Volume Hour", f"{intraday_analysis.get('peak_volume_hour', 'N/A')}:00 UTC")
                    col2.metric("⚡ Peak Volatility Hour", f"{intraday_analysis.get('peak_volatility_hour', 'N/A')}:00 UTC")
                    col3.metric("🟢 Most Bullish Hour", f"{intraday_analysis.get('most_bullish_hour', 'N/A')}:00 UTC")
                    col4.metric("🔴 Most Bearish Hour", f"{intraday_analysis.get('most_bearish_hour', 'N/A')}:00 UTC")
                    
                    # Session analysis
                    if intraday_analysis.get('session_analysis'):
                        with st.expander("🌍 Trading Session Analysis"):
                            for session, data in intraday_analysis['session_analysis'].items():
                                st.write(f"**{session.title()} Session ({session.upper()} Hours):**")
                                scol1, scol2, scol3, scol4 = st.columns(4)
                                scol1.metric("Avg Volume", f"{data['avg_volume']:,.0f}")
                                scol2.metric("Avg Volatility", f"{data['avg_volatility']:.2f}%")
                                scol3.metric("Avg Price Change", f"{data['avg_price_change']:.2f}%")
                                bias_color = "🟢" if data['directional_bias'] == 'bullish' else "🔴"
                                scol4.metric("Directional Bias", f"{bias_color} {data['directional_bias'].title()}")
                
                # Display market microstructure
                if microstructure_features:
                    st.subheader("🧬 Market Microstructure")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    buying_pressure = microstructure_features.get('avg_buying_pressure', 0.5)
                    selling_pressure = microstructure_features.get('avg_selling_pressure', 0.5)
                    net_pressure = microstructure_features.get('net_pressure_trend', 0)
                    momentum = microstructure_features.get('momentum_strength', 0)
                    
                    col1.metric("🟢 Buying Pressure", f"{buying_pressure:.3f}")
                    col2.metric("🔴 Selling Pressure", f"{selling_pressure:.3f}")
                    col3.metric("⚖️ Net Pressure", f"{net_pressure:+.3f}")
                    col4.metric("🚀 Momentum Strength", f"{momentum:+.3f}")
                    
                    # Market regime
                    regime = microstructure_features.get('market_regime', {})
                    vol_regime = regime.get('volatility_regime', 'unknown')
                    volume_regime = regime.get('volume_regime', 'unknown')
                    
                    col1, col2 = st.columns(2)
                    vol_color = "🟡" if vol_regime == 'high' else "🟢"
                    vol_color_regime = "🟡" if volume_regime == 'high' else "🟢"
                    col1.metric("📊 Volatility Regime", f"{vol_color} {vol_regime.title()}")
                    col2.metric("📈 Volume Regime", f"{vol_color_regime} {volume_regime.title()}")
                
                # Display trading insights
                if trading_insights:
                    st.subheader("💡 Trading Insights & Recommendations")
                    
                    # Trading opportunities
                    if trading_insights.get('trading_opportunities'):
                        st.write("**🎯 Trading Opportunities:**")
                        for opp in trading_insights['trading_opportunities']:
                            confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(opp.get('confidence', 'low'), "🔴")
                            st.success(f"{confidence_color} **{opp['type'].replace('_', ' ').title()}**: {opp['signal']}")
                    
                    # Risk factors
                    if trading_insights.get('risk_factors'):
                        st.write("**⚠️ Risk Factors:**")
                        for risk in trading_insights['risk_factors']:
                            st.warning(f"• {risk}")
                    
                    # Pattern signals
                    if trading_insights.get('pattern_signals'):
                        st.write("**📡 Pattern Signals:**")
                        for signal in trading_insights['pattern_signals']:
                            if signal.get('actionable'):
                                st.info(f"💡 **{signal['type'].replace('_', ' ').title()}**: {signal['signal']}")
                            else:
                                st.write(f"• {signal['signal']}")
                
                # Create enhanced trading chart
                st.subheader("📊 Enhanced Trading Chart")
                enhanced_chart = pattern_analyzer.create_enhanced_trading_chart(
                    df, jump_analysis, sr_analysis, symbol
                )
                st.plotly_chart(enhanced_chart, use_container_width=True)
                
                # Enhance prediction features for better accuracy
                enhanced_df = pattern_analyzer.enhance_prediction_features(df.copy())
                st.success("✅ Trading pattern analysis completed! Enhanced features have been integrated for more accurate predictions.")
                
            except Exception as e:
                st.error(f"Error in trading pattern analysis: {str(e)}")
                logging.error(f"Trading pattern analysis error: {str(e)}")
    
    # --- CryptoCompare Enhanced Analysis (for ALL cryptocurrencies) ---
    if cryptocompare_enhancement:
        base_symbol = cryptocompare_loader._extract_base_symbol(symbol)
        st.header(f"🌐 CryptoCompare Market Intelligence - {base_symbol}")
        
        # Current price comparison
        if cryptocompare_enhancement.get('cryptocompare_data'):
            cc_data = cryptocompare_enhancement['cryptocompare_data']
            col1, col2, col3 = st.columns(3)
            
            cc_price = cc_data['price']
            col1.metric("🌐 CryptoCompare Price", f"${cc_price:,.4f}")
            
            if cryptocompare_enhancement.get('price_comparison'):
                comparison = cryptocompare_enhancement['price_comparison']
                exchange_price = comparison['exchange_price']
                deviation = comparison['deviation_pct']
                
                col2.metric("📊 Exchange Price", f"${exchange_price:,.4f}")
                col3.metric("📏 Price Deviation", f"{deviation:.2f}%", 
                           delta=f"±{deviation:.2f}%" if deviation > 0.5 else "Aligned")
        
        # Market metrics
        if cryptocompare_enhancement.get('market_metrics'):
            metrics = cryptocompare_enhancement['market_metrics']
            
            st.subheader("📊 Real-Time Market Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            if 'change_pct_24h' in metrics:
                change_24h = metrics['change_pct_24h']
                change_color = "🟢" if change_24h >= 0 else "🔴"
                col1.metric(f"{change_color} 24h Change", f"{change_24h:+.2f}%")
            
            if 'volume_24h' in metrics and metrics['volume_24h'] > 0:
                volume_m = metrics['volume_24h'] / 1_000_000
                col2.metric("📈 24h Volume", f"${volume_m:.1f}M")
            
            if 'market_cap' in metrics and metrics['market_cap'] > 0:
                market_cap_b = metrics['market_cap'] / 1_000_000_000
                col3.metric("💎 Market Cap", f"${market_cap_b:.1f}B")
            
            if 'high_24h' in metrics and 'low_24h' in metrics:
                high_24h = metrics['high_24h']
                low_24h = metrics['low_24h']
                range_pct = ((high_24h - low_24h) / low_24h) * 100 if low_24h > 0 else 0
                col4.metric("📏 24h Range", f"{range_pct:.2f}%")
        
        # Advanced sentiment analysis
        if cryptocompare_enhancement.get('sentiment_analysis'):
            sentiment = cryptocompare_enhancement['sentiment_analysis']
            
            st.subheader("🧠 Advanced Sentiment Analysis")
            
            # Overall sentiment score
            overall_score = sentiment.get('overall_score', 50)
            confidence = sentiment.get('confidence', 'medium')
            
            col1, col2, col3 = st.columns(3)
            
            # Sentiment gauge
            if overall_score >= 80:
                sentiment_emoji = "🚀"
                sentiment_text = "Very Bullish"
                sentiment_color = "success"
            elif overall_score >= 65:
                sentiment_emoji = "📈"
                sentiment_text = "Bullish"
                sentiment_color = "success"
            elif overall_score >= 35:
                sentiment_emoji = "📊"
                sentiment_text = "Neutral"
                sentiment_color = "info"
            elif overall_score >= 20:
                sentiment_emoji = "📉"
                sentiment_text = "Bearish"
                sentiment_color = "warning"
            else:
                sentiment_emoji = "⛔"
                sentiment_text = "Very Bearish"
                sentiment_color = "error"
            
            col1.metric(f"{sentiment_emoji} Sentiment Score", f"{overall_score:.1f}/100")
            col2.metric("📊 Analysis", sentiment_text)
            col3.metric("🎯 Confidence", confidence.title())
            
            # Sentiment factors breakdown
            if 'factors' in sentiment:
                factors = sentiment['factors']
                if factors:
                    with st.expander("📊 Sentiment Factors Breakdown"):
                        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                        
                        if 'price_momentum' in factors:
                            momentum_score = factors['price_momentum']
                            momentum_color = "🟢" if momentum_score >= 60 else "🟡" if momentum_score >= 40 else "🔴"
                            fcol1.metric(f"{momentum_color} Price Momentum", f"{momentum_score:.1f}/100")
                        
                        if 'volume_strength' in factors:
                            volume_score = factors['volume_strength']
                            volume_color = "🟢" if volume_score >= 60 else "🟡" if volume_score >= 40 else "🔴"
                            fcol2.metric(f"{volume_color} Volume Strength", f"{volume_score:.1f}/100")
                        
                        if 'social_activity' in factors:
                            social_score = factors['social_activity']
                            social_color = "🟢" if social_score >= 60 else "🟡" if social_score >= 40 else "🔴"
                            fcol3.metric(f"{social_color} Social Activity", f"{social_score:.1f}/100")
                        
                        if 'news_sentiment' in factors:
                            news_score = factors['news_sentiment']
                            news_color = "🟢" if news_score >= 60 else "🟡" if news_score >= 40 else "🔴"
                            fcol4.metric(f"{news_color} News Sentiment", f"{news_score:.1f}/100")
            
            # Sentiment signals
            if 'signals' in sentiment and sentiment['signals']:
                st.subheader("📡 Market Signals")
                for signal in sentiment['signals']:
                    if "Very Bullish" in signal or "Bullish" in signal:
                        st.success(signal)
                    elif "Very Bearish" in signal or "Bearish" in signal:
                        st.error(signal)
                    elif "Neutral" in signal:
                        st.info(signal)
                    else:
                        st.warning(signal)
        
        # Prediction confidence and risk assessment
        confidence = cryptocompare_enhancement.get('prediction_confidence', 'medium')
        confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "🟡")
        st.info(f"{confidence_color} **Prediction Confidence**: {confidence.title()}")
        
        # Risk factors
        if cryptocompare_enhancement.get('risk_factors'):
            st.subheader("⚠️ Risk Factors")
            for risk in cryptocompare_enhancement['risk_factors']:
                st.warning(f"• {risk}")
        
        # Opportunities
        if cryptocompare_enhancement.get('opportunities'):
            st.subheader("✅ Market Opportunities")
            for opportunity in cryptocompare_enhancement['opportunities']:
                st.success(f"• {opportunity}")
        
        st.caption(f"💡 CryptoCompare data enhances {base_symbol} predictions with comprehensive market intelligence, sentiment analysis, and social indicators.")

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