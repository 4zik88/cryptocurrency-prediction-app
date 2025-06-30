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
from enhanced_short_term_predictor import EnhancedShortTermPredictor

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
        st.warning(f"{get_text('cryptocompare_data_loader_failed', get_current_language())} {str(e)}")
        return None

@st.cache_resource
def get_trading_pattern_analyzer():
    """Get cached trading pattern analyzer"""
    try:
        return TradingPatternAnalyzer()
    except Exception as e:
        st.warning(f"{get_text('trading_pattern_analyzer_failed', get_current_language())} {str(e)}")
        return None

@st.cache_resource
def get_enhanced_short_term_predictor(horizon, market_type):
    """Get cached enhanced short-term predictor"""
    return EnhancedShortTermPredictor(n_future_steps=horizon, market_type=market_type)

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
        st.session_state.model_type = "classic"

def reset_filters():
    """Reset all filter values to defaults."""
    st.session_state.market_type = "spot"
    st.session_state.symbol = "BTCUSDT"
    st.session_state.n_future_steps = 1
    st.session_state.lookback_days = 365
    st.session_state.threshold = 2.0
    st.session_state.model_type = "classic"
    st.rerun()

# Initialize state at the beginning of the script
initialize_state()

# --- Sidebar Configuration ---
st.sidebar.title(get_text("configuration", current_lang))

# Language switcher
languages = {"English": "en", "Русский": "ru"}
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

# Model Type Selection
model_types_map = {
    get_text("classic_model", current_lang): "classic",
    get_text("enhanced_short_term_model", current_lang): "enhanced_short_term"
}
model_display_options = list(model_types_map.keys())
model_internal_values = list(model_types_map.values())

try:
    current_model_index = model_internal_values.index(st.session_state.model_type)
except (ValueError, KeyError):
    current_model_index = 0
    st.session_state.model_type = "classic"

selected_model_display = st.sidebar.selectbox(
    get_text("select_model_type", current_lang),
    model_display_options,
    index=current_model_index
)
st.session_state.model_type = model_types_map[selected_model_display]

# Initialize appropriate data loader based on market type
data_loader = get_spot_data_loader() if st.session_state.market_type == "spot" else get_futures_data_loader()

# Dynamic Symbol selection based on market type
cache_key = f'available_pairs_{st.session_state.market_type}'
if cache_key not in st.session_state:
    with st.spinner(get_text("fetching_available_pairs", current_lang)):
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

if st.session_state.model_type == "enhanced_short_term":
    predictor = get_enhanced_short_term_predictor(n_future_steps, st.session_state.market_type)
else:
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
            df = data_loader.fetch_historical_data(symbol=symbol, lookback_days=lookback_days, interval='60')
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
            
            X_train, y_train, X_test, y_test, original_y_test = data_loader.prepare_data(df, n_future_steps=n_future_steps, sequence_length=predictor.sequence_length if hasattr(predictor, 'sequence_length') else 24)
        else:
            df = data_loader.fetch_futures_data(symbol=symbol, lookback_days=lookback_days, interval='60')
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
            
            X_train, y_train, X_test, y_test, original_y_test = data_loader.prepare_futures_data(df, n_future_steps=n_future_steps, sequence_length=predictor.sequence_length if hasattr(predictor, 'sequence_length') else 24)

        if X_train is None:
            st.error(get_text("insufficient_data", current_lang))
            st.stop()

    # 2. Load or train model
    n_features = X_train.shape[2]
    if not predictor.load_model(n_features):
        with st.spinner(get_text("training_model", current_lang, selected_horizon_display)):
            if st.session_state.model_type == "enhanced_short_term":
                # For enhanced model, we can pass validation data directly if available
                # This can be improved by splitting data inside the train function.
                predictor.train(X_train, y_train) 
            else:
                predictor.train(X_train, y_train)
    
    # 3. Generate and process predictions
    with st.spinner(get_text("generating_predictions", current_lang)):
        last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
        
        # Enhanced predictor has uncertainty estimation
        if st.session_state.model_type == "enhanced_short_term":
            predicted_scaled_sequence, lower_bound, upper_bound = predictor.predict_with_uncertainty(last_sequence)
            predicted_prices = data_loader.inverse_transform_price(predicted_scaled_sequence.flatten())
            lower_prices = data_loader.inverse_transform_price(lower_bound.flatten())
            upper_prices = data_loader.inverse_transform_price(upper_bound.flatten())
        else:
            predicted_scaled_sequence = predictor.predict(last_sequence)[0]
            predicted_prices = data_loader.inverse_transform_price(predicted_scaled_sequence)
            lower_prices, upper_prices = None, None

    current_price = df['close'].iloc[-1]
    
    # 4. Enhance predictions with CryptoCompare data (for ALL cryptocurrencies)
    cryptocompare_loader = get_cryptocompare_data_loader()
    cryptocompare_enhancement = {}
    if cryptocompare_loader:
        with st.spinner(get_text("enhancing_predictions", current_lang)):
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
        col1.metric(f"💰 {get_text('current_price', current_lang)}", f"${current_price:,.4f}")
        
        predicted_final = predicted_prices[-1]
        change_final = (predicted_final - current_price) / current_price * 100
        col2.metric(f"🎯 {get_text('exact_forecast', current_lang)} ({selected_horizon_display})", 
                   f"${predicted_final:,.4f}", 
                   f"{change_final:+.2f}%")
        
        predicted_avg = np.mean(predicted_prices)
        change_avg = (predicted_avg - current_price) / current_price * 100
        col3.metric(f"📊 {get_text('average_price', current_lang)} ({selected_horizon_display})", 
                   f"${predicted_avg:,.4f}", 
                   f"{change_avg:+.2f}%")
        
        # Detailed step-by-step forecast
        st.subheader(f"📈 {get_text('step_by_step_forecast', current_lang)}")
        
        if n_future_steps > 1:
            forecast_df = pd.DataFrame({
                get_text('hour', current_lang): [f"{get_text('hour', current_lang)} {i+1}" for i in range(n_future_steps)],
                get_text('predicted_price', current_lang): [f"${price:,.4f}" for price in predicted_prices],
                get_text('change_from_current', current_lang): [f"{((price - current_price) / current_price * 100):+.2f}%" for price in predicted_prices],
                get_text('price_movement', current_lang): [
                    f"📈 {get_text('up', current_lang)}" if price > current_price else f"📉 {get_text('down', current_lang)}" if price < current_price else f"➡️ {get_text('flat', current_lang)}"
                    for price in predicted_prices
                ]
            })
            
            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"🎯 **{get_text('exact_forecast_for_next', current_lang)} {selected_horizon_display}:** ${predicted_final:,.4f} ({change_final:+.2f}%)")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        predicted_low = np.min(predicted_prices)
        change_low = (predicted_low - current_price) / current_price * 100
        col1.metric(f"📉 {get_text('minimum_expected', current_lang)}", f"${predicted_low:,.4f}", f"{change_low:+.2f}%")
        
        predicted_high = np.max(predicted_prices)
        change_high = (predicted_high - current_price) / current_price * 100
        col2.metric(f"📈 {get_text('maximum_expected', current_lang)}", f"${predicted_high:,.4f}", f"{change_high:+.2f}%")
        
        price_range = predicted_high - predicted_low
        range_pct = (price_range / current_price) * 100
        col3.metric(f"📏 {get_text('price_range', current_lang)}", f"${price_range:,.4f}", f"{range_pct:.2f}%")
        
        volatility = np.std(predicted_prices)
        vol_pct = (volatility / current_price) * 100
        col4.metric(f"⚡ {get_text('volatility', current_lang)}", f"${volatility:,.4f}", f"{vol_pct:.2f}%")
        
        # Additional futures metrics
        if 'risk_reward_ratio' in futures_metrics:
            st.subheader(f"🎯 {get_text('futures_specific_metrics', current_lang)}")
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
        col1.metric(f"💰 {get_text('current_price', current_lang)}", f"${current_price:,.4f}")
        
        predicted_final = predicted_prices[-1]
        change_final = (predicted_final - current_price) / current_price * 100
        col2.metric(f"🎯 {get_text('exact_forecast', current_lang)} ({selected_horizon_display})", 
                   f"${predicted_final:,.4f}", 
                   f"{change_final:+.2f}%")
        
        predicted_avg = np.mean(predicted_prices)
        change_avg = (predicted_avg - current_price) / current_price * 100
        col3.metric(f"📊 {get_text('average_price', current_lang)} ({selected_horizon_display})", 
                   f"${predicted_avg:,.4f}", 
                   f"{change_avg:+.2f}%")
        
        # Detailed step-by-step forecast
        st.subheader(f"📈 {get_text('step_by_step_forecast', current_lang)}")
        
        if n_future_steps > 1:
            forecast_df = pd.DataFrame({
                get_text('hour', current_lang): [f"{get_text('hour', current_lang)} {i+1}" for i in range(n_future_steps)],
                get_text('predicted_price', current_lang): [f"${price:,.4f}" for price in predicted_prices],
                get_text('change_from_current', current_lang): [f"{((price - current_price) / current_price * 100):+.2f}%" for price in predicted_prices],
                get_text('price_movement', current_lang): [
                    f"📈 {get_text('up', current_lang)}" if price > current_price else f"📉 {get_text('down', current_lang)}" if price < current_price else f"➡️ {get_text('flat', current_lang)}"
                    for price in predicted_prices
                ]
            })
            
            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"🎯 **{get_text('exact_forecast_for_next', current_lang)} {selected_horizon_display}:** ${predicted_final:,.4f} ({change_final:+.2f}%)")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        predicted_low = np.min(predicted_prices)
        change_low = (predicted_low - current_price) / current_price * 100
        col1.metric(f"📉 {get_text('minimum_expected', current_lang)}", f"${predicted_low:,.4f}", f"{change_low:+.2f}%")
        
        predicted_high = np.max(predicted_prices)
        change_high = (predicted_high - current_price) / current_price * 100
        col2.metric(f"📈 {get_text('maximum_expected', current_lang)}", f"${predicted_high:,.4f}", f"{change_high:+.2f}%")
        
        price_range = predicted_high - predicted_low
        range_pct = (price_range / current_price) * 100
        col3.metric(f"📏 {get_text('price_range', current_lang)}", f"${price_range:,.4f}", f"{range_pct:.2f}%")
        
        volatility = np.std(predicted_prices)
        vol_pct = (volatility / current_price) * 100
        col4.metric(f"⚡ {get_text('volatility', current_lang)}", f"${volatility:,.4f}", f"{vol_pct:.2f}%")
    
    st.subheader(get_text("trading_signal", current_lang))
    if signal == "Long":
        st.success(get_text("go_long", current_lang))
    elif signal == "Short":
        st.error(get_text("go_short", current_lang))
    else:
        st.info(get_text("hold_neutral", current_lang))
    st.caption(signal_reason)
    
    # --- Enhanced Trading Pattern Analysis ---
    st.header(f"🔬 {get_text('advanced_trading_pattern_analysis', current_lang)}")
    
    # Initialize trading pattern analyzer
    pattern_analyzer = get_trading_pattern_analyzer()
    
    if pattern_analyzer:
        with st.spinner(get_text("analyzing_trading_patterns", current_lang)):
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
                    st.subheader(f"📊 {get_text('price_jump_analysis', current_lang)}")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric(f"⬆️ {get_text('upward_jumps', current_lang)}", jump_analysis.get('jump_up_count', 0))
                    col2.metric(f"⬇️ {get_text('downward_jumps', current_lang)}", jump_analysis.get('jump_down_count', 0))
                    col3.metric(f"📈 {get_text('avg_jump_size_up', current_lang)}", f"{jump_analysis.get('avg_jump_up_size', 0):.2f}%")
                    col4.metric(f"📉 {get_text('avg_jump_size_down', current_lang)}", f"{jump_analysis.get('avg_jump_down_size', 0):.2f}%")
                    
                    if jump_analysis.get('post_jump_analysis'):
                        with st.expander("🔍 Post-Jump Behavior Analysis"):
                            for period, data in jump_analysis['post_jump_analysis'].items():
                                st.write(f"**{period.replace('_', ' ').title()}:**")
                                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                                pcol1.metric(get_text('avg_return_after_up_jump', current_lang), f"{data['avg_return_after_jump_up']:.2f}%")
                                pcol2.metric(get_text('avg_return_after_down_jump', current_lang), f"{data['avg_return_after_jump_down']:.2f}%")
                                pcol3.metric(get_text('up_jump_reversal_rate', current_lang), f"{data['reversal_probability_up']*100:.1f}%")
                                pcol4.metric(get_text('down_jump_reversal_rate', current_lang), f"{data['reversal_probability_down']*100:.1f}%")
                
                # Display support/resistance analysis
                if sr_analysis and sr_analysis.get('levels'):
                    st.subheader(f"🎯 {get_text('support_resistance_levels', current_lang)}")
                    
                    # Show nearest levels
                    col1, col2 = st.columns(2)
                    
                    if sr_analysis.get('nearest_support'):
                        support = sr_analysis['nearest_support']
                        col1.metric(f"🟢 {get_text('nearest_support', current_lang)}", 
                                   f"${support['price']:,.2f} ({support['strength']} {get_text('touches', current_lang)})", 
                                   f"-{support['distance_pct']:.2f}%")
                    
                    if sr_analysis.get('nearest_resistance'):
                        resistance = sr_analysis['nearest_resistance']
                        col2.metric(f"🔴 {get_text('nearest_resistance', current_lang)}", 
                                   f"${resistance['price']:,.2f} ({resistance['strength']} {get_text('touches', current_lang)})", 
                                   f"+{resistance['distance_pct']:.2f}%")
                    
                    # Show all significant levels
                    with st.expander(f"📋 {get_text('all_significant_levels', current_lang)}"):
                        levels_df = pd.DataFrame(sr_analysis['levels'])
                        if not levels_df.empty:
                            levels_df['price'] = levels_df['price'].apply(lambda x: f"${x:,.2f}")
                            levels_df['distance_pct'] = levels_df['distance_pct'].apply(lambda x: f"{x:.2f}%")
                            st.dataframe(levels_df, use_container_width=True)
                
                # Display intraday patterns
                if intraday_analysis:
                    st.subheader(f"⏰ {get_text('intraday_trading_patterns', current_lang)}")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric(f"📈 {get_text('peak_volume_hour', current_lang)}", f"{intraday_analysis.get('peak_volume_hour', 'N/A')}:00 UTC")
                    col2.metric(f"⚡ {get_text('peak_volatility_hour', current_lang)}", f"{intraday_analysis.get('peak_volatility_hour', 'N/A')}:00 UTC")
                    col3.metric(f"🟢 {get_text('most_bullish_hour', current_lang)}", f"{intraday_analysis.get('most_bullish_hour', 'N/A')}:00 UTC")
                    col4.metric(f"🔴 {get_text('most_bearish_hour', current_lang)}", f"{intraday_analysis.get('most_bearish_hour', 'N/A')}:00 UTC")
                    
                    # Session analysis
                    if intraday_analysis.get('session_analysis'):
                        with st.expander(f"🌍 {get_text('trading_session_analysis', current_lang)}"):
                            for session, data in intraday_analysis['session_analysis'].items():
                                st.write(f"**{session.title()} {get_text('session', current_lang)} ({session.upper()} {get_text('hours', current_lang)}):**")
                                scol1, scol2, scol3, scol4 = st.columns(4)
                                scol1.metric(get_text('avg_volume', current_lang), f"{data['avg_volume']:,.0f}")
                                scol2.metric(get_text('avg_volatility', current_lang), f"{data['avg_volatility']:.2f}%")
                                scol3.metric(get_text('avg_price_change', current_lang), f"{data['avg_price_change']:.2f}%")
                                bias_color = "🟢" if data['directional_bias'] == 'bullish' else "🔴"
                                scol4.metric(get_text('directional_bias', current_lang), f"{bias_color} {data['directional_bias'].title()}")
                
                # Display market microstructure
                if microstructure_features:
                    st.subheader(f"🧬 {get_text('market_microstructure', current_lang)}")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    buying_pressure = microstructure_features.get('avg_buying_pressure', 0.5)
                    selling_pressure = microstructure_features.get('avg_selling_pressure', 0.5)
                    net_pressure = microstructure_features.get('net_pressure_trend', 0)
                    momentum = microstructure_features.get('momentum_strength', 0)
                    
                    col1.metric(f"🟢 {get_text('buying_pressure', current_lang)}", f"{buying_pressure:.3f}")
                    col2.metric(f"🔴 {get_text('selling_pressure', current_lang)}", f"{selling_pressure:.3f}")
                    col3.metric(f"⚖️ {get_text('net_pressure', current_lang)}", f"{net_pressure:+.3f}")
                    col4.metric(f"🚀 {get_text('momentum_strength', current_lang)}", f"{momentum:+.3f}")
                    
                    # Market regime
                    regime = microstructure_features.get('market_regime', {})
                    vol_regime = regime.get('volatility_regime', 'unknown')
                    volume_regime = regime.get('volume_regime', 'unknown')
                    
                    col1, col2 = st.columns(2)
                    vol_color = "🟡" if vol_regime == 'high' else "🟢"
                    vol_color_regime = "🟡" if volume_regime == 'high' else "🟢"
                    col1.metric(f"📊 {get_text('volatility_regime', current_lang)}", f"{vol_color} {vol_regime.title()}")
                    col2.metric(f"📈 {get_text('volume_regime', current_lang)}", f"{vol_color_regime} {volume_regime.title()}")
                
                # Display trading insights
                if trading_insights:
                    st.subheader(f"💡 {get_text('trading_insights_recommendations', current_lang)}")
                    
                    # Trading opportunities
                    if trading_insights.get('trading_opportunities'):
                        st.write(f"**🎯 {get_text('trading_opportunities', current_lang)}:**")
                        for opp in trading_insights['trading_opportunities']:
                            confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(opp.get('confidence', 'low'), "🔴")
                            st.success(f"{confidence_color} **{opp['type'].replace('_', ' ').title()}**: {opp['signal']}")
                    
                    # Risk factors
                    if trading_insights.get('risk_factors'):
                        st.write(f"**⚠️ {get_text('risk_factors', current_lang)}:**")
                        for risk in trading_insights['risk_factors']:
                            st.warning(f"• {risk}")
                    
                    # Pattern signals
                    if trading_insights.get('pattern_signals'):
                        st.write(f"**📡 {get_text('pattern_signals', current_lang)}:**")
                        for signal in trading_insights['pattern_signals']:
                            if signal.get('actionable'):
                                st.info(f"💡 **{signal['type'].replace('_', ' ').title()}**: {signal['signal']}")
                            else:
                                st.write(f"• {signal['signal']}")
                
                # Create enhanced trading chart
                st.subheader(f"📊 {get_text('enhanced_trading_chart', current_lang)}")
                enhanced_chart = pattern_analyzer.create_enhanced_trading_chart(
                    df, jump_analysis, sr_analysis, symbol
                )
                st.plotly_chart(enhanced_chart, use_container_width=True)
                
                # Enhance prediction features for better accuracy
                enhanced_df = pattern_analyzer.enhance_prediction_features(df.copy())
                st.success(f"✅ {get_text('trading_pattern_analysis_completed', current_lang)}")
                
            except Exception as e:
                st.error(f"{get_text('error_in_trading_pattern_analysis', current_lang)} {str(e)}")
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
            col1.metric(f"🌐 {get_text('cryptocompare_price', current_lang)}", f"${cc_price:,.4f}")
            
            if cryptocompare_enhancement.get('price_comparison'):
                comparison = cryptocompare_enhancement['price_comparison']
                exchange_price = comparison['exchange_price']
                deviation = comparison['deviation_pct']
                
                col2.metric(f"📊 {get_text('exchange_price', current_lang)}", f"${exchange_price:,.4f}")
                col3.metric(f"📏 {get_text('price_deviation', current_lang)}", f"{deviation:.2f}%", 
                           delta=f"±{deviation:.2f}%" if deviation > 0.5 else get_text("aligned", current_lang))
        
        # Market metrics
        if cryptocompare_enhancement.get('market_metrics'):
            metrics = cryptocompare_enhancement['market_metrics']
            
            st.subheader(f"📊 {get_text('real_time_market_metrics', current_lang)}")
            col1, col2, col3, col4 = st.columns(4)
            
            if 'change_pct_24h' in metrics:
                change_24h = metrics['change_pct_24h']
                change_color = "🟢" if change_24h >= 0 else "🔴"
                col1.metric(f"{change_color} {get_text('24h_change', current_lang)}", f"{change_24h:+.2f}%")
            
            if 'volume_24h' in metrics and metrics['volume_24h'] > 0:
                volume_m = metrics['volume_24h'] / 1_000_000
                col2.metric(f"📈 {get_text('24h_volume', current_lang)}", f"${volume_m:.1f}M")
            
            if 'market_cap' in metrics and metrics['market_cap'] > 0:
                market_cap_b = metrics['market_cap'] / 1_000_000_000
                col3.metric(f"💎 {get_text('market_cap', current_lang)}", f"${market_cap_b:.1f}B")
            
            if 'high_24h' in metrics and 'low_24h' in metrics:
                high_24h = metrics['high_24h']
                low_24h = metrics['low_24h']
                range_pct = ((high_24h - low_24h) / low_24h) * 100 if low_24h > 0 else 0
                col4.metric(f"📏 {get_text('24h_range', current_lang)}", f"{range_pct:.2f}%")
        
        # Advanced sentiment analysis
        if cryptocompare_enhancement.get('sentiment_analysis'):
            sentiment = cryptocompare_enhancement['sentiment_analysis']
            
            st.subheader(f"🧠 {get_text('advanced_sentiment_analysis', current_lang)}")
            
            # Overall sentiment score
            overall_score = sentiment.get('overall_score', 50)
            confidence = sentiment.get('confidence', 'medium')
            
            col1, col2, col3 = st.columns(3)
            
            # Sentiment gauge
            if overall_score >= 80:
                sentiment_emoji = "🚀"
                sentiment_text = get_text("very_bullish", current_lang)
                sentiment_color = "success"
            elif overall_score >= 65:
                sentiment_emoji = "📈"
                sentiment_text = get_text("bullish", current_lang)
                sentiment_color = "success"
            elif overall_score >= 35:
                sentiment_emoji = "📊"
                sentiment_text = get_text("neutral", current_lang)
                sentiment_color = "info"
            elif overall_score >= 20:
                sentiment_emoji = "📉"
                sentiment_text = get_text("bearish", current_lang)
                sentiment_color = "warning"
            else:
                sentiment_emoji = "⛔"
                sentiment_text = get_text("very_bearish", current_lang)
                sentiment_color = "error"
            
            col1.metric(f"{sentiment_emoji} {get_text('sentiment_score', current_lang)}", f"{overall_score:.1f}/100")
            col2.metric(f"📊 {get_text('analysis', current_lang)}", sentiment_text)
            col3.metric(f"🎯 {get_text('confidence', current_lang)}", confidence.title())
            
            # Sentiment factors breakdown
            if 'factors' in sentiment:
                factors = sentiment['factors']
                if factors:
                    with st.expander(f"📊 {get_text('sentiment_factors_breakdown', current_lang)}"):
                        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                        
                        if 'price_momentum' in factors:
                            momentum_score = factors['price_momentum']
                            momentum_color = "🟢" if momentum_score >= 60 else "🟡" if momentum_score >= 40 else "🔴"
                            fcol1.metric(f"{momentum_color} {get_text('price_momentum', current_lang)}", f"{momentum_score:.1f}/100")
                        
                        if 'volume_strength' in factors:
                            volume_score = factors['volume_strength']
                            volume_color = "🟢" if volume_score >= 60 else "🟡" if volume_score >= 40 else "🔴"
                            fcol2.metric(f"{volume_color} {get_text('volume_strength', current_lang)}", f"{volume_score:.1f}/100")
                        
                        if 'social_activity' in factors:
                            social_score = factors['social_activity']
                            social_color = "🟢" if social_score >= 60 else "🟡" if social_score >= 40 else "🔴"
                            fcol3.metric(f"{social_color} {get_text('social_activity', current_lang)}", f"{social_score:.1f}/100")
                        
                        if 'news_sentiment' in factors:
                            news_score = factors['news_sentiment']
                            news_color = "🟢" if news_score >= 60 else "🟡" if news_score >= 40 else "🔴"
                            fcol4.metric(f"{news_color} {get_text('news_sentiment', current_lang)}", f"{news_score:.1f}/100")
            
            # Sentiment signals
            if 'signals' in sentiment and sentiment['signals']:
                st.subheader(f"📡 {get_text('market_signals', current_lang)}")
                for signal in sentiment['signals']:
                    very_bullish_text = get_text("very_bullish", current_lang)
                    bullish_text = get_text("bullish", current_lang)
                    very_bearish_text = get_text("very_bearish", current_lang) 
                    bearish_text = get_text("bearish", current_lang)
                    neutral_text = get_text("neutral", current_lang)
                    
                    if very_bullish_text in signal or bullish_text in signal:
                        st.success(signal)
                    elif very_bearish_text in signal or bearish_text in signal:
                        st.error(signal)
                    elif neutral_text in signal:
                        st.info(signal)
                    else:
                        st.warning(signal)
        
        # Prediction confidence and risk assessment
        confidence = cryptocompare_enhancement.get('prediction_confidence', 'medium')
        confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "🟡")
        st.info(f"{confidence_color} **Prediction Confidence**: {confidence.title()}")
        
        # Risk factors
        if cryptocompare_enhancement.get('risk_factors'):
            st.subheader(f"⚠️ {get_text('risk_factors_header', current_lang)}")
            for risk in cryptocompare_enhancement['risk_factors']:
                st.warning(f"• {risk}")
        
        # Opportunities
        if cryptocompare_enhancement.get('opportunities'):
            st.subheader(f"✅ {get_text('market_opportunities_header', current_lang)}")
            for opportunity in cryptocompare_enhancement['opportunities']:
                st.success(f"• {opportunity}")
        
        st.caption(f"💡 {get_text('cryptocompare_data_enhances', current_lang).format(base_symbol)}")

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
            name=get_text("bollinger_upper", current_lang), line=dict(color='gray', dash='dot'), opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=df.index[-100:], y=df['bb_lower'].iloc[-100:], 
            name=get_text("bollinger_lower", current_lang), line=dict(color='gray', dash='dot'), opacity=0.5,
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
        ))

    chart_title = f"{symbol} {get_text('price_forecast_for', current_lang)} {selected_horizon_display}"
    if market_type == "futures":
        chart_title += f" {get_text('futures', current_lang)}"
    
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
        st.subheader(f"📊 {get_text('macd_indicator', current_lang)}")
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['macd'], 
            name=get_text("macd", current_lang), line=dict(color='blue')
        ))
        fig_macd.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['macd_signal'], 
            name=get_text("signal", current_lang), line=dict(color='red')
        ))
        fig_macd.add_trace(go.Bar(
            x=recent_data.index, y=recent_data['macd_histogram'], 
            name=get_text("histogram", current_lang), opacity=0.6
        ))
        
        fig_macd.update_layout(
            title=get_text("macd_analysis", current_lang),
            xaxis_title=get_text("date", current_lang),
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Stochastic Oscillator
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        st.subheader(f"📈 {get_text('stochastic_oscillator', current_lang)}")
        fig_stoch = go.Figure()
        
        fig_stoch.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['stoch_k'], 
            name=get_text("fast", current_lang), line=dict(color='blue', width=2)
        ))
        fig_stoch.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['stoch_d'], 
            name=get_text("slow", current_lang), line=dict(color='red', width=2)
        ))
        
        # Add overbought/oversold levels
        fig_stoch.add_hline(y=80, line_dash="dash", line_color="gray", 
                           annotation_text=f"{get_text('overbought', current_lang)} (80)")
        fig_stoch.add_hline(y=20, line_dash="dash", line_color="gray", 
                           annotation_text=f"{get_text('oversold', current_lang)} (20)")
        fig_stoch.add_hline(y=50, line_dash="dot", line_color="lightgray", 
                           annotation_text=f"{get_text('midline', current_lang)} (50)", opacity=0.5)
        
        # Fill areas for overbought/oversold zones
        fig_stoch.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.1, 
                           annotation_text=get_text("overbought_zone", current_lang), annotation_position="top left")
        fig_stoch.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.1, 
                           annotation_text=get_text("oversold_zone", current_lang), annotation_position="bottom left")
        
        fig_stoch.update_layout(
            title=f"{get_text('stochastic_oscillator', current_lang)} (%K and %D)",
            xaxis_title=get_text("date", current_lang),
            yaxis_title=f"{get_text('value', current_lang)} (%)",
            template='plotly_dark',
            height=400,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_stoch, use_container_width=True)
        
        # Add interpretation
        latest_k = recent_data['stoch_k'].iloc[-1]
        latest_d = recent_data['stoch_d'].iloc[-1]
        
        if latest_k > 80 and latest_d > 80:
            st.warning(f"⚠️ {get_text('signal_asset_overbought', current_lang)}")
        elif latest_k < 20 and latest_d < 20:
            st.success(f"💡 {get_text('signal_asset_oversold', current_lang)}")
        elif latest_k > latest_d and latest_k > 50:
            st.info(f"📈 {get_text('signal_bullish', current_lang)}")
        elif latest_k < latest_d and latest_k < 50:
            st.info(f"📉 {get_text('signal_bearish', current_lang)}")
        else:
            st.info(f"➡️ {get_text('status_neutral', current_lang)}")
        
        st.caption(f"{get_text('current_values', current_lang)}: %K = {latest_k:.2f}, %D = {latest_d:.2f}")
    
    # Ichimoku Cloud Analysis
    if all(col in df.columns for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']):
        st.subheader(f"☁️ {get_text('ichimoku_analysis', current_lang)}")
        
        # Create Ichimoku chart
        fig_ichimoku = go.Figure()
        
        # Add price candlesticks
        fig_ichimoku.add_trace(go.Candlestick(
            x=recent_data.index,
            open=recent_data['open'],
            high=recent_data['high'],
            low=recent_data['low'],
            close=recent_data['close'],
            name=get_text("historical_price", current_lang),
            opacity=0.8
        ))
        
        # Add Tenkan-sen (Conversion Line)
        fig_ichimoku.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['tenkan_sen'],
            name=get_text("tenkan_sen", current_lang), 
            line=dict(color='red', width=1),
            opacity=0.8
        ))
        
        # Add Kijun-sen (Base Line)
        fig_ichimoku.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['kijun_sen'],
            name=get_text("kijun_sen", current_lang), 
            line=dict(color='blue', width=2),
            opacity=0.8
        ))
        
        # Add Chikou Span (Lagging Span)
        valid_chikou = recent_data.dropna(subset=['chikou_span'])
        if not valid_chikou.empty:
            fig_ichimoku.add_trace(go.Scatter(
                x=valid_chikou.index, y=valid_chikou['chikou_span'],
                name=get_text("chikou_span", current_lang), 
                line=dict(color='purple', width=1, dash='dot'),
                opacity=0.7
            ))
        
        # Add Senkou Span A (Leading Span A)
        valid_span_a = recent_data.dropna(subset=['senkou_span_a'])
        if not valid_span_a.empty:
            fig_ichimoku.add_trace(go.Scatter(
                x=valid_span_a.index, y=valid_span_a['senkou_span_a'],
                name=get_text("senkou_span_a", current_lang), 
                line=dict(color='green', width=1),
                opacity=0.6
            ))
        
        # Add Senkou Span B (Leading Span B)
        valid_span_b = recent_data.dropna(subset=['senkou_span_b'])
        if not valid_span_b.empty:
            fig_ichimoku.add_trace(go.Scatter(
                x=valid_span_b.index, y=valid_span_b['senkou_span_b'],
                name=get_text("senkou_span_b", current_lang), 
                line=dict(color='orange', width=1),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                opacity=0.6
            ))
        
        # Color the cloud based on bullish/bearish
        if 'cloud_color' in recent_data.columns:
            bullish_periods = recent_data[recent_data['cloud_color'] == 1]
            bearish_periods = recent_data[recent_data['cloud_color'] == -1]
            
            # Add bullish cloud sections
            if not bullish_periods.empty and 'senkou_span_a' in bullish_periods.columns and 'senkou_span_b' in bullish_periods.columns:
                valid_bullish = bullish_periods.dropna(subset=['senkou_span_a', 'senkou_span_b'])
                if not valid_bullish.empty:
                    fig_ichimoku.add_trace(go.Scatter(
                        x=valid_bullish.index, y=valid_bullish['senkou_span_a'],
                        fill='tonexty', fillcolor='rgba(0,255,0,0.1)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name=get_text("bullish_cloud", current_lang),
                        showlegend=False
                    ))
            
            # Add bearish cloud sections
            if not bearish_periods.empty and 'senkou_span_a' in bearish_periods.columns and 'senkou_span_b' in bearish_periods.columns:
                valid_bearish = bearish_periods.dropna(subset=['senkou_span_a', 'senkou_span_b'])
                if not valid_bearish.empty:
                    fig_ichimoku.add_trace(go.Scatter(
                        x=valid_bearish.index, y=valid_bearish['senkou_span_a'],
                        fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name=get_text("bearish_cloud", current_lang),
                        showlegend=False
                    ))
        
        fig_ichimoku.update_layout(
            title=get_text("ichimoku_analysis", current_lang),
            xaxis_title=get_text("date", current_lang),
            yaxis_title=get_text("price_usdt", current_lang),
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig_ichimoku, use_container_width=True)
        
        # Ichimoku Analysis and Signals
        latest_data = recent_data.iloc[-1]
        if pd.notna(latest_data.get('tenkan_sen')) and pd.notna(latest_data.get('kijun_sen')):
            latest_price = latest_data['close']
            latest_tenkan = latest_data['tenkan_sen']
            latest_kijun = latest_data['kijun_sen']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(get_text("tenkan_sen", current_lang), f"${latest_tenkan:.4f}")
                st.metric(get_text("kijun_sen", current_lang), f"${latest_kijun:.4f}")
            
            with col2:
                if 'cloud_top' in latest_data and 'cloud_bottom' in latest_data:
                    if pd.notna(latest_data['cloud_top']) and pd.notna(latest_data['cloud_bottom']):
                        cloud_top = latest_data['cloud_top']
                        cloud_bottom = latest_data['cloud_bottom']
                        st.metric(f"{get_text('kumo_cloud', current_lang)} {get_text('high', current_lang)}", f"${cloud_top:.4f}")
                        st.metric(f"{get_text('kumo_cloud', current_lang)} {get_text('low', current_lang)}", f"${cloud_bottom:.4f}")
                
                        # Determine price position relative to cloud
                        if latest_price > cloud_top:
                            price_position = get_text("price_above_cloud", current_lang)
                            position_color = "success"
                        elif latest_price < cloud_bottom:
                            price_position = get_text("price_below_cloud", current_lang)
                            position_color = "error"
                        else:
                            price_position = get_text("price_in_cloud", current_lang)
                            position_color = "warning"
                        
                        st.write(f"**{get_text('current_price', current_lang)}**: {price_position}")
            
            with col3:
                # Tenkan-Kijun Cross Analysis
                if latest_tenkan > latest_kijun:
                    cross_signal = get_text("golden_cross", current_lang)
                    cross_color = "success"
                else:
                    cross_signal = get_text("death_cross", current_lang)
                    cross_color = "error"
                
                st.write(f"**{get_text('tenkan_kijun_cross', current_lang)}**: {cross_signal}")
                
                # Cloud color
                if 'cloud_color' in latest_data and pd.notna(latest_data['cloud_color']):
                    if latest_data['cloud_color'] == 1:
                        cloud_trend = get_text("bullish_cloud", current_lang)
                    else:
                        cloud_trend = get_text("bearish_cloud", current_lang)
                    st.write(f"**{get_text('kumo_cloud', current_lang)}**: {cloud_trend}")
            
            # Generate Ichimoku trading signal
            ichimoku_signal = ""
            if 'price_vs_cloud' in latest_data and pd.notna(latest_data['price_vs_cloud']):
                price_vs_cloud = latest_data['price_vs_cloud']
                
                if price_vs_cloud == 1 and latest_tenkan > latest_kijun:  # Above cloud + bullish cross
                    ichimoku_signal = get_text("ichimoku_signal_bullish", current_lang)
                    st.success(ichimoku_signal)
                elif price_vs_cloud == -1 and latest_tenkan < latest_kijun:  # Below cloud + bearish cross
                    ichimoku_signal = get_text("ichimoku_signal_bearish", current_lang)
                    st.error(ichimoku_signal)
                elif price_vs_cloud == 0:  # Inside cloud
                    ichimoku_signal = get_text("ichimoku_signal_consolidation", current_lang)
                    st.warning(ichimoku_signal)
                else:
                    ichimoku_signal = get_text("ichimoku_signal_neutral", current_lang)
                    st.info(ichimoku_signal)
            
            # Additional insights
            st.caption("💡 Ichimoku Cloud provides comprehensive trend analysis combining momentum, support/resistance, and future trend projection.")

except Exception as e:
    logging.error(f"Application error: {str(e)}")
    st.error(f"{get_text('error_occurred', current_lang)} {str(e)}")
    st.error(get_text("try_refresh", current_lang)) 