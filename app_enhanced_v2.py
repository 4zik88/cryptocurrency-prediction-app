import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from data_loader import DataLoader
from enhanced_predictor import EnhancedCryptoPredictor
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
    page_title=get_text("page_title", current_lang) + " - Enhanced",
    page_icon="🚀",
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
st.sidebar.title("🚀 Enhanced Crypto Predictor")

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

# Enhanced Parameters
st.sidebar.subheader("📊 Enhanced Features")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Enhanced Ensemble", "Basic LSTM"]
)

# Use enhanced features
use_enhanced_features = st.sidebar.checkbox("Use Enhanced Technical Indicators", value=True)
use_market_regime = st.sidebar.checkbox("Market Regime Detection", value=True)
use_correlation = st.sidebar.checkbox("Correlation Analysis", value=True)
use_uncertainty_quantification = st.sidebar.checkbox("Uncertainty Quantification", value=True)

# Model instantiation
@st.cache_resource
def get_predictor(horizon, enhanced=True):
    if enhanced:
        return EnhancedCryptoPredictor(n_future_steps=horizon)
    else:
        from predictor import LSTMPredictor
        return LSTMPredictor(n_future_steps=horizon)

predictor = get_predictor(n_future_steps, enhanced=(model_type == "Enhanced Ensemble"))

# Other parameters
lookback_days = st.sidebar.slider(get_text("historical_data_days", current_lang), min_value=30, max_value=365, value=180)
threshold = st.sidebar.slider(get_text("signal_threshold", current_lang), min_value=0.1, max_value=10.0, value=2.0, step=0.1)

# Risk management parameters
st.sidebar.subheader("⚠️ Risk Management")
show_risk_metrics = st.sidebar.checkbox("Show Risk Metrics", value=True)
confidence_level = st.sidebar.slider("Confidence Level (%)", min_value=80, max_value=99, value=95)

# --- Main Content ---
st.title(f"🚀 {symbol} Enhanced Price Prediction - {selected_horizon_label}")

def generate_enhanced_trading_signal(current_price, uncertainty_result, threshold, lang):
    """Generate enhanced trading signal with uncertainty."""
    try:
        if uncertainty_result is None:
            return "Hold", get_text("no_significant_movement", lang), {}
        
        predictions = uncertainty_result['mean_prediction']
        lower_bound = uncertainty_result[f'confidence_{confidence_level}_lower']
        upper_bound = uncertainty_result[f'confidence_{confidence_level}_upper']
        prediction_std = uncertainty_result['lstm_std']
        
        max_predicted_price = np.max(predictions)
        min_predicted_price = np.min(predictions)
        
        upside_potential = (max_predicted_price - current_price) / current_price * 100
        downside_risk = (min_predicted_price - current_price) / current_price * 100
        
        # Calculate confidence-adjusted signal
        avg_std = np.mean(prediction_std)
        confidence_penalty = avg_std / current_price * 100  # Penalize high uncertainty
        
        adjusted_upside = upside_potential - confidence_penalty
        adjusted_downside = downside_risk + confidence_penalty
        
        signal_metrics = {
            'upside_potential': upside_potential,
            'downside_risk': downside_risk,
            'confidence_penalty': confidence_penalty,
            'adjusted_upside': adjusted_upside,
            'adjusted_downside': adjusted_downside,
            'prediction_uncertainty': avg_std,
            'confidence_interval_width': np.mean(upper_bound - lower_bound)
        }
        
        if adjusted_upside > threshold:
            return "Long", f"Predicted peak gain: {upside_potential:.2f}% (confidence-adjusted: {adjusted_upside:.2f}%)", signal_metrics
        elif adjusted_downside < -threshold:
            return "Short", f"Predicted potential drop: {abs(downside_risk):.2f}% (confidence-adjusted: {abs(adjusted_downside):.2f}%)", signal_metrics
        else:
            return "Hold", f"No significant movement expected (uncertainty: ±{avg_std:.2f})", signal_metrics
    
    except Exception as e:
        logging.error(f"Error generating enhanced signal: {str(e)}")
        return "Hold", "Error in signal generation", {}

try:
    # 1. Fetch and prepare data
    with st.spinner("🔄 Fetching enhanced data..."):
        df = st.session_state.data_loader.fetch_historical_data(symbol=symbol, lookback_days=lookback_days)
        if df.empty:
            st.error(get_text("failed_fetch_data", current_lang))
            st.stop()
        
        # Add enhanced technical indicators
        if use_enhanced_features:
            df = st.session_state.data_loader.add_technical_indicators(df)
        
        # Add market regime detection
        if use_market_regime:
            df = st.session_state.data_loader.add_market_regime_detection(df)
        
        # Prepare data
        X_train, y_train, X_test, y_test, original_y_test = st.session_state.data_loader.prepare_data(df, n_future_steps=n_future_steps)

        if X_train is None:
            st.error(get_text("insufficient_data", current_lang))
            st.stop()

    # 2. Load or train model
    with st.spinner("🤖 Loading/Training enhanced model..."):
        if not predictor.load_model():
            with st.spinner("🎯 Training enhanced model..."):
                if predictor.train(X_train, y_train):
                    st.success("✅ Model trained successfully!")
                else:
                    st.error("❌ Model training failed!")
                    st.stop()
    
    # 3. Generate predictions
    with st.spinner("🔮 Generating enhanced predictions..."):
        last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
        
        if use_uncertainty_quantification and hasattr(predictor, 'predict_with_uncertainty'):
            uncertainty_result = predictor.predict_with_uncertainty(last_sequence, n_simulations=100)
            predicted_prices = st.session_state.data_loader.inverse_transform_price(uncertainty_result['mean_prediction'][0])
        else:
            predicted_scaled_sequence = predictor.predict(last_sequence)[0]
            predicted_prices = st.session_state.data_loader.inverse_transform_price(predicted_scaled_sequence)
            uncertainty_result = None
    
    current_price = df['close'].iloc[-1]
    
    # 4. Get correlation data
    correlations = {}
    if use_correlation:
        with st.spinner("📊 Calculating correlations..."):
            correlations = st.session_state.data_loader.get_correlation_data(symbol)
    
    # 5. Calculate risk metrics
    risk_metrics = {}
    if show_risk_metrics:
        with st.spinner("⚠️ Calculating risk metrics..."):
            risk_metrics = st.session_state.data_loader.calculate_risk_metrics(predicted_prices, current_price)
    
    # --- Display Enhanced Metrics ---
    st.header("📊 Enhanced Prediction Summary")
    
    # Enhanced metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Current Price", f"${current_price:,.4f}")
        
    with col2:
        predicted_final = predicted_prices[-1]
        change_final = (predicted_final - current_price) / current_price * 100
        st.metric(f"🎯 Exact Forecast ({selected_horizon_label})", 
                 f"${predicted_final:,.4f}", 
                 f"{change_final:+.2f}%")
    
    with col3:
        predicted_avg = np.mean(predicted_prices)
        change_avg = (predicted_avg - current_price) / current_price * 100
        st.metric(f"📊 Average Price ({selected_horizon_label})", 
                 f"${predicted_avg:,.4f}", 
                 f"{change_avg:+.2f}%")
    
    # Detailed step-by-step forecast
    st.subheader("📈 Step-by-Step Forecast")
    
    # Create forecast breakdown
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
        # For single step predictions
        st.info(f"🎯 **Exact Forecast for next {selected_horizon_label}:** ${predicted_final:,.4f} ({change_final:+.2f}%)")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        predicted_low = np.min(predicted_prices)
        change_low = (predicted_low - current_price) / current_price * 100
        st.metric("📉 Minimum Expected", f"${predicted_low:,.4f}", f"{change_low:+.2f}%")
    with col2:
        predicted_high = np.max(predicted_prices)
        change_high = (predicted_high - current_price) / current_price * 100
        st.metric("📈 Maximum Expected", f"${predicted_high:,.4f}", f"{change_high:+.2f}%")
    with col3:
        price_range = predicted_high - predicted_low
        range_pct = (price_range / current_price) * 100
        st.metric("📏 Price Range", f"${price_range:,.4f}", f"{range_pct:.2f}%")
    with col4:
        volatility = np.std(predicted_prices)
        vol_pct = (volatility / current_price) * 100
        st.metric("⚡ Volatility", f"${volatility:,.4f}", f"{vol_pct:.2f}%")
    
    # Enhanced Trading Signal
    st.subheader("🎯 Enhanced Trading Signal")
    signal, signal_reason, signal_metrics = generate_enhanced_trading_signal(current_price, uncertainty_result, threshold, current_lang)
    
    if signal == "Long":
        st.success(f"📈 **GO LONG** - {signal_reason}")
    elif signal == "Short":
        st.error(f"📉 **GO SHORT** - {signal_reason}")
    else:
        st.info(f"⏸️ **HOLD** - {signal_reason}")
    
    # Display signal metrics
    if signal_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Upside Potential", f"{signal_metrics.get('upside_potential', 0):.2f}%")
        with col2:
            st.metric("⚠️ Downside Risk", f"{signal_metrics.get('downside_risk', 0):.2f}%")
        with col3:
            st.metric("🔒 Prediction Uncertainty", f"±{signal_metrics.get('prediction_uncertainty', 0):.2f}")
    
    # --- Risk Metrics ---
    if show_risk_metrics and risk_metrics:
        st.subheader("⚠️ Risk Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR 95%", f"{risk_metrics.get('var_95', 0):.2f}%", 
                     help="Value at Risk at 95% confidence level")
        with col2:
            st.metric("Expected Shortfall", f"{risk_metrics.get('expected_shortfall_95', 0):.2f}%",
                     help="Expected loss in worst 5% scenarios")
        with col3:
            st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.3f}",
                     help="Risk-adjusted return ratio")
        with col4:
            st.metric("Probability of Loss", f"{risk_metrics.get('probability_of_loss', 0):.1f}%")
    
    # --- Correlation Analysis ---
    if correlations:
        st.subheader("🔗 Market Correlations")
        corr_cols = st.columns(len(correlations))
        for i, (coin, corr) in enumerate(correlations.items()):
            if not coin.endswith('_rolling_correlation'):
                with corr_cols[i % len(corr_cols)]:
                    st.metric(coin.replace('_correlation', ''), f"{corr:.3f}")
    
    # --- Enhanced Plotting ---
    st.header("📊 Enhanced Price Forecast Chart")
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['close'], 
        name="Historical Price", 
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Predicted sequence
    future_dates = pd.to_datetime([df.index[-1] + timedelta(hours=i+1) for i in range(n_future_steps)])
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=predicted_prices, 
        name="Predicted Price", 
        line=dict(color='#ff7f0e', dash='dash', width=3)
    ))
    
    # Add uncertainty bands if available
    if uncertainty_result is not None and use_uncertainty_quantification:
        upper_bound = st.session_state.data_loader.inverse_transform_price(uncertainty_result[f'confidence_{confidence_level}_upper'][0])
        lower_bound = st.session_state.data_loader.inverse_transform_price(uncertainty_result[f'confidence_{confidence_level}_lower'][0])
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name=f'{confidence_level}% Confidence',
            fillcolor='rgba(255, 127, 14, 0.2)',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f"{symbol} Enhanced Price Forecast - {selected_horizon_label}",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        template='plotly_dark',
        legend=dict(x=0, y=1),
        hovermode='x unified',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators chart
    if use_enhanced_features:
        st.subheader("📈 Technical Indicators")
        
        # Create subplots for indicators
        indicators_to_show = st.multiselect(
            "Select indicators to display:",
            ['RSI', 'MACD', 'Bollinger Bands', 'Volume', 'ATR'],
            default=['RSI', 'MACD']
        )
        
        if indicators_to_show:
            fig_indicators = go.Figure()
            
            for indicator in indicators_to_show:
                if indicator == 'RSI' and 'rsi' in df.columns:
                    fig_indicators.add_trace(go.Scatter(
                        x=df.index, y=df['rsi'], name='RSI',
                        line=dict(color='purple')
                    ))
                elif indicator == 'MACD' and 'macd' in df.columns:
                    fig_indicators.add_trace(go.Scatter(
                        x=df.index, y=df['macd'], name='MACD',
                        line=dict(color='blue')
                    ))
                    if 'macd_signal' in df.columns:
                        fig_indicators.add_trace(go.Scatter(
                            x=df.index, y=df['macd_signal'], name='MACD Signal',
                            line=dict(color='red')
                        ))
            
            fig_indicators.update_layout(
                title="Technical Indicators",
                xaxis_title="Date",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_indicators, use_container_width=True)
    
    # Model performance metrics
    if hasattr(predictor, 'evaluate_ensemble'):
        st.subheader("🎯 Model Performance")
        with st.spinner("Evaluating model performance..."):
            perf_metrics = predictor.evaluate_ensemble(X_test, y_test)
            if perf_metrics:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{perf_metrics.get('rmse', 0):.6f}")
                with col2:
                    st.metric("MAE", f"{perf_metrics.get('mae', 0):.6f}")
                with col3:
                    st.metric("Coverage 95%", f"{perf_metrics.get('coverage_95', 0):.1%}")
                with col4:
                    st.metric("Model Agreement", f"{perf_metrics.get('mean_model_disagreement', 0):.6f}")

except Exception as e:
    logging.error(f"Application error: {str(e)}")
    st.error(f"⚠️ An error occurred: {str(e)}")
    st.error("Please try refreshing the page or contact support.") 