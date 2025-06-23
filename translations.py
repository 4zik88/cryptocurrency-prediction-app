import streamlit as st

# Translation dictionaries
TRANSLATIONS = {
    "en": {
        # Page config
        "page_title": "Crypto Price Prediction",
        
        # Sidebar
        "configuration": "Configuration",
        "select_cryptocurrency": "Select Cryptocurrency",
        "select_market_type": "Select Market Type",
        "spot_market": "Spot Market",
        "futures_market": "Futures Market",
        "refresh_pairs": "🔄 Refresh Pairs",
        "select_prediction_horizon": "Select Prediction Horizon",
        "historical_data_days": "Historical Data (days)",
        "signal_threshold": "Signal Threshold (%)",
        "language": "Language",
        "reset_filters": "Reset Filters",
        
        # Time horizons
        "1_hour": "1 Hour",
        "4_hours": "4 Hours", 
        "8_hours": "8 Hours",
        "12_hours": "12 Hours",
        "1_day": "1 Day (24H)",
        
        # Main content
        "price_prediction_title": "Price Prediction for the Next",
        "futures_prediction_title": "Futures Price Prediction for the Next",
        "prediction_summary": "Prediction Summary",
        "futures_metrics": "Futures Metrics",
        "current_price": "Current Price",
        "predicted_low": "Predicted Low",
        "predicted_high": "Predicted High",
        "trading_signal": "Trading Signal",
        "price_forecast_chart": "Price Forecast Chart",
        "risk_reward_ratio": "Risk/Reward Ratio",
        "atr": "Average True Range",
        "volatility": "Volatility",
        "volume_ratio": "Volume Ratio",
        
        # Trading signals
        "go_long": "**Go Long** 🟢",
        "go_short": "**Go Short** 🔴", 
        "hold_neutral": "**Hold/Neutral** ⚪",
        "predicted_peak_gain": "Predicted peak gain of",
        "predicted_potential_drop": "Predicted potential drop of",
        "no_significant_movement": "No significant price movement predicted.",
        
        # Chart labels
        "historical_price": "Historical Price",
        "predicted_price_path": "Predicted Price Path",
        "date": "Date",
        "price_usdt": "Price (USDT)",
        "price_forecast_for": "Price Forecast for the Next",
        
        # Status messages
        "fetching_data": "Fetching data and preparing for {} forecast...",
        "training_model": "Training model for {} horizon... This might take a moment.",
        "generating_predictions": "Generating predictions...",
        
        # Error messages
        "failed_initialize_dataloader": "Failed to initialize DataLoader:",
        "failed_fetch_data": "Failed to fetch data. Please check symbol or network connection.",
        "insufficient_data": "Insufficient data for prediction. Try increasing the historical data period.",
        "error_occurred": "An error occurred:",
        "try_refresh": "Please try refreshing the page or selecting different parameters."
    },
    
    "uk": {
        # Page config
        "page_title": "Прогнозування цін криптовалют",
        
        # Sidebar
        "configuration": "Налаштування",
        "select_cryptocurrency": "Оберіть криптовалюту",
        "select_market_type": "Оберіть тип ринку",
        "spot_market": "Спотовий ринок",
        "futures_market": "Ф'ючерсний ринок",
        "refresh_pairs": "🔄 Оновити пари",
        "select_prediction_horizon": "Оберіть горизонт прогнозування",
        "historical_data_days": "Історичні дані (дні)",
        "signal_threshold": "Поріг сигналу (%)",
        "language": "Мова",
        "reset_filters": "Скинути фільтри",
        
        # Time horizons
        "1_hour": "1 година",
        "4_hours": "4 години",
        "8_hours": "8 годин",
        "12_hours": "12 годин",
        "1_day": "1 день (24г)",
        
        # Main content
        "price_prediction_title": "Прогноз ціни на наступні",
        "futures_prediction_title": "Прогноз ціни ф'ючерсів на наступні",
        "prediction_summary": "Підсумок прогнозу",
        "futures_metrics": "Метрики ф'ючерсів",
        "current_price": "Поточна ціна",
        "predicted_low": "Прогнозований мінімум",
        "predicted_high": "Прогнозований максимум",
        "trading_signal": "Торговий сигнал",
        "price_forecast_chart": "Графік прогнозу ціни",
        "risk_reward_ratio": "Співвідношення ризик/прибуток",
        "atr": "Середній істинний діапазон",
        "volatility": "Волатильність",
        "volume_ratio": "Співвідношення обсягу",
        
        # Trading signals
        "go_long": "**Купувати** 🟢",
        "go_short": "**Продавати** 🔴",
        "hold_neutral": "**Утримувати/Нейтрально** ⚪",
        "predicted_peak_gain": "Прогнозований пік зростання",
        "predicted_potential_drop": "Прогнозоване потенційне падіння",
        "no_significant_movement": "Значних змін ціни не прогнозується.",
        
        # Chart labels
        "historical_price": "Історична ціна",
        "predicted_price_path": "Прогнозований шлях ціни",
        "date": "Дата",
        "price_usdt": "Ціна (USDT)",
        "price_forecast_for": "Прогноз ціни на наступні",
        
        # Status messages
        "fetching_data": "Завантаження даних та підготовка до прогнозу на {}...",
        "training_model": "Навчання моделі для горизонту {}... Це може зайняти деякий час.",
        "generating_predictions": "Генерація прогнозів...",
        
        # Error messages
        "failed_initialize_dataloader": "Не вдалося ініціалізувати DataLoader:",
        "failed_fetch_data": "Не вдалося завантажити дані. Перевірте символ або мережеве з'єднання.",
        "insufficient_data": "Недостатньо даних для прогнозування. Спробуйте збільшити період історичних даних.",
        "error_occurred": "Сталася помилка:",
        "try_refresh": "Будь ласка, спробуйте оновити сторінку або вибрати інші параметри."
    }
}

def get_text(key: str, lang: str = "en", *args, **kwargs) -> str:
    """
    Get translated text for the given key and language.
    
    Args:
        key: Translation key
        lang: Language code ('en' or 'uk')
        *args: Positional arguments for string formatting
        **kwargs: Keyword arguments for string formatting
    
    Returns:
        Translated text
    """
    try:
        text = TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))
        if args or kwargs:
            return text.format(*args, **kwargs)
        return text
    except Exception:
        return TRANSLATIONS["en"].get(key, key)

def init_language():
    """Initialize language selection in session state."""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'

def get_current_language():
    """Get current language from session state."""
    return st.session_state.get('language', 'en')

def set_language(lang: str):
    """Set language in session state."""
    st.session_state.language = lang 