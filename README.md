# 🚀 Crypto AI Prediction App

Enhanced cryptocurrency prediction application with SPOT and FUTURES market support, now featuring **CryptoCompare Market Intelligence** for comprehensive analysis of ALL cryptocurrencies.

## 📋 Features

- 📈 **SPOT Markets** - Regular cryptocurrency trading analysis
- 🚀 **FUTURES Markets** - Futures trading analysis and predictions  
- 🤖 **AI Predictions** - LSTM-based predictions for 1h, 4h, 8h, 24h horizons
- 📊 **Technical Analysis** - MACD, RSI, Bollinger Bands, and more
- 🕯️ **Candlestick Pattern Recognition** - AI-powered detection of 61+ candlestick patterns
- 🌐 **CryptoCompare Integration** - Enhanced predictions for ALL cryptocurrencies with comprehensive market data
- 🧠 **Advanced Sentiment Analysis** - Multi-factor sentiment scoring for every cryptocurrency
- 🔍 **Multi-Source Validation** - Price comparison and cross-validation across data sources
- 🌍 **Multi-language** - Russian and English support
- 🌐 **Public Access** - ngrok tunnel for sharing

## 🏃‍♂️ Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Configuration**
   - Create `.streamlit/secrets.toml` with your Bybit API credentials
   
3. **Run Application**
   ```bash
   python3 run_enhanced_app.py
   ```

## 📁 Core Files

- `run_enhanced_app.py` - Main application runner
- `app_enhanced.py` - Streamlit web application
- `data_loader.py` - SPOT market data handling
- `futures_data_loader.py` - FUTURES market data handling
- `cryptocompare_data_loader.py` - CryptoCompare API integration for universal cryptocurrency analysis
- `trading_pattern_analyzer.py` - **NEW**: Advanced pattern recognition with 61+ candlestick patterns
- `predictor.py` - LSTM predictor for SPOT markets
- `futures_predictor.py` - Enhanced LSTM predictor for FUTURES
- `translations.py` - Multi-language support
- `TALIB_INSTALLATION.md` - **NEW**: TA-Lib installation guide for pattern recognition

## 🔧 Configuration

Create `.streamlit/secrets.toml`:
```toml
[bybit]
api_key = "your_api_key"
api_secret = "your_api_secret"
testnet = true

# CryptoCompare API key for enhanced market intelligence
CRYPTOCOMPARE_API_KEY = "your_cryptocompare_api_key"
```

## 🌐 Access

- **Local**: http://localhost:8501
- **Public**: Generated ngrok URL (displayed on startup)

## 📊 Supported Markets

- **SPOT**: BTC, ETH, and 600+ cryptocurrency pairs
- **FUTURES**: Major cryptocurrency futures contracts

## 🆕 CryptoCompare Integration Features

### For ALL Cryptocurrencies (BTC, ETH, ADA, SOL, and 1000+ others):
- 🌐 **Universal Coverage** - Works with every cryptocurrency in spot and futures markets
- 📊 **Real-time Market Metrics** - Price, volume, market cap, 24h changes for all coins
- 🧠 **Advanced Sentiment Analysis** - Multi-factor sentiment scoring (0-100 scale)
- 📈 **Social Sentiment Tracking** - Reddit, Twitter, Facebook activity monitoring
- 📰 **News Sentiment Analysis** - Real-time news sentiment with ML scoring
- 🔍 **Price Validation** - Cross-validation between exchange and CryptoCompare data
- 🎯 **Enhanced Prediction Confidence** - Sentiment-adjusted confidence scoring
- ⚠️ **Risk & Opportunity Detection** - Automated market risk and opportunity identification

### Multi-Factor Sentiment Analysis:
- **Price Momentum** (40% weight) - 24h price movement and trend direction
- **Volume Strength** (20% weight) - Trading volume and liquidity indicators
- **Social Activity** (15% weight) - Reddit posts, Twitter mentions, social engagement
- **News Sentiment** (15% weight) - Recent news articles with sentiment scoring
- **Technical Indicators** (10% weight) - RSI, moving averages, trend analysis

### Technical Features:
- Professional CryptoCompare API integration with authentication
- Supports 1000+ cryptocurrencies across all major exchanges
- Real-time social media and news sentiment analysis
- Advanced sentiment scoring with confidence levels
- Historical data analysis for trend identification
- Multi-source data validation for improved accuracy

### Setup:
Add your CryptoCompare API key to environment variables:
```bash
export CRYPTOCOMPARE_API_KEY="your_api_key_here"
```

## 🕯️ Candlestick Pattern Recognition

Advanced AI-powered candlestick pattern recognition system using TA-Lib for professional technical analysis.

### Features:
- **61+ Pattern Detection** - Complete library of candlestick patterns including:
  - **Reversal Patterns**: Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star
  - **Continuation Patterns**: Three White Soldiers, Three Black Crows, Rising/Falling Three Methods
  - **Indecision Patterns**: Spinning Top, High-Wave Candle, Long-Legged Doji
- **Visual Pattern Markers** - Patterns highlighted directly on price charts with color coding
- **Market Sentiment Analysis** - AI-powered sentiment scoring based on detected patterns
- **Pattern Classification** - Automatic categorization by bullish/bearish and reversal/continuation
- **Educational Insights** - Learn about each pattern as it's detected
- **Trading Signals** - Enhanced signals combining patterns with volume and support/resistance

### Pattern Categories:
- 🟢 **Bullish Reversal** - Patterns suggesting upward price reversals
- 🔴 **Bearish Reversal** - Patterns suggesting downward price reversals  
- ➡️ **Continuation** - Patterns suggesting trend continuation
- ⚪ **Indecision** - Patterns showing market uncertainty
- 🔄 **General Reversal** - Patterns that can be bullish or bearish

### Technical Implementation:
- **TA-Lib Integration** - Professional-grade pattern recognition library
- **Real-time Detection** - Patterns detected on every data update
- **Historical Analysis** - Pattern frequency and reliability tracking
- **Chart Integration** - Seamlessly integrated with existing technical analysis charts
- **Performance Optimized** - Efficient pattern scanning with caching

### Installation:
The candlestick pattern recognition requires TA-Lib installation. See `TALIB_INSTALLATION.md` for detailed setup instructions.

```bash
# Quick install (macOS with Homebrew)
brew install ta-lib
pip install TA-Lib

# For other systems, see TALIB_INSTALLATION.md
```

### Usage:
1. Enable "Candlestick Pattern Analysis" in the sidebar
2. Patterns will be automatically detected and displayed
3. View pattern categories in organized tabs
4. Check the enhanced chart with pattern markers
5. Read educational insights about detected patterns

## Language Support

This application now supports **full internationalization** with proper language separation:

### Supported Languages
- **English** (`en`) - Complete interface in English
- **Russian** (`ru`) - Complete interface in Russian

### Language Features
- Dynamic language switching in the app interface
- Console output language can be configured via environment variable
- No mixed language content - clean separation between English and Russian
- All user-facing text is properly translated

### Console Language Configuration
You can set the console language for the runner script:
```bash
# For English console output
export CONSOLE_LANGUAGE=en
python3 run_enhanced_app.py

# For Russian console output (default)
export CONSOLE_LANGUAGE=ru
python3 run_enhanced_app.py
```

## Development

The application uses Streamlit for the web interface and includes comprehensive language support through the `translations.py` module.

---
*Built with Streamlit, TensorFlow, Bybit API, and CryptoCompare Market Intelligence* 