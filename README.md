# 🚀 Crypto AI Prediction App

Enhanced cryptocurrency prediction application with SPOT and FUTURES market support, now featuring **CryptoCompare Market Intelligence** for comprehensive analysis of ALL cryptocurrencies.

## 📋 Features

- 📈 **SPOT Markets** - Regular cryptocurrency trading analysis
- 🚀 **FUTURES Markets** - Futures trading analysis and predictions  
- 🤖 **AI Predictions** - LSTM-based predictions for 1h, 4h, 8h, 24h horizons
- 📊 **Technical Analysis** - MACD, RSI, Bollinger Bands, and more
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
- `cryptocompare_data_loader.py` - **NEW**: CryptoCompare API integration for universal cryptocurrency analysis
- `predictor.py` - LSTM predictor for SPOT markets
- `futures_predictor.py` - Enhanced LSTM predictor for FUTURES
- `translations.py` - Multi-language support

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

---
*Built with Streamlit, TensorFlow, Bybit API, and CryptoCompare Market Intelligence* 