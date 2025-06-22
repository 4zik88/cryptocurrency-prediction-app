# 🚀 Crypto AI Prediction App

Enhanced cryptocurrency prediction application with SPOT and FUTURES market support.

## 📋 Features

- 📈 **SPOT Markets** - Regular cryptocurrency trading analysis
- 🚀 **FUTURES Markets** - Futures trading analysis and predictions  
- 🤖 **AI Predictions** - LSTM-based predictions for 1h, 4h, 8h, 24h horizons
- 📊 **Technical Analysis** - MACD, RSI, Bollinger Bands, and more
- 🌍 **Multi-language** - Ukrainian and English support
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
```

## 🌐 Access

- **Local**: http://localhost:8501
- **Public**: Generated ngrok URL (displayed on startup)

## 📊 Supported Markets

- **SPOT**: BTC, ETH, and 600+ cryptocurrency pairs
- **FUTURES**: Major cryptocurrency futures contracts

---
*Built with Streamlit, TensorFlow, and Bybit API* 