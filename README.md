# Cryptocurrency Price Prediction

A machine learning application that predicts cryptocurrency prices using historical data from Bybit and LSTM neural networks. The application features a Streamlit interface for interactive analysis and real-time predictions.

## Features

- **Dual Market Support**: Both spot and futures market prediction
- Real-time cryptocurrency price data fetching from Bybit
- **Enhanced Technical Analysis**: 
  - Spot: SMA, RSI indicators
  - Futures: SMA, EMA, MACD, RSI, Bollinger Bands, ATR, Volume analysis
- **Advanced LSTM Models**: 
  - Standard LSTM for spot markets
  - Enhanced multi-layer LSTM for futures with additional features
- Interactive Streamlit web interface with market type selection
- Historical data visualization with futures-specific charts
- **Futures-Specific Features**:
  - Risk/Reward ratio calculation
  - Volatility analysis
  - Volume ratio indicators
  - MACD analysis charts
- Trading signals based on predictions
- Multi-language support (English and Ukrainian)
- Dynamic language switching

## Prerequisites

- Python 3.10 or later
- Bybit account with API credentials
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd cryptocurrency-price-prediction
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The project uses specific package versions for compatibility:
- pybit==2.4.1 (Bybit API client)
- streamlit==1.31.0
- tensorflow==2.15.0
- pandas==2.2.0
- numpy==1.26.0

4. Create a `.env` file in the project root and add your Bybit API credentials:
```
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
```

To get your API credentials:
1. Log in to your Bybit account
2. Go to Account Settings → API Management
3. Create a new API key pair with "Read-only" permissions (since we only fetch data)
4. Copy the API key and secret to your `.env` file

## Usage

1. Start the Streamlit application:

**For the enhanced version with futures support:**
```bash
streamlit run app_enhanced.py
```

**For the original spot-only version:**
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. In the application:
   - Select your preferred language (English or Ukrainian) from the sidebar
   - **Choose market type**: Spot Market or Futures Market
   - Select a cryptocurrency pair (e.g., BTCUSDT)
   - Choose the prediction horizon (1 hour to 1 day)
   - Choose the historical data period
   - Adjust the trading signal threshold
   - View price predictions and technical indicators
   - **For futures**: Additional risk metrics and MACD analysis

## Project Structure

- `app.py`: Original Streamlit web application (spot markets only)
- `app_enhanced.py`: **Enhanced application with futures support**
- `data_loader.py`: Handles spot market data fetching and preprocessing
- `futures_data_loader.py`: **Futures market data fetching with advanced indicators**
- `predictor.py`: Standard LSTM model for spot markets
- `futures_predictor.py`: **Enhanced LSTM model for futures markets**
- `translations.py`: Multi-language support system
- `requirements.txt`: Project dependencies
- `.env`: API credentials (not tracked in git)

## Technical Details

### Spot Markets
- Uses Bybit's spot API for market data
- Implements 1-hour OHLCV data resampling from trade records
- Features standard LSTM neural network for time series prediction
- Includes basic technical indicators:
  - Simple Moving Averages (20 and 50 periods)
  - Relative Strength Index (14 periods)
  - Price change percentage

### Futures Markets
- Uses Bybit's linear futures API for enhanced market data
- **Enhanced LSTM Architecture**: Multi-layer model with batch normalization and dropout
- **Advanced Technical Indicators**:
  - Simple Moving Averages (SMA 20, 50)
  - Exponential Moving Averages (EMA 12, 26)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands with width calculation
  - ATR (Average True Range) for volatility
  - Volume analysis and ratios
  - High-Low spread analysis
- **Risk Management Features**:
  - Risk/Reward ratio calculation
  - Volatility assessment
  - Directional accuracy measurement
  - Monte Carlo dropout for prediction confidence intervals

## Multi-Language Support

The application supports multiple languages through a centralized translation system:

- **Supported Languages**: English (en) and Ukrainian (uk)
- **Language Switcher**: Available in the sidebar for instant language switching
- **Translation Coverage**: All UI elements, messages, and chart labels are translated
- **Session Persistence**: Language preference is maintained during the session

### Adding New Languages

To add support for additional languages:

1. Open `translations.py`
2. Add a new language code to the `TRANSLATIONS` dictionary
3. Translate all text keys for the new language
4. Update the language selector in `app.py` if needed

Example structure:
```python
TRANSLATIONS = {
    "en": {"key": "English text"},
    "uk": {"key": "Ukrainian text"},
    "new_lang": {"key": "New language text"}
}
```

## Known Limitations

1. API Rate Limits:
   - Bybit API has rate limits that may affect data fetching
   - Large historical data requests may be throttled

2. Data Availability:
   - Some trading pairs may have limited historical data
   - Market data might be delayed during high volatility

3. Prediction Accuracy:
   - The model is for educational purposes only
   - Predictions should not be used for actual trading decisions

## Troubleshooting

1. API Connection Issues:
   - Verify your API credentials in the `.env` file
   - Ensure you're using correct trading pair symbols (e.g., 'BTCUSDT')
   - Check your internet connection
   - If you see "JSON decode error", wait a few minutes and try again (API rate limit)

2. Data Loading Issues:
   - Make sure you have selected a valid date range
   - Verify the trading pair is active on Bybit
   - Check the console for detailed error messages
   - Try reducing the time range if loading large amounts of data

3. Model Training Issues:
   - Ensure sufficient historical data is available (minimum 1000 data points recommended)
   - Check system memory availability
   - Verify data preprocessing steps completed successfully

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Futures Trading Considerations

When using the futures market features, please be aware of additional risks:

- **Leverage Risk**: Futures contracts often involve leverage, amplifying both gains and losses
- **Funding Rates**: Perpetual futures have funding rates that can affect profitability
- **Liquidation Risk**: Leveraged positions can be liquidated if margin requirements aren't met
- **Market Volatility**: Futures markets can be more volatile than spot markets
- **Advanced Indicators**: The additional technical indicators provide more data but require understanding

## Disclaimer

This software is for educational purposes only. Do not use it for financial decisions. Cryptocurrency trading carries significant risks, especially in futures markets. The predictions made by this application are not financial advice and may be inaccurate. Users should conduct their own research and consult with financial professionals before making any investment decisions. 

**Futures trading involves additional risks including leverage, liquidation, and funding costs. Never trade with money you cannot afford to lose.** 