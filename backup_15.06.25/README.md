# Cryptocurrency Price Prediction

A machine learning application that predicts cryptocurrency prices using historical data from Bybit and LSTM neural networks. The application features a Streamlit interface for interactive analysis and real-time predictions.

## Features

- Real-time cryptocurrency price data fetching from Bybit
- Technical indicators calculation (SMA, RSI)
- LSTM-based price prediction model
- Interactive Streamlit web interface
- Historical data visualization
- Trading signals based on predictions

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
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. In the application:
   - Select a cryptocurrency pair (e.g., BTCUSDT)
   - Choose the historical data period
   - Adjust the trading signal threshold
   - View price predictions and technical indicators

## Project Structure

- `app.py`: Streamlit web application interface
- `data_loader.py`: Handles data fetching from Bybit API and preprocessing
- `predictor.py`: LSTM model implementation and training
- `requirements.txt`: Project dependencies
- `.env`: API credentials (not tracked in git)

## Technical Details

- Uses Bybit's v2 API for market data
- Implements 1-hour OHLCV data resampling from trade records
- Features LSTM neural network for time series prediction
- Includes technical indicators:
  - Simple Moving Averages (20 and 50 periods)
  - Relative Strength Index (14 periods)
  - Price change percentage

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

## Disclaimer

This software is for educational purposes only. Do not use it for financial decisions. Cryptocurrency trading carries significant risks. The predictions made by this application are not financial advice and may be inaccurate. Users should conduct their own research and consult with financial professionals before making any investment decisions. 