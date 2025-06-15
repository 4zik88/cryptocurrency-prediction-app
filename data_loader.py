import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class DataLoader:
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY', '')
        self.api_secret = os.getenv('BYBIT_API_SECRET', '')
        
        if not self.api_key or not self.api_secret:
            logging.error("API credentials not found in .env file")
            raise ValueError("API credentials not found. Please check your .env file.")
            
        try:
            self.client = HTTP(
                testnet=False,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            logging.info("Successfully initialized Bybit client")
        except Exception as e:
            logging.error(f"Failed to initialize Bybit client: {str(e)}")
            raise
            
        self.scaler = MinMaxScaler()
        
    def fetch_historical_data(self, symbol, interval='60', lookback_days=180):
        """Fetch historical OHLCV data from Bybit.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval in minutes ('60' for 1h)
            lookback_days: Number of days to look back
        """
        logging.info(f"Fetching historical data for {symbol}, interval={interval}, lookback_days={lookback_days}")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        try:
            # Convert interval to API format
            interval_map = {
                '1': '1',
                '5': '5',
                '15': '15',
                '30': '30',
                '60': '60',
                '240': '240',
                'D': 'D',
                'W': 'W',
                'M': 'M'
            }
            
            api_interval = interval_map.get(interval, '60')
            
            # Fetch kline data
            response = self.client.get_kline(
                category="spot",
                symbol=symbol,
                interval=api_interval,
                limit=1000,
                start=int(start_time.timestamp() * 1000),
                end=int(end_time.timestamp() * 1000)
            )
            
            if 'retCode' in response and response['retCode'] != 0:
                logging.error(f"API Error: {response.get('retMsg', 'Unknown error')}")
                raise Exception(f"API Error: {response.get('retMsg', 'Unknown error')}")
            
            if not response.get('result', {}).get('list'):
                logging.error("No data returned from API")
                raise Exception("No data returned from API")
            
            # Process the response
            klines = response['result']['list']
            logging.info(f"Received {len(klines)} klines from API")
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'turnover'
            ])
            
            # Convert timestamps to datetime (milliseconds to datetime)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df = df.set_index('timestamp')
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle any missing or invalid data
            df = df.dropna()
            if df.empty:
                logging.error("No valid data after processing")
                raise Exception("No valid data after processing")
            
            logging.info(f"Successfully processed data: {len(df)} rows")
            return df.sort_index()
            
        except Exception as e:
            logging.error(f"Error in fetch_historical_data: {str(e)}")
            return pd.DataFrame()

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe."""
        if df.empty:
            logging.warning("Empty dataframe provided to add_technical_indicators")
            return df
            
        try:
            # Calculate SMA
            sma_20 = SMAIndicator(df['close'], window=20)
            sma_50 = SMAIndicator(df['close'], window=50)
            df['sma_20'] = sma_20.sma_indicator()
            df['sma_50'] = sma_50.sma_indicator()
            
            # Calculate RSI
            rsi = RSIIndicator(df['close'], window=14)
            df['rsi'] = rsi.rsi()
            
            # Add price changes
            df['price_change'] = df['close'].pct_change()
            
            logging.info("Successfully added technical indicators")
            return df
        except Exception as e:
            logging.error(f"Error in add_technical_indicators: {str(e)}")
            return df

    def prepare_data(self, df, sequence_length=24, n_future_steps=1, train_split=0.8):
        """Prepare data for LSTM model."""
        if df.empty:
            logging.warning("Empty dataframe provided to prepare_data")
            return None, None, None, None, None
            
        try:
            # Select features for training
            features = ['close', 'volume', 'sma_20', 'sma_50', 'rsi', 'price_change']
            data = df[features].copy()
            
            # Handle missing values
            data = data.dropna()
            
            if len(data) < sequence_length + n_future_steps:
                logging.error(f"Insufficient data: {len(data)} rows, need at least {sequence_length + n_future_steps}")
                return None, None, None, None, None
            
            # Scale the features
            data_scaled = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            for i in range(len(data_scaled) - sequence_length - n_future_steps + 1):
                X.append(data_scaled[i : i + sequence_length])
                y.append(data_scaled[i + sequence_length : i + sequence_length + n_future_steps, 0])
                
            X, y = np.array(X), np.array(y)
            
            # Split into train and test sets
            train_size = int(len(X) * train_split)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Keep the original unscaled test data for evaluation
            original_y_test = []
            for i in range(train_size, len(data) - sequence_length - n_future_steps + 1):
                original_y_test.append(data['close'].iloc[i + sequence_length : i + sequence_length + n_future_steps].values)
            original_y_test = np.array(original_y_test)

            logging.info(f"Data preparation complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
            return X_train, y_train, X_test, y_test, original_y_test
            
        except Exception as e:
            logging.error(f"Error in prepare_data: {str(e)}")
            return None, None, None, None, None

    def inverse_transform_price(self, scaled_prices):
        """Convert scaled prices back to original scale."""
        # Create a dummy array with the same shape as the scaler expects
        if scaled_prices.ndim == 1:
            scaled_prices = scaled_prices.reshape(-1, 1)
        
        num_features = self.scaler.n_features_in_
        dummy_features = np.zeros((scaled_prices.shape[0], num_features))
        dummy_features[:, 0] = scaled_prices.flatten()
        
        # Inverse transform
        return self.scaler.inverse_transform(dummy_features)[:, 0]

    def get_latest_price(self, symbol):
        """Get the latest price for a symbol."""
        try:
            response = self.client.get_tickers(
                category="spot",
                symbol=symbol
            )
            
            if 'retCode' in response and response['retCode'] != 0:
                logging.error(f"API Error in get_latest_price: {response.get('retMsg', 'Unknown error')}")
                return None
            
            if not response.get('result', {}).get('list'):
                logging.error("No ticker data returned")
                return None
                
            price = float(response['result']['list'][0]['lastPrice'])
            logging.info(f"Latest price for {symbol}: {price}")
            return price
            
        except Exception as e:
            logging.error(f"Error in get_latest_price: {str(e)}")
            return None

    def get_available_pairs(self):
        """Get all available trading pairs from Bybit spot market."""
        try:
            response = self.client.get_tickers(
                category="spot"
            )
            
            if 'retCode' in response and response['retCode'] != 0:
                logging.error(f"API Error in get_available_pairs: {response.get('retMsg', 'Unknown error')}")
                return []
            
            if not response.get('result', {}).get('list'):
                logging.error("No ticker data returned")
                return []
            
            # Extract all available symbols
            pairs = [item['symbol'] for item in response['result']['list']]
            pairs.sort()  # Sort alphabetically
            logging.info(f"Found {len(pairs)} available trading pairs")
            return pairs
            
        except Exception as e:
            logging.error(f"Error in get_available_pairs: {str(e)}")
            return [] 