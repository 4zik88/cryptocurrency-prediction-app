import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class FuturesDataLoader:
    def __init__(self):
        # Try to get API keys from Streamlit secrets first, then environment variables
        try:
            import streamlit as st
            self.api_key = st.secrets.get('BYBIT_API_KEY', '') or os.getenv('BYBIT_API_KEY', '')
            self.api_secret = st.secrets.get('BYBIT_API_SECRET', '') or os.getenv('BYBIT_API_SECRET', '')
        except ImportError:
            # Streamlit not available, use environment variables only
            self.api_key = os.getenv('BYBIT_API_KEY', '')
            self.api_secret = os.getenv('BYBIT_API_SECRET', '')
        except Exception:
            # Fallback to environment variables
            self.api_key = os.getenv('BYBIT_API_KEY', '')
            self.api_secret = os.getenv('BYBIT_API_SECRET', '')
        
        if not self.api_key or not self.api_secret:
            logging.error("API credentials not found. Please check your Streamlit secrets or .env file")
            raise ValueError("API credentials not found. Please check your Streamlit secrets or .env file.")
            
        try:
            self.client = HTTP(
                testnet=False,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            logging.info("Successfully initialized Bybit client for futures")
        except Exception as e:
            logging.error(f"Failed to initialize Bybit client: {str(e)}")
            raise
            
        self.scaler = MinMaxScaler()
        
    def fetch_futures_data(self, symbol, interval='60', lookback_days=180):
        """Fetch historical futures OHLCV data from Bybit.
        
        Args:
            symbol: Futures trading pair (e.g., 'BTCUSDT')
            interval: Kline interval in minutes ('60' for 1h)
            lookback_days: Number of days to look back
        """
        logging.info(f"Fetching futures data for {symbol}, interval={interval}, lookback_days={lookback_days}")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        try:
            import time
            
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
            
            # Add retry logic for rate limiting
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # Add small delay to avoid rate limiting
                    if attempt > 0:
                        time.sleep(retry_delay * attempt)
                    
                    # Fetch futures kline data
                    response = self.client.get_kline(
                        category="linear",  # Linear futures
                        symbol=symbol,
                        interval=api_interval,
                        limit=1000,
                        start=int(start_time.timestamp() * 1000),
                        end=int(end_time.timestamp() * 1000)
                    )
                    
                    # If successful, break out of retry loop
                    break
                    
                except Exception as api_error:
                    if "rate limit" in str(api_error).lower() and attempt < max_retries - 1:
                        logging.warning(f"Rate limit hit, retrying in {retry_delay * (attempt + 1)} seconds...")
                        continue
                    else:
                        raise api_error
            
            if 'retCode' in response and response['retCode'] != 0:
                logging.error(f"API Error: {response.get('retMsg', 'Unknown error')}")
                raise Exception(f"API Error: {response.get('retMsg', 'Unknown error')}")
            
            if not response.get('result', {}).get('list'):
                logging.error("No futures data returned from API")
                raise Exception("No futures data returned from API")
            
            # Process the response
            klines = response['result']['list']
            logging.info(f"Received {len(klines)} futures klines from API")
            
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
                logging.error("No valid futures data after processing")
                raise Exception("No valid futures data after processing")
            
            logging.info(f"Successfully processed futures data: {len(df)} rows")
            return df.sort_index()
            
        except Exception as e:
            logging.error(f"Error in fetch_futures_data: {str(e)}")
            return pd.DataFrame()

    def get_open_interest(self, symbol):
        """Get open interest data for futures contract."""
        try:
            response = self.client.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime="5min",
                limit=200
            )
            
            if 'retCode' in response and response['retCode'] != 0:
                logging.error(f"API Error in get_open_interest: {response.get('retMsg', 'Unknown error')}")
                return pd.DataFrame()
            
            if not response.get('result', {}).get('list'):
                logging.warning("No open interest data returned")
                return pd.DataFrame()
            
            # Convert to DataFrame
            oi_data = response['result']['list']
            df_oi = pd.DataFrame(oi_data)
            
            if not df_oi.empty:
                df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'].astype(float), unit='ms')
                df_oi = df_oi.set_index('timestamp')
                df_oi['openInterest'] = pd.to_numeric(df_oi['openInterest'], errors='coerce')
                
            logging.info(f"Retrieved {len(df_oi)} open interest records")
            return df_oi.sort_index()
            
        except Exception as e:
            logging.error(f"Error in get_open_interest: {str(e)}")
            return pd.DataFrame()

    def get_funding_rate(self, symbol):
        """Get funding rate history for futures contract."""
        try:
            response = self.client.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                limit=200
            )
            
            if 'retCode' in response and response['retCode'] != 0:
                logging.error(f"API Error in get_funding_rate: {response.get('retMsg', 'Unknown error')}")
                return pd.DataFrame()
            
            if not response.get('result', {}).get('list'):
                logging.warning("No funding rate data returned")
                return pd.DataFrame()
            
            # Convert to DataFrame
            funding_data = response['result']['list']
            df_funding = pd.DataFrame(funding_data)
            
            if not df_funding.empty:
                df_funding['fundingRateTimestamp'] = pd.to_datetime(df_funding['fundingRateTimestamp'].astype(float), unit='ms')
                df_funding = df_funding.set_index('fundingRateTimestamp')
                df_funding['fundingRate'] = pd.to_numeric(df_funding['fundingRate'], errors='coerce')
                
            logging.info(f"Retrieved {len(df_funding)} funding rate records")
            return df_funding.sort_index()
            
        except Exception as e:
            logging.error(f"Error in get_funding_rate: {str(e)}")
            return pd.DataFrame()

    def add_futures_indicators(self, df):
        """Add technical indicators specific to futures trading."""
        if df.empty:
            logging.warning("Empty dataframe provided to add_futures_indicators")
            return df
            
        try:
            # Basic technical indicators
            sma_20 = SMAIndicator(df['close'], window=20)
            sma_50 = SMAIndicator(df['close'], window=50)
            ema_12 = EMAIndicator(df['close'], window=12)
            ema_26 = EMAIndicator(df['close'], window=26)
            
            df['sma_20'] = sma_20.sma_indicator()
            df['sma_50'] = sma_50.sma_indicator()
            df['ema_12'] = ema_12.ema_indicator()
            df['ema_26'] = ema_26.ema_indicator()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = EMAIndicator(df['macd'], window=9).ema_indicator()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            rsi = RSIIndicator(df['close'], window=14)
            df['rsi'] = rsi.rsi()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Bollinger Bands
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Price changes and volatility
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # High-Low spread (important for futures)
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            
            # True Range and ATR
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=14).mean()
            
            # Clean up temporary columns
            df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)
            
            logging.info("Successfully added futures technical indicators")
            return df
        except Exception as e:
            logging.error(f"Error in add_futures_indicators: {str(e)}")
            return df

    def prepare_futures_data(self, df, sequence_length=24, n_future_steps=1, train_split=0.8):
        """Prepare futures data for LSTM model with additional features."""
        if df.empty:
            logging.warning("Empty dataframe provided to prepare_futures_data")
            return None, None, None, None, None
            
        try:
            # Select features for training (compatible with existing model - 15 features)
            features = [
                'close', 'volume', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'macd', 'macd_signal', 'rsi', 'bb_width', 'price_change',
                'volatility', 'volume_ratio', 'hl_spread', 'atr'
            ]
            
            # Filter features that exist in the dataframe
            available_features = [f for f in features if f in df.columns]
            data = df[available_features].copy()
            
            # Handle missing values
            data = data.dropna()
            
            if len(data) < sequence_length + n_future_steps:
                logging.error(f"Insufficient futures data: {len(data)} rows, need at least {sequence_length + n_future_steps}")
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

            logging.info(f"Futures data preparation complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
            logging.info(f"Features used: {available_features}")
            return X_train, y_train, X_test, y_test, original_y_test
            
        except Exception as e:
            logging.error(f"Error in prepare_futures_data: {str(e)}")
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

    def get_futures_pairs(self):
        """Get all available futures trading pairs from Bybit."""
        try:
            response = self.client.get_tickers(
                category="linear"
            )
            
            if 'retCode' in response and response['retCode'] != 0:
                logging.error(f"API Error in get_futures_pairs: {response.get('retMsg', 'Unknown error')}")
                return []
            
            if not response.get('result', {}).get('list'):
                logging.error("No futures ticker data returned")
                return []
            
            # Extract all available futures symbols
            pairs = [item['symbol'] for item in response['result']['list']]
            pairs.sort()  # Sort alphabetically
            logging.info(f"Found {len(pairs)} available futures trading pairs")
            return pairs
            
        except Exception as e:
            logging.error(f"Error in get_futures_pairs: {str(e)}")
            return []

    def calculate_futures_metrics(self, df, current_price, predicted_prices):
        """Calculate futures-specific trading metrics."""
        try:
            metrics = {}
            
            # Basic price metrics
            metrics['current_price'] = current_price
            metrics['predicted_min'] = np.min(predicted_prices)
            metrics['predicted_max'] = np.max(predicted_prices)
            metrics['predicted_range'] = metrics['predicted_max'] - metrics['predicted_min']
            
            # Percentage changes
            metrics['max_upside_pct'] = (metrics['predicted_max'] - current_price) / current_price * 100
            metrics['max_downside_pct'] = (metrics['predicted_min'] - current_price) / current_price * 100
            
            # Risk metrics
            if 'atr' in df.columns and not df['atr'].empty:
                latest_atr = df['atr'].iloc[-1]
                metrics['atr'] = latest_atr
                metrics['risk_reward_ratio'] = abs(metrics['max_upside_pct']) / abs(metrics['max_downside_pct']) if metrics['max_downside_pct'] != 0 else float('inf')
            
            # Volatility
            if 'volatility' in df.columns and not df['volatility'].empty:
                metrics['volatility'] = df['volatility'].iloc[-1]
            
            # Volume analysis
            if 'volume_ratio' in df.columns and not df['volume_ratio'].empty:
                metrics['volume_ratio'] = df['volume_ratio'].iloc[-1]
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating futures metrics: {str(e)}")
            return {} 