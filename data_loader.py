import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator, ChaikinMoneyFlowIndicator
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class DataLoader:
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
            logging.info("Successfully initialized Bybit client")
        except Exception as e:
            logging.error(f"Failed to initialize Bybit client: {str(e)}")
            raise
            
        self.scaler = RobustScaler()
        self.min_max_scaler = MinMaxScaler()
        
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
                    
                    # Fetch kline data
                    response = self.client.get_kline(
                        category="spot",
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
        """Add comprehensive technical indicators to the dataframe."""
        if df.empty:
            logging.warning("Empty dataframe provided to add_technical_indicators")
            return df
            
        try:
            # Moving Averages
            df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
            
            # Momentum Indicators
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Bollinger Bands
            bb = BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Average True Range (Volatility)
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Volume Indicators
            df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['cmf'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
            df['volume_price_trend'] = VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
            
            # Price Action Features
            df['price_change'] = df['close'].pct_change()
            df['price_momentum_3'] = df['close'].pct_change(3)
            df['price_momentum_12'] = df['close'].pct_change(12)
            df['price_momentum_24'] = df['close'].pct_change(24)
            
            # Volatility Features
            df['volatility_3'] = df['close'].rolling(3).std()
            df['volatility_12'] = df['close'].rolling(12).std()
            df['volatility_24'] = df['close'].rolling(24).std()
            
            # Volume Features
            df['volume_change'] = df['volume'].pct_change()
            df['volume_price_avg'] = (df['volume'] * df['close']).rolling(24).mean()
            
            # Market Microstructure
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['open']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
            
            # Support and Resistance
            df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
            df['resistance_1'] = 2 * df['pivot_point'] - df['low']
            df['support_1'] = 2 * df['pivot_point'] - df['high']
            
            # Ichimoku Cloud
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
            senkou_span_a = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
            senkou_span_b = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
            
            # Price vs. Cloud
            df['price_vs_cloud'] = np.where(df['close'] > senkou_span_a, 1, np.where(df['close'] < senkou_span_b, -1, 0))
            
            # Remove rows with NaN values after adding indicators
            df.dropna(inplace=True)
            logging.info(f"Technical indicators added, dataframe shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error adding technical indicators: {str(e)}")
            return df

    def prepare_data(self, df, sequence_length=24, n_future_steps=1, train_split=0.8):
        """Prepare data for LSTM model."""
        if df.empty:
            logging.warning("Empty dataframe provided to prepare_data")
            return None, None, None, None, None
            
        try:
            # Select features for the model. More features can capture more complex patterns.
            features = [
                'close', 'volume', 'sma_20', 'sma_50', 'rsi', 'macd', 
                'bb_width', 'stoch_k', 'williams_r', 'atr', 'obv', 'cmf',
                'tenkan_sen', 'kijun_sen', 'price_vs_cloud'
            ]
            
            # Ensure all features exist in the dataframe
            available_features = [f for f in features if f in df.columns]
            if len(available_features) < len(features):
                missing_features = set(features) - set(available_features)
                logging.warning(f"Missing features: {missing_features}")
            
            data = df[available_features].copy()
            
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
        if scaled_prices.ndim == 1:
            scaled_prices = scaled_prices.reshape(-1, 1)
        
        num_features = self.scaler.n_features_in_
        dummy_features = np.zeros((scaled_prices.shape[0], num_features))
        dummy_features[:, 0] = scaled_prices.flatten()
        
        # Inverse transform
        return self.scaler.inverse_transform(dummy_features)[:, 0]

    def get_latest_price(self, symbol):
        """Get the latest price for a given symbol."""
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
    
    def fetch_multi_timeframe_data(self, symbol, timeframes=['60', '240', 'D'], lookback_days=180):
        """Fetch data from multiple timeframes for context."""
        multi_data = {}
        try:
            for tf in timeframes:
                logging.info(f"Fetching {tf} timeframe data for {symbol}")
                df = self.fetch_historical_data(symbol, interval=tf, lookback_days=lookback_days)
                if not df.empty:
                    df = self.add_technical_indicators(df)
                    # Add timeframe-specific suffix to column names (except OHLCV)
                    base_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                    for col in df.columns:
                        if col not in base_cols:
                            df[f'{col}_{tf}'] = df[col]
                            df = df.drop(columns=[col])
                    multi_data[tf] = df
                else:
                    logging.warning(f"No data received for {symbol} at {tf} timeframe")
            
            logging.info(f"Successfully fetched multi-timeframe data for {len(multi_data)} timeframes")
            return multi_data
        except Exception as e:
            logging.error(f"Error in fetch_multi_timeframe_data: {str(e)}")
            return {}
    
    def add_market_regime_detection(self, df):
        """Add market regime detection features."""
        try:
            if df.empty:
                return df
                
            # Volatility regime
            df['vol_20'] = df['close'].rolling(20).std()
            df['vol_percentile'] = df['vol_20'].rolling(100).rank(pct=True)
            df['volatility_regime'] = pd.cut(df['vol_percentile'], 
                                           bins=[0, 0.33, 0.66, 1.0], 
                                           labels=['Low', 'Medium', 'High'])
            
            # Trend regime
            df['trend_20'] = df['close'].rolling(20).mean()
            df['trend_50'] = df['close'].rolling(50).mean()
            df['trend_strength'] = (df['trend_20'] - df['trend_50']) / df['trend_50']
            df['trend_regime'] = pd.cut(df['trend_strength'], 
                                      bins=[-np.inf, -0.02, 0.02, np.inf], 
                                      labels=['Bear', 'Sideways', 'Bull'])
            
            # Volume regime
            df['volume_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_20']
            df['volume_regime'] = pd.cut(df['volume_ratio'], 
                                       bins=[0, 0.8, 1.2, np.inf], 
                                       labels=['Low', 'Normal', 'High'])
            
            # Convert categorical to numerical
            df['volatility_regime_num'] = pd.Categorical(df['volatility_regime']).codes
            df['trend_regime_num'] = pd.Categorical(df['trend_regime']).codes
            df['volume_regime_num'] = pd.Categorical(df['volume_regime']).codes
            
            logging.info("Successfully added market regime detection features")
            return df
        except Exception as e:
            logging.error(f"Error in add_market_regime_detection: {str(e)}")
            return df
    
    def calculate_risk_metrics(self, predictions, current_price):
        """Calculate comprehensive risk metrics."""
        try:
            if len(predictions) == 0:
                return {}
                
            predictions = np.array(predictions)
            returns = (predictions - current_price) / current_price * 100
            
            metrics = {
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1),
                'expected_shortfall_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
                'expected_shortfall_99': np.mean(returns[returns <= np.percentile(returns, 1)]),
                'max_potential_loss': np.min(returns),
                'max_potential_gain': np.max(returns),
                'mean_return': np.mean(returns),
                'volatility': np.std(returns),
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns),
                'downside_deviation': np.std(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0,
                'upside_deviation': np.std(returns[returns > 0]) if len(returns[returns > 0]) > 0 else 0,
                'probability_of_loss': len(returns[returns < 0]) / len(returns) * 100,
                'probability_of_gain': len(returns[returns > 0]) / len(returns) * 100
            }
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0%)
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0
            
            # Calculate Sortino ratio
            if metrics['downside_deviation'] > 0:
                metrics['sortino_ratio'] = metrics['mean_return'] / metrics['downside_deviation']
            else:
                metrics['sortino_ratio'] = 0
            
            logging.info("Successfully calculated risk metrics")
            return metrics
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        try:
            n = len(data)
            if n < 3:
                return 0
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return n / ((n-1) * (n-2)) * np.sum(((data - mean) / std) ** 3)
        except:
            return 0
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data."""
        try:
            n = len(data)
            if n < 4:
                return 0
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            kurt = n * (n+1) / ((n-1) * (n-2) * (n-3)) * np.sum(((data - mean) / std) ** 4)
            kurt -= 3 * (n-1)**2 / ((n-2) * (n-3))
            return kurt
        except:
            return 0
    
    def get_correlation_data(self, symbol, major_coins=['BTCUSDT', 'ETHUSDT'], lookback_days=30):
        """Get correlation data with major cryptocurrencies."""
        try:
            correlations = {}
            main_data = self.fetch_historical_data(symbol, lookback_days=lookback_days)
            
            if main_data.empty:
                return correlations
                
            for coin in major_coins:
                if coin != symbol:
                    try:
                        coin_data = self.fetch_historical_data(coin, lookback_days=lookback_days)
                        if not coin_data.empty:
                            # Align timestamps
                            common_index = main_data.index.intersection(coin_data.index)
                            if len(common_index) > 10:  # Need sufficient data points
                                main_prices = main_data.loc[common_index, 'close']
                                coin_prices = coin_data.loc[common_index, 'close']
                                correlation = np.corrcoef(main_prices, coin_prices)[0, 1]
                                correlations[f'{coin}_correlation'] = correlation
                                
                                # Also calculate rolling correlation
                                main_returns = main_prices.pct_change().dropna()
                                coin_returns = coin_prices.pct_change().dropna()
                                if len(main_returns) > 5:
                                    rolling_corr = main_returns.rolling(7).corr(coin_returns).iloc[-1]
                                    correlations[f'{coin}_rolling_correlation'] = rolling_corr
                    except Exception as e:
                        logging.warning(f"Could not calculate correlation with {coin}: {str(e)}")
                        
            logging.info(f"Successfully calculated correlations with {len(correlations)} pairs")
            return correlations
        except Exception as e:
            logging.error(f"Error in get_correlation_data: {str(e)}")
            return {}