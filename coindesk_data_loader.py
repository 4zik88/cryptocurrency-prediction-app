import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class CoindeskDataLoader:
    """
    CoinDesk Data Loader for fetching Bitcoin and crypto market data
    Uses the free CoinDesk Bitcoin Price Index API and other public endpoints
    """
    
    def __init__(self):
        self.base_url = "https://api.coindesk.com/v1/bpi"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoPredictor/1.0',
            'Accept': 'application/json'
        })
        
    def get_current_bitcoin_price(self, currency='USD') -> Optional[Dict]:
        """
        Fetch current Bitcoin price from CoinDesk API
        
        Args:
            currency: Currency code (USD, EUR, GBP)
            
        Returns:
            Dict with current price information or None if failed
        """
        try:
            url = f"{self.base_url}/currentprice/{currency}.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'bpi' in data and currency in data['bpi']:
                price_info = data['bpi'][currency]
                return {
                    'price': price_info['rate_float'],
                    'currency': currency,
                    'symbol': price_info['symbol'],
                    'description': price_info['description'],
                    'timestamp': data['time']['updated'],
                    'source': 'CoinDesk'
                }
                
        except Exception as e:
            logging.error(f"Error fetching current Bitcoin price: {str(e)}")
            
        return None
    
    def get_historical_bitcoin_data(self, start_date=None, end_date=None, currency='USD') -> Optional[pd.DataFrame]:
        """
        Fetch historical Bitcoin price data from CoinDesk
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            currency: Currency code
            
        Returns:
            DataFrame with historical price data or None if failed
        """
        try:
            url = f"{self.base_url}/historical/close.json"
            
            params = {}
            if start_date:
                params['start'] = start_date
            if end_date:
                params['end'] = end_date
            if currency != 'USD':
                params['currency'] = currency
                
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'bpi' in data:
                # Convert to DataFrame
                dates = []
                prices = []
                
                for date_str, price in data['bpi'].items():
                    dates.append(pd.to_datetime(date_str))
                    prices.append(float(price))
                
                df = pd.DataFrame({
                    'timestamp': dates,
                    'close': prices,
                    'source': 'CoinDesk'
                })
                
                df = df.set_index('timestamp').sort_index()
                
                # Add basic OHLCV columns (using close price as approximation)
                df['open'] = df['close'].shift(1).fillna(df['close'])
                df['high'] = df['close'] * 1.005  # Approximate high as 0.5% above close
                df['low'] = df['close'] * 0.995   # Approximate low as 0.5% below close
                df['volume'] = 0  # Volume not available in free API
                
                logging.info(f"Retrieved {len(df)} historical records from CoinDesk")
                return df
                
        except Exception as e:
            logging.error(f"Error fetching historical Bitcoin data: {str(e)}")
            
        return None
    
    def get_supported_currencies(self) -> List[str]:
        """
        Get list of supported currencies from CoinDesk API
        
        Returns:
            List of supported currency codes
        """
        try:
            url = f"{self.base_url}/supported-currencies.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return list(data.keys()) if isinstance(data, dict) else ['USD', 'EUR', 'GBP']
            
        except Exception as e:
            logging.warning(f"Could not fetch supported currencies: {str(e)}")
            return ['USD', 'EUR', 'GBP']  # Default fallback
    
    def get_market_sentiment_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate market sentiment indicators from price data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dict with sentiment indicators
        """
        if df.empty or len(df) < 10:
            return {}
            
        try:
            # Calculate various sentiment indicators
            recent_prices = df['close'].tail(10)
            
            # Price momentum (7-day vs 1-day change)
            if len(df) >= 7:
                week_change = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] * 100
            else:
                week_change = 0
                
            day_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100 if len(df) >= 2 else 0
            
            # Volatility (standard deviation of recent returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.tail(10).std() * 100 if len(returns) >= 10 else 0
            
            # Trend strength (correlation with time)
            if len(df) >= 10:
                time_index = np.arange(len(recent_prices))
                correlation = np.corrcoef(time_index, recent_prices)[0, 1]
                trend_strength = abs(correlation) * 100
            else:
                trend_strength = 0
            
            # Support/Resistance levels
            recent_high = recent_prices.max()
            recent_low = recent_prices.min()
            current_price = df['close'].iloc[-1]
            
            # Position within recent range
            if recent_high != recent_low:
                range_position = (current_price - recent_low) / (recent_high - recent_low) * 100
            else:
                range_position = 50
            
            return {
                'week_change_pct': round(week_change, 2),
                'day_change_pct': round(day_change, 2),
                'volatility_pct': round(volatility, 2),
                'trend_strength': round(trend_strength, 2),
                'range_position_pct': round(range_position, 2),
                'recent_high': round(recent_high, 2),
                'recent_low': round(recent_low, 2),
                'price_range': round(recent_high - recent_low, 2),
                'data_points': len(df)
            }
            
        except Exception as e:
            logging.error(f"Error calculating sentiment indicators: {str(e)}")
            return {}
    
    def enhance_prediction_with_coindesk_data(self, symbol: str, predicted_prices: np.ndarray, 
                                            current_market_data: Optional[Dict] = None) -> Dict:
        """
        Enhance predictions with CoinDesk market data and sentiment analysis
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            predicted_prices: Array of predicted prices
            current_market_data: Current market data from other sources
            
        Returns:
            Dict with enhanced prediction analysis
        """
        enhancement = {
            'coindesk_current_price': None,
            'price_deviation': None,
            'market_sentiment': {},
            'prediction_confidence': 'medium',
            'risk_factors': [],
            'source_comparison': {}
        }
        
        try:
            # Only enhance Bitcoin predictions for now (CoinDesk specializes in Bitcoin)
            if 'BTC' not in symbol.upper():
                enhancement['risk_factors'].append("CoinDesk data only available for Bitcoin")
                return enhancement
            
            # Get current Bitcoin price from CoinDesk
            coindesk_data = self.get_current_bitcoin_price()
            if coindesk_data:
                enhancement['coindesk_current_price'] = coindesk_data['price']
                
                # Compare with current market data if available
                if current_market_data and 'current_price' in current_market_data:
                    market_price = current_market_data['current_price']
                    deviation = abs(coindesk_data['price'] - market_price) / market_price * 100
                    enhancement['price_deviation'] = round(deviation, 2)
                    enhancement['source_comparison'] = {
                        'coindesk_price': coindesk_data['price'],
                        'market_price': market_price,
                        'deviation_pct': deviation
                    }
                    
                    # Add risk factor if prices deviate significantly
                    if deviation > 1.0:  # More than 1% deviation
                        enhancement['risk_factors'].append(f"Price deviation between sources: {deviation:.2f}%")
            
            # Get historical data for sentiment analysis
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            historical_df = self.get_historical_bitcoin_data(start_date, end_date)
            if historical_df is not None and not historical_df.empty:
                sentiment = self.get_market_sentiment_indicators(historical_df)
                enhancement['market_sentiment'] = sentiment
                
                # Adjust prediction confidence based on sentiment
                if sentiment:
                    volatility = sentiment.get('volatility_pct', 0)
                    trend_strength = sentiment.get('trend_strength', 0)
                    
                    if volatility > 5:  # High volatility
                        enhancement['prediction_confidence'] = 'low'
                        enhancement['risk_factors'].append(f"High market volatility: {volatility:.1f}%")
                    elif trend_strength > 70:  # Strong trend
                        enhancement['prediction_confidence'] = 'high'
                    else:
                        enhancement['prediction_confidence'] = 'medium'
            
            # Analyze prediction vs historical patterns
            if predicted_prices is not None and len(predicted_prices) > 0:
                predicted_change = (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0] * 100
                
                if coindesk_data:
                    predicted_final_price = predicted_prices[-1]
                    current_coindesk_price = coindesk_data['price']
                    coindesk_vs_prediction = abs(predicted_final_price - current_coindesk_price) / current_coindesk_price * 100
                    
                    if coindesk_vs_prediction > 5:  # More than 5% difference
                        enhancement['risk_factors'].append(f"Prediction deviates {coindesk_vs_prediction:.1f}% from CoinDesk current price")
            
        except Exception as e:
            logging.error(f"Error enhancing prediction with CoinDesk data: {str(e)}")
            enhancement['risk_factors'].append(f"CoinDesk data enhancement failed: {str(e)}")
        
        return enhancement
    
    def get_bitcoin_fear_greed_proxy(self, df: pd.DataFrame) -> Dict:
        """
        Calculate a simplified Fear & Greed index proxy using price data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dict with fear/greed indicators
        """
        if df.empty or len(df) < 14:
            return {'score': 50, 'sentiment': 'neutral', 'confidence': 'low'}
        
        try:
            # Calculate multiple factors
            factors = {}
            
            # 1. Price momentum (30%)
            if len(df) >= 7:
                week_return = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7]
                factors['momentum'] = min(max(week_return * 100 + 50, 0), 100)
            else:
                factors['momentum'] = 50
            
            # 2. Volatility (20%) - lower volatility = more greed
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 10:
                volatility = returns.tail(10).std()
                # Normalize volatility (typical BTC daily volatility is 0.02-0.08)
                vol_score = max(0, 100 - (volatility * 1000))
                factors['volatility'] = min(vol_score, 100)
            else:
                factors['volatility'] = 50
            
            # 3. Trend consistency (25%)
            if len(df) >= 10:
                recent_prices = df['close'].tail(10).values
                time_index = np.arange(len(recent_prices))
                correlation = np.corrcoef(time_index, recent_prices)[0, 1]
                # Strong positive correlation = greed, strong negative = fear
                trend_score = (correlation + 1) * 50  # Convert -1,1 to 0,100
                factors['trend'] = trend_score
            else:
                factors['trend'] = 50
            
            # 4. Support/Resistance position (25%)
            recent_prices = df['close'].tail(20)
            current_price = df['close'].iloc[-1]
            recent_high = recent_prices.max()
            recent_low = recent_prices.min()
            
            if recent_high != recent_low:
                position_score = (current_price - recent_low) / (recent_high - recent_low) * 100
                factors['position'] = position_score
            else:
                factors['position'] = 50
            
            # Calculate weighted average
            weights = {'momentum': 0.30, 'volatility': 0.20, 'trend': 0.25, 'position': 0.25}
            fear_greed_score = sum(factors[key] * weights[key] for key in factors)
            fear_greed_score = max(0, min(100, fear_greed_score))
            
            # Determine sentiment
            if fear_greed_score >= 75:
                sentiment = 'extreme_greed'
            elif fear_greed_score >= 55:
                sentiment = 'greed'
            elif fear_greed_score >= 45:
                sentiment = 'neutral'
            elif fear_greed_score >= 25:
                sentiment = 'fear'
            else:
                sentiment = 'extreme_fear'
            
            return {
                'score': round(fear_greed_score, 1),
                'sentiment': sentiment,
                'factors': {k: round(v, 1) for k, v in factors.items()},
                'confidence': 'high' if len(df) >= 20 else 'medium'
            }
            
        except Exception as e:
            logging.error(f"Error calculating fear/greed proxy: {str(e)}")
            return {'score': 50, 'sentiment': 'neutral', 'confidence': 'low'}
