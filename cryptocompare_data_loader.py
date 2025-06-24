import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, List, Union
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class CryptoCompareDataLoader:
    """
    CryptoCompare Data Loader for comprehensive cryptocurrency market data
    Supports all cryptocurrencies, both spot and futures markets
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Try to get API key from various sources
        self.api_key = api_key or self._get_api_key()
        
        if not self.api_key:
            logging.warning("CryptoCompare API key not found. Some features may be limited.")
        
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoPredictor/1.0',
            'Accept': 'application/json'
        })
        
        if self.api_key:
            self.session.headers.update({'Authorization': f'Apikey {self.api_key}'})
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from various sources"""
        try:
            # Try Streamlit secrets first
            import streamlit as st
            api_key = st.secrets.get('CRYPTOCOMPARE_API_KEY', '')
            if api_key:
                return api_key
        except (ImportError, Exception):
            pass
        
        # Try environment variable
        api_key = os.getenv('CRYPTOCOMPARE_API_KEY', '')
        if api_key:
            return api_key
        
        return None
    
    def get_current_price(self, symbol: str, currency: str = 'USD') -> Optional[Dict]:
        """
        Get current price for any cryptocurrency
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'ADA')
            currency: Target currency (USD, EUR, etc.)
            
        Returns:
            Dict with current price information
        """
        try:
            # Extract base symbol from trading pair
            base_symbol = self._extract_base_symbol(symbol)
            
            url = f"{self.base_url}/price"
            params = {
                'fsym': base_symbol,
                'tsyms': currency
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if currency in data:
                return {
                    'symbol': base_symbol,
                    'price': float(data[currency]),
                    'currency': currency,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'CryptoCompare'
                }
                
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {str(e)}")
            
        return None
    
    def get_historical_data(self, symbol: str, currency: str = 'USD', 
                          days: int = 30, interval: str = 'day') -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for any cryptocurrency
        
        Args:
            symbol: Crypto symbol
            currency: Target currency
            days: Number of days of history
            interval: 'minute', 'hour', 'day'
            
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            base_symbol = self._extract_base_symbol(symbol)
            
            # Map interval to API endpoint
            endpoint_map = {
                'minute': 'histominute',
                'hour': 'histohour', 
                'day': 'histoday'
            }
            
            if interval not in endpoint_map:
                interval = 'hour'
            
            url = f"{self.base_url}/{endpoint_map[interval]}"
            
            # Calculate limit based on interval and days
            if interval == 'minute':
                limit = min(days * 24 * 60, 2000)  # Max 2000 points
            elif interval == 'hour':
                limit = min(days * 24, 2000)
            else:
                limit = min(days, 365)  # Max 365 days
            
            params = {
                'fsym': base_symbol,
                'tsym': currency,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Response') == 'Success' and 'Data' in data:
                df_data = []
                # Handle both old and new API response formats
                data_items = data['Data'].get('Data', data['Data']) if isinstance(data['Data'], dict) else data['Data']
                
                for item in data_items:
                    df_data.append({
                        'timestamp': pd.to_datetime(item['time'], unit='s'),
                        'open': float(item['open']),
                        'high': float(item['high']),
                        'low': float(item['low']),
                        'close': float(item['close']),
                        'volume': float(item['volumefrom']),
                        'source': 'CryptoCompare'
                    })
                
                df = pd.DataFrame(df_data)
                df = df.set_index('timestamp').sort_index()
                
                # Remove zero entries (weekends/no trading)
                df = df[(df['open'] > 0) & (df['close'] > 0)]
                
                logging.info(f"Retrieved {len(df)} historical records for {base_symbol}")
                return df
                
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
            
        return None
    
    def get_market_metrics(self, symbol: str, currency: str = 'USD') -> Optional[Dict]:
        """
        Get comprehensive market metrics for any cryptocurrency
        
        Args:
            symbol: Crypto symbol
            currency: Target currency
            
        Returns:
            Dict with market metrics
        """
        try:
            base_symbol = self._extract_base_symbol(symbol)
            
            # Get current price data with additional metrics
            url = f"{self.base_url}/pricemultifull"
            params = {
                'fsyms': base_symbol,
                'tsyms': currency
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'RAW' in data and base_symbol in data['RAW'] and currency in data['RAW'][base_symbol]:
                raw_data = data['RAW'][base_symbol][currency]
                
                metrics = {
                    'symbol': base_symbol,
                    'price': float(raw_data.get('PRICE', 0)),
                    'market_cap': float(raw_data.get('MKTCAP', 0)),
                    'volume_24h': float(raw_data.get('VOLUME24HOUR', 0)),
                    'change_24h': float(raw_data.get('CHANGE24HOUR', 0)),
                    'change_pct_24h': float(raw_data.get('CHANGEPCT24HOUR', 0)),
                    'high_24h': float(raw_data.get('HIGH24HOUR', 0)),
                    'low_24h': float(raw_data.get('LOW24HOUR', 0)),
                    'open_24h': float(raw_data.get('OPEN24HOUR', 0)),
                    'supply': float(raw_data.get('SUPPLY', 0)),
                    'last_update': raw_data.get('LASTUPDATE', 0),
                    'source': 'CryptoCompare'
                }
                
                return metrics
                
        except Exception as e:
            logging.error(f"Error fetching market metrics for {symbol}: {str(e)}")
            
        return None
    
    def get_social_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Get social sentiment data for cryptocurrency
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Dict with social sentiment metrics
        """
        try:
            base_symbol = self._extract_base_symbol(symbol)
            
            url = f"{self.base_url}/social/coin/latest"
            params = {'coinId': base_symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Response') == 'Success' and 'Data' in data:
                social_data = data['Data']
                
                sentiment = {
                    'symbol': base_symbol,
                    'reddit_posts_per_hour': social_data.get('Reddit', {}).get('posts_per_hour', 0),
                    'reddit_comments_per_hour': social_data.get('Reddit', {}).get('comments_per_hour', 0),
                    'twitter_followers': social_data.get('Twitter', {}).get('followers', 0),
                    'twitter_following': social_data.get('Twitter', {}).get('following', 0),
                    'facebook_likes': social_data.get('Facebook', {}).get('likes', 0),
                    'general_points': social_data.get('General', {}).get('Points', 0),
                    'source': 'CryptoCompare'
                }
                
                return sentiment
                
        except Exception as e:
            logging.warning(f"Social sentiment not available for {symbol}: {str(e)}")
            
        return None
    
    def get_news_sentiment(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        """
        Get recent news with sentiment for cryptocurrency
        
        Args:
            symbol: Crypto symbol
            limit: Number of news articles
            
        Returns:
            List of news articles with sentiment
        """
        try:
            base_symbol = self._extract_base_symbol(symbol)
            
            url = f"{self.base_url}/v2/news/"
            params = {
                'categories': base_symbol,
                'limit': limit,
                'sortOrder': 'latest'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Response') == 'Success' and 'Data' in data:
                news_list = []
                for article in data['Data']:
                    news_item = {
                        'title': article.get('title', ''),
                        'published_on': article.get('published_on', 0),
                        'sentiment': article.get('sentiment', 0),  # -1 to 1 scale
                        'categories': article.get('categories', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source_info', {}).get('name', ''),
                        'lang': article.get('lang', 'EN')
                    }
                    news_list.append(news_item)
                
                return news_list
                
        except Exception as e:
            logging.warning(f"News sentiment not available for {symbol}: {str(e)}")
            
        return None
    
    def calculate_advanced_sentiment(self, symbol: str, historical_df: pd.DataFrame = None) -> Dict:
        """
        Calculate advanced sentiment indicators combining multiple data sources
        
        Args:
            symbol: Crypto symbol
            historical_df: Historical price data
            
        Returns:
            Dict with comprehensive sentiment analysis
        """
        sentiment_data = {
            'symbol': symbol,
            'overall_score': 50,  # Neutral baseline
            'confidence': 'low',
            'factors': {},
            'signals': []
        }
        
        try:
            base_symbol = self._extract_base_symbol(symbol)
            
            # 1. Market Metrics Analysis (40% weight)
            market_metrics = self.get_market_metrics(symbol)
            if market_metrics:
                # Price momentum
                change_24h = market_metrics.get('change_pct_24h', 0)
                momentum_score = min(max(change_24h * 2 + 50, 0), 100)
                sentiment_data['factors']['price_momentum'] = momentum_score
                
                # Volume analysis
                volume_24h = market_metrics.get('volume_24h', 0)
                if volume_24h > 0:
                    # Normalize volume (this is simplified)
                    volume_score = min(max((volume_24h / 1000000) * 10 + 40, 0), 100)
                    sentiment_data['factors']['volume_strength'] = volume_score
            
            # 2. Social Sentiment Analysis (30% weight)
            social_data = self.get_social_sentiment(symbol)
            if social_data:
                # Reddit activity
                reddit_activity = (social_data.get('reddit_posts_per_hour', 0) + 
                                 social_data.get('reddit_comments_per_hour', 0))
                if reddit_activity > 0:
                    social_score = min(max(reddit_activity * 5 + 30, 0), 100)
                    sentiment_data['factors']['social_activity'] = social_score
            
            # 3. News Sentiment Analysis (20% weight)
            news_data = self.get_news_sentiment(symbol, limit=20)
            if news_data:
                sentiments = [article.get('sentiment', 0) for article in news_data if article.get('sentiment') is not None]
                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    news_score = (avg_sentiment + 1) * 50  # Convert -1,1 to 0,100
                    sentiment_data['factors']['news_sentiment'] = news_score
            
            # 4. Technical Analysis (10% weight)
            if historical_df is not None and not historical_df.empty:
                tech_indicators = self._calculate_technical_sentiment(historical_df)
                sentiment_data['factors'].update(tech_indicators)
            
            # Calculate overall score
            factors = sentiment_data['factors']
            if factors:
                weights = {
                    'price_momentum': 0.4,
                    'volume_strength': 0.2,
                    'social_activity': 0.15,
                    'news_sentiment': 0.15,
                    'technical_score': 0.1
                }
                
                weighted_sum = 0
                total_weight = 0
                
                for factor, score in factors.items():
                    weight = weights.get(factor, 0.1)
                    weighted_sum += score * weight
                    total_weight += weight
                
                if total_weight > 0:
                    sentiment_data['overall_score'] = round(weighted_sum / total_weight, 1)
                    sentiment_data['confidence'] = 'high' if len(factors) >= 3 else 'medium'
            
            # Generate signals
            overall_score = sentiment_data['overall_score']
            if overall_score >= 80:
                sentiment_data['signals'].append("🚀 Very Bullish - Strong positive sentiment across multiple indicators")
            elif overall_score >= 65:
                sentiment_data['signals'].append("📈 Bullish - Positive sentiment signals detected")
            elif overall_score >= 35:
                sentiment_data['signals'].append("📊 Neutral - Mixed sentiment signals")
            elif overall_score >= 20:
                sentiment_data['signals'].append("📉 Bearish - Negative sentiment indicators")
            else:
                sentiment_data['signals'].append("⛔ Very Bearish - Strong negative sentiment")
                
        except Exception as e:
            logging.error(f"Error calculating advanced sentiment for {symbol}: {str(e)}")
            sentiment_data['signals'].append(f"⚠️ Sentiment analysis failed: {str(e)}")
        
        return sentiment_data
    
    def _calculate_technical_sentiment(self, df: pd.DataFrame) -> Dict:
        """Calculate technical sentiment from price data"""
        if df.empty or len(df) < 10:
            return {}
        
        try:
            # RSI-like momentum
            recent_changes = df['close'].pct_change().tail(14)
            gains = recent_changes[recent_changes > 0].sum()
            losses = abs(recent_changes[recent_changes < 0].sum())
            
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
                technical_score = rsi
            else:
                technical_score = 80  # Strong uptrend
            
            return {'technical_score': round(technical_score, 1)}
            
        except Exception as e:
            logging.warning(f"Technical sentiment calculation failed: {e}")
            return {}
    
    def _extract_base_symbol(self, symbol: str) -> str:
        """Extract base cryptocurrency symbol from trading pair"""
        # Handle common trading pair formats
        if 'USDT' in symbol.upper():
            return symbol.upper().replace('USDT', '')
        elif 'USD' in symbol.upper():
            return symbol.upper().replace('USD', '')
        elif 'BTC' in symbol.upper() and symbol.upper() != 'BTC':
            return symbol.upper().replace('BTC', '')
        elif 'ETH' in symbol.upper() and symbol.upper() != 'ETH':
            return symbol.upper().replace('ETH', '')
        else:
            return symbol.upper()
    
    def enhance_prediction_with_cryptocompare_data(self, symbol: str, predicted_prices: np.ndarray, 
                                                 current_market_data: Optional[Dict] = None) -> Dict:
        """
        Enhance predictions with comprehensive CryptoCompare market intelligence
        
        Args:
            symbol: Trading symbol
            predicted_prices: Array of predicted prices
            current_market_data: Current market data from exchanges
            
        Returns:
            Dict with enhanced prediction analysis
        """
        enhancement = {
            'cryptocompare_data': {},
            'market_metrics': {},
            'sentiment_analysis': {},
            'prediction_confidence': 'medium',
            'risk_factors': [],
            'opportunities': [],
            'source': 'CryptoCompare'
        }
        
        try:
            base_symbol = self._extract_base_symbol(symbol)
            
            # Get current price and market metrics
            current_price_data = self.get_current_price(symbol)
            market_metrics = self.get_market_metrics(symbol)
            
            if current_price_data:
                enhancement['cryptocompare_data'] = current_price_data
                
                # Compare with exchange data
                if current_market_data and 'current_price' in current_market_data:
                    exchange_price = current_market_data['current_price']
                    cc_price = current_price_data['price']
                    deviation = abs(cc_price - exchange_price) / exchange_price * 100
                    
                    enhancement['price_comparison'] = {
                        'exchange_price': exchange_price,
                        'cryptocompare_price': cc_price,
                        'deviation_pct': round(deviation, 2)
                    }
                    
                    if deviation > 2.0:
                        enhancement['risk_factors'].append(f"Significant price deviation: {deviation:.2f}%")
            
            if market_metrics:
                enhancement['market_metrics'] = market_metrics
                
                # Analyze market health
                volume_24h = market_metrics.get('volume_24h', 0)
                change_24h = market_metrics.get('change_pct_24h', 0)
                
                if volume_24h > 1000000:  # High volume
                    enhancement['opportunities'].append("High trading volume indicates strong market interest")
                elif volume_24h < 100000:  # Low volume
                    enhancement['risk_factors'].append("Low trading volume may indicate reduced liquidity")
                
                if abs(change_24h) > 10:  # High volatility
                    enhancement['risk_factors'].append(f"High 24h volatility: {change_24h:+.2f}%")
                    enhancement['prediction_confidence'] = 'low'
                elif abs(change_24h) < 2:  # Low volatility
                    enhancement['prediction_confidence'] = 'high'
            
            # Get historical data for sentiment analysis
            historical_df = self.get_historical_data(symbol, days=30, interval='day')
            sentiment_analysis = self.calculate_advanced_sentiment(symbol, historical_df)
            enhancement['sentiment_analysis'] = sentiment_analysis
            
            # Adjust confidence based on sentiment
            sentiment_score = sentiment_analysis.get('overall_score', 50)
            if sentiment_score > 70 or sentiment_score < 30:
                # Strong sentiment in either direction increases confidence
                if enhancement['prediction_confidence'] == 'medium':
                    enhancement['prediction_confidence'] = 'high'
            
            # Analyze prediction vs market trends
            if predicted_prices is not None and len(predicted_prices) > 0 and market_metrics:
                predicted_change = (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0] * 100
                market_trend = market_metrics.get('change_pct_24h', 0)
                
                if abs(predicted_change - market_trend) > 15:
                    enhancement['risk_factors'].append(
                        f"Prediction ({predicted_change:+.1f}%) diverges from recent trend ({market_trend:+.1f}%)"
                    )
                elif abs(predicted_change - market_trend) < 5:
                    enhancement['opportunities'].append("Prediction aligns with current market trend")
            
        except Exception as e:
            logging.error(f"Error enhancing prediction with CryptoCompare data: {str(e)}")
            enhancement['risk_factors'].append(f"CryptoCompare enhancement failed: {str(e)}")
        
        return enhancement
