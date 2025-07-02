import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import TA-Lib for candlestick pattern recognition
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Candlestick pattern recognition will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class TradingPatternAnalyzer:
    """
    Advanced trading pattern analyzer that focuses on:
    - Price jumps and movements between support/resistance levels
    - Intraday trading patterns and volume analysis
    - Market microstructure and order flow
    - Enhanced prediction features for accurate forecasting
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        logging.info("Initialized TradingPatternAnalyzer")
    
    def detect_price_jumps(self, df: pd.DataFrame, 
                          jump_threshold: float = 2.0,
                          min_volume_ratio: float = 1.5) -> Dict:
        """
        Detect significant price jumps up and down with volume confirmation.
        """
        try:
            # Calculate price changes and volume ratios
            df['price_change_pct'] = df['close'].pct_change() * 100
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma20']
            
            # Detect significant jumps
            jump_up_mask = (df['price_change_pct'] >= jump_threshold) & (df['volume_ratio'] >= min_volume_ratio)
            jump_down_mask = (df['price_change_pct'] <= -jump_threshold) & (df['volume_ratio'] >= min_volume_ratio)
            
            jumps_up = df[jump_up_mask].copy()
            jumps_down = df[jump_down_mask].copy()
            
            # Analyze jump characteristics
            jump_analysis = {
                'jumps_up': jumps_up,
                'jumps_down': jumps_down,
                'jump_up_count': len(jumps_up),
                'jump_down_count': len(jumps_down),
                'avg_jump_up_size': jumps_up['price_change_pct'].mean() if len(jumps_up) > 0 else 0,
                'avg_jump_down_size': jumps_down['price_change_pct'].mean() if len(jumps_down) > 0 else 0,
                'avg_jump_up_volume': jumps_up['volume_ratio'].mean() if len(jumps_up) > 0 else 0,
                'avg_jump_down_volume': jumps_down['volume_ratio'].mean() if len(jumps_down) > 0 else 0,
                'jump_frequency': (len(jumps_up) + len(jumps_down)) / len(df) * 100
            }
            
            # Analyze post-jump behavior
            jump_analysis['post_jump_analysis'] = self._analyze_post_jump_behavior(df, jumps_up, jumps_down)
            
            logging.info(f"Detected {len(jumps_up)} upward jumps and {len(jumps_down)} downward jumps")
            return jump_analysis
            
        except Exception as e:
            logging.error(f"Error in detect_price_jumps: {str(e)}")
            return {}
    
    def _analyze_post_jump_behavior(self, df: pd.DataFrame, 
                                   jumps_up: pd.DataFrame, 
                                   jumps_down: pd.DataFrame) -> Dict:
        """Analyze price behavior after jumps to identify patterns."""
        try:
            post_jump_periods = [1, 3, 6, 12, 24]  # Hours after jump
            analysis = {}
            
            for period in post_jump_periods:
                up_returns = []
                down_returns = []
                
                # Analyze returns after upward jumps
                for idx in jumps_up.index:
                    try:
                        idx_pos = df.index.get_loc(idx)
                        if idx_pos + period < len(df):
                            current_price = df.iloc[idx_pos]['close']
                            future_price = df.iloc[idx_pos + period]['close']
                            return_pct = (future_price - current_price) / current_price * 100
                            up_returns.append(return_pct)
                    except:
                        continue
                
                # Analyze returns after downward jumps
                for idx in jumps_down.index:
                    try:
                        idx_pos = df.index.get_loc(idx)
                        if idx_pos + period < len(df):
                            current_price = df.iloc[idx_pos]['close']
                            future_price = df.iloc[idx_pos + period]['close']
                            return_pct = (future_price - current_price) / current_price * 100
                            down_returns.append(return_pct)
                    except:
                        continue
                
                analysis[f'period_{period}h'] = {
                    'avg_return_after_jump_up': np.mean(up_returns) if up_returns else 0,
                    'avg_return_after_jump_down': np.mean(down_returns) if down_returns else 0,
                    'reversal_probability_up': len([r for r in up_returns if r < 0]) / len(up_returns) if up_returns else 0,
                    'reversal_probability_down': len([r for r in down_returns if r > 0]) / len(down_returns) if down_returns else 0,
                    'continuation_probability_up': len([r for r in up_returns if r > 0]) / len(up_returns) if up_returns else 0,
                    'continuation_probability_down': len([r for r in down_returns if r < 0]) / len(down_returns) if down_returns else 0
                }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in _analyze_post_jump_behavior: {str(e)}")
            return {}
    
    def identify_support_resistance_levels(self, df: pd.DataFrame, 
                                         window: int = 20,
                                         min_touches: int = 3) -> Dict:
        """
        Identify key support and resistance levels where price bounces occur.
        """
        try:
            # Find local highs and lows
            df['local_high'] = df['high'].rolling(window=window, center=True).max() == df['high']
            df['local_low'] = df['low'].rolling(window=window, center=True).min() == df['low']
            
            # Extract significant levels
            highs = df[df['local_high']]['high'].values
            lows = df[df['local_low']]['low'].values
            
            # Cluster levels to find key support/resistance
            all_levels = np.concatenate([highs, lows])
            if len(all_levels) < min_touches:
                return {}
            
            # Use KMeans to cluster price levels
            n_clusters = min(10, len(all_levels) // 2)
            if n_clusters < 2:
                return {}
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(all_levels.reshape(-1, 1))
            
            # Identify significant levels
            significant_levels = []
            for i in range(n_clusters):
                cluster_levels = all_levels[clusters == i]
                if len(cluster_levels) >= min_touches:
                    level_price = np.mean(cluster_levels)
                    level_strength = len(cluster_levels)
                    level_std = np.std(cluster_levels)
                    
                    # Determine if support or resistance
                    current_price = df['close'].iloc[-1]
                    level_type = 'resistance' if level_price > current_price else 'support'
                    
                    significant_levels.append({
                        'price': level_price,
                        'type': level_type,
                        'strength': level_strength,
                        'std_dev': level_std,
                        'distance_pct': abs(level_price - current_price) / current_price * 100
                    })
            
            # Sort by strength (number of touches)
            significant_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'levels': significant_levels,
                'nearest_support': self._find_nearest_level(significant_levels, df['close'].iloc[-1], 'support'),
                'nearest_resistance': self._find_nearest_level(significant_levels, df['close'].iloc[-1], 'resistance')
            }
            
        except Exception as e:
            logging.error(f"Error in identify_support_resistance_levels: {str(e)}")
            return {}
    
    def _find_nearest_level(self, levels: List[Dict], current_price: float, level_type: str) -> Optional[Dict]:
        """Find the nearest support or resistance level."""
        try:
            filtered_levels = [level for level in levels if level['type'] == level_type]
            if not filtered_levels:
                return None
            
            nearest = min(filtered_levels, key=lambda x: x['distance_pct'])
            return nearest
            
        except Exception as e:
            logging.error(f"Error in _find_nearest_level: {str(e)}")
            return None
    
    def analyze_intraday_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze intraday trading patterns including volume profiles and volatility.
        """
        try:
            # Add hour of day
            df['hour'] = df.index.hour
            
            # Calculate hourly volatility
            df['volatility'] = (df['high'] - df['low']) / df['close'] * 100
            
            # Price movement patterns by hour
            df['price_change'] = df['close'].pct_change() * 100
            
            # Market session analysis (assuming UTC time)
            sessions = {
                'asian': list(range(0, 8)),      # 00:00-08:00 UTC
                'european': list(range(8, 16)),   # 08:00-16:00 UTC
                'american': list(range(16, 24))   # 16:00-24:00 UTC
            }
            
            session_analysis = {}
            for session_name, hours in sessions.items():
                session_data = df[df['hour'].isin(hours)]
                if not session_data.empty:
                    session_analysis[session_name] = {
                        'avg_volume': session_data['volume'].mean(),
                        'avg_volatility': session_data['volatility'].mean(),
                        'avg_price_change': session_data['price_change'].mean(),
                        'directional_bias': 'bullish' if session_data['price_change'].mean() > 0 else 'bearish',
                        'activity_level': 'high' if session_data['volume'].mean() > df['volume'].median() else 'low'
                    }
            
            return {
                'session_analysis': session_analysis,
                'peak_volume_hour': df.groupby('hour')['volume'].mean().idxmax(),
                'peak_volatility_hour': df.groupby('hour')['volatility'].mean().idxmax(),
                'most_bullish_hour': df.groupby('hour')['price_change'].mean().idxmax(),
                'most_bearish_hour': df.groupby('hour')['price_change'].mean().idxmin()
            }
            
        except Exception as e:
            logging.error(f"Error in analyze_intraday_patterns: {str(e)}")
            return {}
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect candlestick patterns using TA-Lib library.
        Returns a comprehensive analysis of all detected patterns.
        """
        if not TALIB_AVAILABLE:
            logging.warning("TA-Lib not available. Candlestick pattern detection skipped.")
            return {}
        
        try:
            # Ensure we have the required OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logging.error("Missing required columns for candlestick pattern detection")
                return {}
            
            # Validate data quality
            if len(df) < 10:
                logging.warning(f"Insufficient data for pattern detection: {len(df)} candles")
                return {}
            
            # Clean and validate data
            df_clean = df.dropna(subset=required_columns)
            if len(df_clean) < len(df) * 0.8:  # If we lose more than 20% of data
                logging.warning(f"Too many NaN values in data: {len(df) - len(df_clean)} out of {len(df)}")
            
            # Use cleaned data
            df = df_clean
            
            # Extract OHLC data and ensure proper types
            open_prices = df['open'].astype(float).values
            high_prices = df['high'].astype(float).values
            low_prices = df['low'].astype(float).values
            close_prices = df['close'].astype(float).values
            
            # Validate OHLC logic
            invalid_candles = (
                (high_prices < low_prices) | 
                (high_prices < open_prices) | 
                (high_prices < close_prices) |
                (low_prices > open_prices) |
                (low_prices > close_prices)
            )
            
            if np.any(invalid_candles):
                logging.warning(f"Found {np.sum(invalid_candles)} invalid OHLC candles - this may affect pattern detection")
            
            logging.info(f"Starting pattern detection on {len(df)} candles with price range ${low_prices.min():.2f} - ${high_prices.max():.2f}")
            
            # Define all available candlestick patterns with their categories
            patterns_config = {
                # Reversal Patterns (Bullish)
                'CDL2CROWS': {'name': 'Two Crows', 'type': 'bearish_reversal'},
                'CDL3BLACKCROWS': {'name': 'Three Black Crows', 'type': 'bearish_reversal'},
                'CDL3INSIDE': {'name': 'Three Inside Up/Down', 'type': 'reversal'},
                'CDL3LINESTRIKE': {'name': 'Three-Line Strike', 'type': 'reversal'},
                'CDL3OUTSIDE': {'name': 'Three Outside Up/Down', 'type': 'reversal'},
                'CDL3STARSINSOUTH': {'name': 'Three Stars In The South', 'type': 'bullish_reversal'},
                'CDL3WHITESOLDIERS': {'name': 'Three Advancing White Soldiers', 'type': 'bullish_reversal'},
                'CDLABANDONEDBABY': {'name': 'Abandoned Baby', 'type': 'reversal'},
                'CDLADVANCEBLOCK': {'name': 'Advance Block', 'type': 'bearish_reversal'},
                'CDLBELTHOLD': {'name': 'Belt-hold', 'type': 'reversal'},
                'CDLBREAKAWAY': {'name': 'Breakaway', 'type': 'reversal'},
                'CDLCLOSINGMARUBOZU': {'name': 'Closing Marubozu', 'type': 'continuation'},
                'CDLCONCEALBABYSWALL': {'name': 'Concealing Baby Swallow', 'type': 'bullish_reversal'},
                'CDLCOUNTERATTACK': {'name': 'Counterattack', 'type': 'reversal'},
                'CDLDARKCLOUDCOVER': {'name': 'Dark Cloud Cover', 'type': 'bearish_reversal'},
                'CDLDOJI': {'name': 'Doji', 'type': 'indecision'},
                'CDLDOJISTAR': {'name': 'Doji Star', 'type': 'reversal'},
                'CDLDRAGONFLYDOJI': {'name': 'Dragonfly Doji', 'type': 'bullish_reversal'},
                'CDLENGULFING': {'name': 'Engulfing Pattern', 'type': 'reversal'},
                'CDLEVENINGDOJISTAR': {'name': 'Evening Doji Star', 'type': 'bearish_reversal'},
                'CDLEVENINGSTAR': {'name': 'Evening Star', 'type': 'bearish_reversal'},
                'CDLGAPSIDESIDEWHITE': {'name': 'Up/Down-gap side-by-side white lines', 'type': 'continuation'},
                'CDLGRAVESTONEDOJI': {'name': 'Gravestone Doji', 'type': 'bearish_reversal'},
                'CDLHAMMER': {'name': 'Hammer', 'type': 'bullish_reversal'},
                'CDLHANGINGMAN': {'name': 'Hanging Man', 'type': 'bearish_reversal'},
                'CDLHARAMI': {'name': 'Harami Pattern', 'type': 'reversal'},
                'CDLHARAMICROSS': {'name': 'Harami Cross Pattern', 'type': 'reversal'},
                'CDLHIGHWAVE': {'name': 'High-Wave Candle', 'type': 'indecision'},
                'CDLHIKKAKE': {'name': 'Hikkake Pattern', 'type': 'reversal'},
                'CDLHIKKAKEMOD': {'name': 'Modified Hikkake Pattern', 'type': 'reversal'},
                'CDLHOMINGPIGEON': {'name': 'Homing Pigeon', 'type': 'bullish_reversal'},
                'CDLIDENTICAL3CROWS': {'name': 'Identical Three Crows', 'type': 'bearish_reversal'},
                'CDLINNECK': {'name': 'In-Neck Pattern', 'type': 'bearish_continuation'},
                'CDLINVERTEDHAMMER': {'name': 'Inverted Hammer', 'type': 'bullish_reversal'},
                'CDLKICKING': {'name': 'Kicking', 'type': 'reversal'},
                'CDLKICKINGBYLENGTH': {'name': 'Kicking - bull/bear determined by the longer marubozu', 'type': 'reversal'},
                'CDLLADDERBOTTOM': {'name': 'Ladder Bottom', 'type': 'bullish_reversal'},
                'CDLLONGLEGGEDDOJI': {'name': 'Long Legged Doji', 'type': 'indecision'},
                'CDLLONGLINE': {'name': 'Long Line Candle', 'type': 'continuation'},
                'CDLMARUBOZU': {'name': 'Marubozu', 'type': 'continuation'},
                'CDLMATCHINGLOW': {'name': 'Matching Low', 'type': 'bullish_reversal'},
                'CDLMATHOLD': {'name': 'Mat Hold', 'type': 'bullish_continuation'},
                'CDLMORNINGDOJISTAR': {'name': 'Morning Doji Star', 'type': 'bullish_reversal'},
                'CDLMORNINGSTAR': {'name': 'Morning Star', 'type': 'bullish_reversal'},
                'CDLONNECK': {'name': 'On-Neck Pattern', 'type': 'bearish_continuation'},
                'CDLPIERCING': {'name': 'Piercing Pattern', 'type': 'bullish_reversal'},
                'CDLRICKSHAWMAN': {'name': 'Rickshaw Man', 'type': 'indecision'},
                'CDLRISEFALL3METHODS': {'name': 'Rising/Falling Three Methods', 'type': 'continuation'},
                'CDLSEPARATINGLINES': {'name': 'Separating Lines', 'type': 'continuation'},
                'CDLSHOOTINGSTAR': {'name': 'Shooting Star', 'type': 'bearish_reversal'},
                'CDLSHORTLINE': {'name': 'Short Line Candle', 'type': 'indecision'},
                'CDLSPINNINGTOP': {'name': 'Spinning Top', 'type': 'indecision'},
                'CDLSTALLEDPATTERN': {'name': 'Stalled Pattern', 'type': 'bearish_reversal'},
                'CDLSTICKSANDWICH': {'name': 'Stick Sandwich', 'type': 'bullish_reversal'},
                'CDLTAKURI': {'name': 'Takuri (Dragonfly Doji with very long lower shadow)', 'type': 'bullish_reversal'},
                'CDLTASUKIGAP': {'name': 'Tasuki Gap', 'type': 'continuation'},
                'CDLTHRUSTING': {'name': 'Thrusting Pattern', 'type': 'bearish_continuation'},
                'CDLTRISTAR': {'name': 'Tristar Pattern', 'type': 'reversal'},
                'CDLUNIQUE3RIVER': {'name': 'Unique 3 River', 'type': 'bullish_reversal'},
                'CDLUPSIDEGAP2CROWS': {'name': 'Upside Gap Two Crows', 'type': 'bearish_reversal'},
                'CDLXSIDEGAP3METHODS': {'name': 'Upside/Downside Gap Three Methods', 'type': 'continuation'},
            }
            
            # Detect all patterns
            detected_patterns = {}
            pattern_summary = {
                'bullish_reversal': [],
                'bearish_reversal': [],
                'continuation': [],
                'indecision': [],
                'reversal': []
            }
            
            for pattern_func, config in patterns_config.items():
                try:
                    # Get the TA-Lib function
                    talib_func = getattr(talib, pattern_func)
                    
                    # Apply the pattern detection
                    if pattern_func in ['CDLMORNINGSTAR', 'CDLEVENINGSTAR']:
                        # These functions require a penetration parameter
                        pattern_result = talib_func(open_prices, high_prices, low_prices, close_prices, penetration=0.3)
                    else:
                        pattern_result = talib_func(open_prices, high_prices, low_prices, close_prices)
                    
                    # Store non-zero results
                    pattern_signals = pattern_result[pattern_result != 0]
                    if len(pattern_signals) > 0:
                        detected_patterns[pattern_func] = {
                            'name': config['name'],
                            'type': config['type'],
                            'signals': pattern_result,
                            'count': len(pattern_signals),
                            'bullish_signals': len(pattern_signals[pattern_signals > 0]),
                            'bearish_signals': len(pattern_signals[pattern_signals < 0]),
                            'signal_indices': np.where(pattern_result != 0)[0],
                            'signal_values': pattern_signals
                        }
                        
                        # Add to summary
                        pattern_summary[config['type']].append({
                            'pattern': pattern_func,
                            'name': config['name'],
                            'count': len(pattern_signals),
                            'latest_signal': pattern_signals[-1] if len(pattern_signals) > 0 else 0,
                            'latest_index': np.where(pattern_result != 0)[0][-1] if len(pattern_signals) > 0 else -1
                        })
                
                except Exception as e:
                    logging.warning(f"Error detecting pattern {pattern_func}: {str(e)}")
                    continue
            
            # Calculate pattern statistics
            total_patterns = len(detected_patterns)
            total_signals = sum(p['count'] for p in detected_patterns.values())
            
            # Create all patterns list (not just recent)
            all_patterns = []
            for pattern_key, pattern_data in detected_patterns.items():
                for idx in pattern_data['signal_indices']:
                    all_patterns.append({
                        'pattern': pattern_key,
                        'name': pattern_data['name'],
                        'type': pattern_data['type'],
                        'index': idx,
                        'timestamp': df.index[idx],
                        'signal_strength': pattern_data['signals'][idx],
                        'price': df.iloc[idx]['close']
                    })
            
            # Sort all patterns by recency
            all_patterns.sort(key=lambda x: x['index'], reverse=True)
            
            # Find patterns in different time windows
            recent_windows = {
                'last_10': min(10, len(df)),
                'last_20': min(20, len(df)),
                'last_50': min(50, len(df)),
                'last_100': min(100, len(df))
            }
            
            recent_patterns_by_window = {}
            for window_name, window_size in recent_windows.items():
                patterns_in_window = [p for p in all_patterns if p['index'] >= (len(df) - window_size)]
                recent_patterns_by_window[window_name] = patterns_in_window
            
            # Use the most appropriate window (prefer larger windows if they have patterns)
            recent_patterns = []
            for window_name in ['last_100', 'last_50', 'last_20', 'last_10']:
                if recent_patterns_by_window[window_name]:
                    recent_patterns = recent_patterns_by_window[window_name]
                    logging.info(f"Found {len(recent_patterns)} patterns in {window_name} candles")
                    break
            
            # If no patterns found in any recent window, take the most recent patterns overall
            if not recent_patterns and all_patterns:
                recent_patterns = all_patterns[:10]  # Most recent 10 patterns from entire dataset
                logging.info(f"No recent patterns found, showing {len(recent_patterns)} most recent from entire dataset")
            
            # Add detailed debugging statistics
            window_stats = {}
            for window_name, patterns in recent_patterns_by_window.items():
                window_stats[window_name] = len(patterns)
            
            analysis_result = {
                'detected_patterns': detected_patterns,
                'pattern_summary': pattern_summary,
                'statistics': {
                    'total_patterns_detected': total_patterns,
                    'total_signals': total_signals,
                    'recent_patterns_count': len(recent_patterns),
                    'pattern_frequency': total_signals / len(df) * 100 if len(df) > 0 else 0,
                    'data_length': len(df),
                    'window_stats': window_stats,
                    'all_patterns_count': len(all_patterns)
                },
                'recent_patterns': recent_patterns[:10],  # Last 10 patterns
                'all_patterns': all_patterns[:20],  # Most recent 20 patterns from entire dataset
                'market_sentiment': self._analyze_pattern_sentiment(detected_patterns),
                'debug_info': {
                    'patterns_by_type': {
                        pattern_type: len(patterns) for pattern_type, patterns in pattern_summary.items()
                    },
                    'most_common_patterns': sorted(
                        [(name, data['count']) for name, data in detected_patterns.items()],
                        key=lambda x: x[1], reverse=True
                    )[:10]
                }
            }
            
            # Enhanced logging
            if total_patterns > 0:
                logging.info(f"✅ Detected {total_patterns} different candlestick patterns with {total_signals} total signals")
                logging.info(f"📊 Window breakdown: {window_stats}")
                most_common = analysis_result['debug_info']['most_common_patterns'][:3]
                if most_common:
                    logging.info(f"🔥 Top patterns: {', '.join([f'{name}({count})' for name, count in most_common])}")
            else:
                logging.info("❌ No candlestick patterns detected in the dataset")
                
            return analysis_result
            
        except Exception as e:
            logging.error(f"Error in detect_candlestick_patterns: {str(e)}")
            return {}
    
    def _analyze_pattern_sentiment(self, detected_patterns: Dict) -> Dict:
        """Analyze overall market sentiment based on detected patterns."""
        try:
            bullish_score = 0
            bearish_score = 0
            indecision_score = 0
            
            # Weight recent patterns more heavily
            for pattern_key, pattern_data in detected_patterns.items():
                pattern_type = pattern_data['type']
                recent_signals = pattern_data['signal_values'][-5:]  # Last 5 signals
                
                for signal in recent_signals:
                    if 'bullish' in pattern_type:
                        bullish_score += abs(signal)
                    elif 'bearish' in pattern_type:
                        bearish_score += abs(signal)
                    elif 'indecision' in pattern_type or 'doji' in pattern_data['name'].lower():
                        indecision_score += abs(signal)
                    elif pattern_type == 'reversal':
                        # Reversal patterns - add to bullish if signal is positive, bearish if negative
                        if signal > 0:
                            bullish_score += abs(signal)
                        else:
                            bearish_score += abs(signal)
            
            total_score = bullish_score + bearish_score + indecision_score
            
            if total_score == 0:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0,
                    'bullish_ratio': 0,
                    'bearish_ratio': 0,
                    'indecision_ratio': 0
                }
            
            bullish_ratio = bullish_score / total_score
            bearish_ratio = bearish_score / total_score
            indecision_ratio = indecision_score / total_score
            
            # Determine overall sentiment
            if bullish_ratio > 0.6:
                sentiment = 'bullish'
                confidence = bullish_ratio
            elif bearish_ratio > 0.6:
                sentiment = 'bearish'
                confidence = bearish_ratio
            elif indecision_ratio > 0.5:
                sentiment = 'indecision'
                confidence = indecision_ratio
            else:
                sentiment = 'mixed'
                confidence = max(bullish_ratio, bearish_ratio, indecision_ratio)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'bullish_ratio': bullish_ratio,
                'bearish_ratio': bearish_ratio,
                'indecision_ratio': indecision_ratio,
                'scores': {
                    'bullish': bullish_score,
                    'bearish': bearish_score,
                    'indecision': indecision_score
                }
            }
            
        except Exception as e:
            logging.error(f"Error in _analyze_pattern_sentiment: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0}

    def calculate_market_microstructure_features(self, df: pd.DataFrame) -> Dict:
        """
        Calculate advanced market microstructure features for better prediction accuracy.
        """
        try:
            features = {}
            
            # Calculate volatility if not exists
            if 'volatility' not in df.columns:
                df['volatility'] = (df['high'] - df['low']) / df['close'] * 100
            
            # Order flow proxies
            df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
            df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
            
            # Momentum and mean reversion indicators
            df['momentum_strength'] = df['close'].pct_change().rolling(window=5).sum()
            df['mean_reversion_indicator'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
            
            # Aggregate features
            features = {
                'avg_buying_pressure': df['buying_pressure'].mean(),
                'avg_selling_pressure': df['selling_pressure'].mean(),
                'net_pressure_trend': df['net_pressure'].rolling(window=10).mean().iloc[-1],
                'momentum_strength': df['momentum_strength'].iloc[-1],
                'mean_reversion_level': df['mean_reversion_indicator'].iloc[-1]
            }
            
            # Market regime classification
            volatility_regime = 'high' if df['volatility'].rolling(window=20).mean().iloc[-1] > df['volatility'].quantile(0.75) else 'low'
            volume_regime = 'high' if df['volume'].rolling(window=20).mean().iloc[-1] > df['volume'].quantile(0.75) else 'low'
            
            features['market_regime'] = {
                'volatility_regime': volatility_regime,
                'volume_regime': volume_regime,
                'combined_regime': f"{volatility_regime}_vol_{volume_regime}_vol"
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error in calculate_market_microstructure_features: {str(e)}")
            return {}
    
    def create_enhanced_trading_chart(self, df: pd.DataFrame, 
                                    jump_analysis: Dict,
                                    sr_analysis: Dict,
                                    symbol: str,
                                    candlestick_analysis: Dict = None) -> go.Figure:
        """
        Create an enhanced trading chart with price jumps, support/resistance levels,
        and trading patterns.
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price Action & Levels', 'Volume Profile', 'Price Jumps', 'Trading Patterns'),
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )
            
            # Main price chart with candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add support/resistance levels
            if 'levels' in sr_analysis:
                for level in sr_analysis['levels'][:5]:  # Top 5 levels
                    color = 'red' if level['type'] == 'resistance' else 'green'
                    fig.add_hline(
                        y=level['price'],
                        line_dash="dash",
                        line_color=color,
                        annotation_text=f"{level['type'].title()} (${level['price']:,.2f} - {level['strength']} touches)",
                        row=1, col=1
                    )
            
            # Mark price jumps
            if 'jumps_up' in jump_analysis and not jump_analysis['jumps_up'].empty:
                fig.add_trace(
                    go.Scatter(
                        x=jump_analysis['jumps_up'].index,
                        y=jump_analysis['jumps_up']['close'],
                        mode='markers',
                        marker=dict(color='lime', size=10, symbol='triangle-up'),
                        name='Price Jump Up',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            if 'jumps_down' in jump_analysis and not jump_analysis['jumps_down'].empty:
                fig.add_trace(
                    go.Scatter(
                        x=jump_analysis['jumps_down'].index,
                        y=jump_analysis['jumps_down']['close'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name='Price Jump Down',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Add candlestick pattern markers
            if candlestick_analysis and 'recent_patterns' in candlestick_analysis:
                # Define colors and symbols for different pattern types
                pattern_colors = {
                    'bullish_reversal': '#00FF00',      # Bright green
                    'bearish_reversal': '#FF0000',      # Bright red
                    'bullish_continuation': '#90EE90',   # Light green
                    'bearish_continuation': '#FFB6C1',   # Light pink
                    'continuation': '#FFA500',           # Orange
                    'indecision': '#FFFF00',            # Yellow
                    'reversal': '#9932CC'               # Purple
                }
                
                pattern_symbols = {
                    'bullish_reversal': 'triangle-up',
                    'bearish_reversal': 'triangle-down',
                    'bullish_continuation': 'arrow-up',
                    'bearish_continuation': 'arrow-down',
                    'continuation': 'diamond',
                    'indecision': 'circle',
                    'reversal': 'star'
                }
                
                # Group patterns by type for better visualization
                pattern_groups = {}
                for pattern in candlestick_analysis['recent_patterns']:
                    pattern_type = pattern['type']
                    if pattern_type not in pattern_groups:
                        pattern_groups[pattern_type] = []
                    pattern_groups[pattern_type].append(pattern)
                
                # Add markers for each pattern type
                for pattern_type, patterns in pattern_groups.items():
                    if patterns:
                        x_values = [df.index[p['index']] for p in patterns]
                        y_values = [p['price'] for p in patterns]
                        pattern_names = [p['name'] for p in patterns]
                        
                        # Create hover text with pattern details
                        hover_text = [
                            f"{p['name']}<br>Type: {p['type']}<br>Signal: {p['signal_strength']}<br>Price: ${p['price']:,.2f}"
                            for p in patterns
                        ]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='markers',
                                marker=dict(
                                    color=pattern_colors.get(pattern_type, '#FFFFFF'),
                                    size=12,
                                    symbol=pattern_symbols.get(pattern_type, 'circle'),
                                    line=dict(width=1, color='black')
                                ),
                                name=f'{pattern_type.replace("_", " ").title()} Patterns',
                                text=pattern_names,
                                hovertext=hover_text,
                                hoverinfo='text',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
            
            # Volume profile
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='rgba(158,202,225,0.8)',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Price change visualization
            colors = ['green' if x > 0 else 'red' for x in df['close'].pct_change() * 100]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['close'].pct_change() * 100,
                    name='Price Change %',
                    marker_color=colors,
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Trading patterns (volume ratio)
            if 'volume_ratio' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['volume_ratio'],
                        mode='lines',
                        name='Volume Ratio',
                        line=dict(color='orange'),
                        showlegend=False
                    ),
                    row=4, col=1
                )
                
                # Add volume spike threshold
                fig.add_hline(
                    y=1.5,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="Volume Spike Threshold",
                    row=4, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Enhanced Trading Pattern Analysis',
                template='plotly_dark',
                height=1000,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update x-axis
            fig.update_xaxes(title_text="Time", row=4, col=1)
            
            # Update y-axes
            fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="Price Change %", row=3, col=1)
            fig.update_yaxes(title_text="Volume Ratio", row=4, col=1)
            
            return fig
            
        except Exception as e:
            logging.error(f"Error in create_enhanced_trading_chart: {str(e)}")
            return go.Figure()
    
    def enhance_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced features for more accurate price prediction.
        """
        try:
            # Price jump features
            df['price_jump_up'] = ((df['close'].pct_change() > 0.02) & 
                                  (df['volume'] > df['volume'].rolling(20).mean() * 1.5)).astype(int)
            df['price_jump_down'] = ((df['close'].pct_change() < -0.02) & 
                                    (df['volume'] > df['volume'].rolling(20).mean() * 1.5)).astype(int)
            
            # Intraday pattern features
            df['hour_of_day'] = df.index.hour
            df['is_high_volatility_hour'] = (df['hour_of_day'].isin([8, 9, 16, 17, 22, 23])).astype(int)
            
            # Market microstructure features
            df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
            df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
            
            # Volume-based features
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['volume_momentum'] = df['volume'].pct_change(5)
            
            # Volatility features
            df['volatility'] = (df['high'] - df['low']) / df['close']
            df['volatility_ma_ratio'] = df['volatility'] / df['volatility'].rolling(20).mean()
            
            # Momentum features
            df['momentum_1h'] = df['close'].pct_change(1)
            df['momentum_4h'] = df['close'].pct_change(4)
            df['momentum_24h'] = df['close'].pct_change(24)
            
            # Mean reversion features
            df['price_deviation'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            df['volume_deviation'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            
            # Regime features
            df['high_vol_regime'] = (df['volatility'] > df['volatility'].quantile(0.75)).astype(int)
            df['high_volume_regime'] = (df['volume'] > df['volume'].quantile(0.75)).astype(int)
            
            # Fill any NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            logging.info("Enhanced prediction features added successfully")
            return df
            
        except Exception as e:
            logging.error(f"Error in enhance_prediction_features: {str(e)}")
            return df
    
    def generate_trading_insights(self, df: pd.DataFrame, 
                                jump_analysis: Dict,
                                sr_analysis: Dict,
                                intraday_analysis: Dict,
                                microstructure_features: Dict) -> Dict:
        """
        Generate comprehensive trading insights based on all analyses.
        """
        try:
            insights = {
                'market_summary': {},
                'key_levels': {},
                'trading_opportunities': [],
                'risk_factors': [],
                'pattern_signals': []
            }
            
            current_price = df['close'].iloc[-1]
            
            # Market summary
            insights['market_summary'] = {
                'current_price': current_price,
                'trend': 'bullish' if df['close'].pct_change(10).iloc[-1] > 0 else 'bearish',
                'volatility_level': microstructure_features.get('market_regime', {}).get('volatility_regime', 'unknown'),
                'volume_level': microstructure_features.get('market_regime', {}).get('volume_regime', 'unknown'),
                'jump_frequency': jump_analysis.get('jump_frequency', 0),
                'market_efficiency': microstructure_features.get('avg_buying_pressure', 0.5)
            }
            
            # Key levels analysis
            if 'nearest_support' in sr_analysis and sr_analysis['nearest_support']:
                insights['key_levels']['support'] = {
                    'price': sr_analysis['nearest_support']['price'],
                    'distance_pct': sr_analysis['nearest_support']['distance_pct'],
                    'strength': sr_analysis['nearest_support']['strength']
                }
            
            if 'nearest_resistance' in sr_analysis and sr_analysis['nearest_resistance']:
                insights['key_levels']['resistance'] = {
                    'price': sr_analysis['nearest_resistance']['price'],
                    'distance_pct': sr_analysis['nearest_resistance']['distance_pct'],
                    'strength': sr_analysis['nearest_resistance']['strength']
                }
            
            # Trading opportunities
            net_pressure = microstructure_features.get('net_pressure_trend', 0)
            if net_pressure > 0.1:
                insights['trading_opportunities'].append({
                    'type': 'bullish_momentum',
                    'signal': 'Strong buying pressure detected',
                    'confidence': 'high' if net_pressure > 0.2 else 'medium'
                })
            elif net_pressure < -0.1:
                insights['trading_opportunities'].append({
                    'type': 'bearish_momentum',
                    'signal': 'Strong selling pressure detected',  
                    'confidence': 'high' if net_pressure < -0.2 else 'medium'
                })
            
            # Jump-based opportunities
            if 'post_jump_analysis' in jump_analysis:
                for period, data in jump_analysis['post_jump_analysis'].items():
                    if data['reversal_probability_up'] > 0.6:
                        insights['trading_opportunities'].append({
                            'type': 'mean_reversion',
                            'signal': f'High probability of reversal after upward jumps ({period})',
                            'confidence': 'medium'
                        })
            
            # Risk factors
            if jump_analysis.get('jump_frequency', 0) > 5:
                insights['risk_factors'].append('High price jump frequency indicates elevated volatility')
            
            # Pattern signals
            if intraday_analysis.get('peak_volume_hour'):
                insights['pattern_signals'].append({
                    'type': 'volume_pattern',
                    'signal': f'Peak volume typically occurs at hour {intraday_analysis["peak_volume_hour"]}',
                    'actionable': True
                })
            
            if intraday_analysis.get('session_analysis'):
                for session, data in intraday_analysis['session_analysis'].items():
                    if data['directional_bias'] != 'neutral':
                        insights['pattern_signals'].append({
                            'type': 'session_bias',
                            'signal': f'{session.title()} session shows {data["directional_bias"]} bias',
                            'actionable': True
                        })
            
            return insights
            
        except Exception as e:
            logging.error(f"Error in generate_trading_insights: {str(e)}")
            return {}
