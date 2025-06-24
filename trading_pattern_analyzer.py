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
                                    symbol: str) -> go.Figure:
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
                        annotation_text=f"{level['type'].title()} ({level['strength']} touches)",
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
