#!/usr/bin/env python3
"""
Candlestick Pattern Recognition Test Script

This script demonstrates the new candlestick pattern recognition functionality
without running the full Streamlit application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def create_sample_data(days=30):
    """Create sample OHLCV data for testing patterns"""
    np.random.seed(42)  # For reproducible results
    
    dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
    
    # Create realistic price movements
    base_price = 45000  # Base price like BTC
    prices = []
    current_price = base_price
    
    for i in range(days):
        # Random walk with some volatility
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        current_price *= (1 + change)
        
        # Create OHLC for the day
        volatility = abs(np.random.normal(0, 0.01))  # Intraday volatility
        high = current_price * (1 + volatility)
        low = current_price * (1 - volatility)
        
        # Randomly decide if it's a bullish or bearish day
        if np.random.random() > 0.5:
            open_price = low + (high - low) * np.random.random() * 0.3
            close_price = high - (high - low) * np.random.random() * 0.3
        else:
            open_price = high - (high - low) * np.random.random() * 0.3
            close_price = low + (high - low) * np.random.random() * 0.3
        
        volume = np.random.normal(1000000, 200000)  # Random volume
        volume = max(volume, 100000)  # Minimum volume
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(prices, index=dates)
    return df

def test_pattern_recognition():
    """Test the candlestick pattern recognition functionality"""
    print("🕯️ Testing Candlestick Pattern Recognition")
    print("=" * 50)
    
    try:
        # Try to import TA-Lib
        import talib
        print("✅ TA-Lib is available")
    except ImportError:
        print("❌ TA-Lib is not installed")
        print("📋 Please follow the installation guide in TALIB_INSTALLATION.md")
        return False
    
    try:
        # Import our pattern analyzer
        from trading_pattern_analyzer import TradingPatternAnalyzer
        print("✅ TradingPatternAnalyzer imported successfully")
    except ImportError:
        print("❌ Cannot import TradingPatternAnalyzer")
        return False
    
    # Create sample data
    print("\n📊 Creating sample cryptocurrency data...")
    df = create_sample_data(60)  # 60 days of data
    print(f"✅ Created {len(df)} days of sample OHLCV data")
    print(f"   Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
    
    # Initialize pattern analyzer
    print("\n🔍 Initializing pattern analyzer...")
    analyzer = TradingPatternAnalyzer()
    print("✅ Pattern analyzer initialized")
    
    # Detect candlestick patterns
    print("\n🕯️ Detecting candlestick patterns...")
    candlestick_analysis = analyzer.detect_candlestick_patterns(df)
    
    if not candlestick_analysis:
        print("❌ No patterns detected - this might indicate an issue")
        return False
    
    # Display results
    stats = candlestick_analysis.get('statistics', {})
    print(f"✅ Pattern detection completed!")
    print(f"   📊 Patterns detected: {stats.get('total_patterns_detected', 0)}")
    print(f"   📈 Total signals: {stats.get('total_signals', 0)}")
    print(f"   📅 Recent patterns: {stats.get('recent_patterns_count', 0)}")
    print(f"   📊 Pattern frequency: {stats.get('pattern_frequency', 0):.2f}%")
    
    # Show pattern sentiment
    sentiment = candlestick_analysis.get('market_sentiment', {})
    if sentiment:
        sentiment_emoji = {
            'bullish': '🟢', 'bearish': '🔴', 
            'mixed': '🟡', 'indecision': '⚪', 'neutral': '⚫'
        }.get(sentiment.get('sentiment', 'neutral'), '⚫')
        
        print(f"\n🎭 Market Sentiment Analysis:")
        print(f"   {sentiment_emoji} Sentiment: {sentiment.get('sentiment', 'neutral').title()}")
        print(f"   🎯 Confidence: {sentiment.get('confidence', 0):.1%}")
        print(f"   🟢 Bullish ratio: {sentiment.get('bullish_ratio', 0):.1%}")
        print(f"   🔴 Bearish ratio: {sentiment.get('bearish_ratio', 0):.1%}")
        print(f"   ⚪ Indecision ratio: {sentiment.get('indecision_ratio', 0):.1%}")
    
    # Show recent patterns
    recent_patterns = candlestick_analysis.get('recent_patterns', [])
    if recent_patterns:
        print(f"\n🔥 Most Recent Patterns (Top 5):")
        for i, pattern in enumerate(recent_patterns[:5], 1):
            signal_emoji = "🟢" if pattern['signal_strength'] > 0 else "🔴"
            print(f"   {i}. {pattern['name']}")
            print(f"      Type: {pattern['type'].replace('_', ' ').title()}")
            print(f"      Signal: {signal_emoji} {pattern['signal_strength']}")
            print(f"      Price: ${pattern['price']:,.2f}")
            print()
    
    # Show pattern categories
    pattern_summary = candlestick_analysis.get('pattern_summary', {})
    print("📋 Pattern Categories Summary:")
    
    categories = [
        ('bullish_reversal', '🟢 Bullish Reversal'),
        ('bearish_reversal', '🔴 Bearish Reversal'),
        ('continuation', '➡️ Continuation'),
        ('indecision', '⚪ Indecision'),
        ('reversal', '🔄 General Reversal')
    ]
    
    for category_key, category_name in categories:
        patterns = pattern_summary.get(category_key, [])
        if patterns:
            print(f"   {category_name}: {len(patterns)} pattern types")
            for pattern in patterns[:3]:  # Show top 3
                print(f"      • {pattern['name']} ({pattern['count']} signals)")
        else:
            print(f"   {category_name}: No patterns detected")
    
    # Test additional analysis features
    print("\n🔍 Testing additional analysis features...")
    
    # Price jumps
    jump_analysis = analyzer.detect_price_jumps(df, jump_threshold=2.0)
    jump_count = jump_analysis.get('jump_up_count', 0) + jump_analysis.get('jump_down_count', 0)
    print(f"   📈 Price jumps detected: {jump_count}")
    
    # Support/Resistance levels
    sr_analysis = analyzer.identify_support_resistance_levels(df)
    levels_count = len(sr_analysis.get('levels', []))
    print(f"   📊 Support/Resistance levels: {levels_count}")
    
    # Market microstructure
    microstructure = analyzer.calculate_market_microstructure_features(df)
    regime = microstructure.get('market_regime', {})
    print(f"   🏛️ Market regime: {regime.get('volatility_regime', 'unknown')} volatility, {regime.get('volume_regime', 'unknown')} volume")
    
    print("\n✅ All tests completed successfully!")
    print("🎉 Candlestick pattern recognition is working correctly!")
    
    return True

def main():
    """Main test function"""
    print("🚀 Candlestick Pattern Recognition Test")
    print("This script tests the new candlestick pattern functionality")
    print("without running the full Streamlit application.\n")
    
    success = test_pattern_recognition()
    
    if success:
        print("\n💡 Next Steps:")
        print("1. Run the full application: python3 run_enhanced_app.py")
        print("2. Enable 'Candlestick Pattern Analysis' in the sidebar")
        print("3. Explore the pattern recognition features!")
        print("\n📚 For TA-Lib installation help, see: TALIB_INSTALLATION.md")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        print("📚 For TA-Lib installation help, see: TALIB_INSTALLATION.md")
        sys.exit(1)

if __name__ == "__main__":
    main() 