#!/usr/bin/env python3
"""
Run script for the Enhanced Crypto Price Predictor
"""

import subprocess
import sys
import os

def run_enhanced_app():
    """Run the enhanced crypto predictor application."""
    try:
        # Ensure we're in the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        print("🚀 Starting Enhanced Crypto Price Predictor...")
        print("📊 Features included:")
        print("   ✅ Enhanced Technical Indicators (30+ indicators)")
        print("   ✅ Ensemble Models (LSTM + Random Forest)")
        print("   ✅ Uncertainty Quantification")
        print("   ✅ Risk Management Metrics")
        print("   ✅ Market Regime Detection")
        print("   ✅ Correlation Analysis")
        print("   ✅ Advanced Visualization")
        print("")
        
        # Run the enhanced streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_enhanced_v2.py"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_enhanced_app() 