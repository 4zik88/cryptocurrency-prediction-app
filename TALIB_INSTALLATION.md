# TA-Lib Installation Guide for Candlestick Pattern Recognition

TA-Lib (Technical Analysis Library) is required for candlestick pattern recognition functionality. Follow the instructions below for your operating system.

## 📋 Prerequisites

Before installing TA-Lib, ensure you have:
- Python 3.7+ installed
- pip package manager
- Virtual environment activated (recommended)

## 🍎 macOS Installation

### Method 1: Using Homebrew (Recommended)
```bash
# Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install TA-Lib dependencies
brew install ta-lib

# Install Python wrapper
pip install TA-Lib
```

### Method 2: Manual Installation
```bash
# Download and compile TA-Lib
curl -O https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr/local
make
sudo make install

# Install Python wrapper
pip install TA-Lib
```

## 🐧 Linux (Ubuntu/Debian) Installation

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential

# Download and compile TA-Lib
wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Update library cache
sudo ldconfig

# Install Python wrapper
pip install TA-Lib
```

## 🪟 Windows Installation

### Method 1: Using Pre-compiled Wheels (Easiest)
```bash
# For 64-bit Python
pip install --only-binary=all TA-Lib

# If the above fails, try:
pip install TA-Lib==0.4.25
```

### Method 2: Using Conda
```bash
# Install using conda-forge
conda install -c conda-forge ta-lib
```

### Method 3: Manual Installation
1. Download pre-compiled TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Choose the appropriate `.whl` file for your Python version
3. Install using pip:
```bash
pip install path/to/downloaded/TA_Lib-0.4.xx-cpXX-cpXXm-win_amd64.whl
```

## 🐳 Docker Installation

If you're using Docker, add this to your Dockerfile:

```dockerfile
# Install TA-Lib dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

RUN ldconfig

# Install Python wrapper
RUN pip install TA-Lib
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Microsoft Visual C++ 14.0 is required" (Windows)
- Install Visual Studio Build Tools
- Or use pre-compiled wheels as shown above

#### 2. "ta_libc.h: No such file or directory" (Linux/macOS)
```bash
# Make sure TA-Lib C library is installed first
export LDFLAGS="-L/usr/local/lib"
export CPPFLAGS="-I/usr/local/include"
pip install TA-Lib
```

#### 3. Import Error: "No module named 'talib'"
```bash
# Reinstall with verbose output to see errors
pip install --upgrade --force-reinstall TA-Lib -v
```

#### 4. "Library not loaded" (macOS)
```bash
# Fix library path issues
export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"
```

## ✅ Verification

Test your installation:

```python
import talib
import numpy as np

# Test data
open_prices = np.array([1, 2, 3, 4, 5], dtype=float)
high_prices = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=float)
low_prices = np.array([0.9, 1.9, 2.9, 3.9, 4.9], dtype=float)
close_prices = np.array([1.05, 2.05, 3.05, 4.05, 5.05], dtype=float)

# Test pattern detection
doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
print("TA-Lib installation successful!")
print(f"Doji pattern detected: {doji}")
```

## 🚀 Features Enabled

Once TA-Lib is installed, you'll have access to:

- **61+ Candlestick Patterns**: Including Doji, Hammer, Engulfing, Morning/Evening Star, and more
- **Pattern Classification**: Bullish/Bearish reversal, continuation, and indecision patterns
- **Visual Pattern Markers**: Patterns marked directly on your price charts
- **Market Sentiment Analysis**: AI-powered sentiment based on detected patterns
- **Educational Insights**: Learn about each pattern as it's detected
- **Trading Signals**: Enhanced signals combining patterns with other indicators

## 📚 Pattern Categories Available

### Reversal Patterns
- Doji, Hammer, Shooting Star
- Engulfing Patterns
- Morning/Evening Star
- Harami Patterns

### Continuation Patterns
- Three White Soldiers / Three Black Crows
- Rising/Falling Three Methods
- Mat Hold Pattern

### Indecision Patterns
- Spinning Top
- High-Wave Candle
- Long-Legged Doji

## 💡 Usage Tips

1. **Enable Pattern Analysis**: Use the checkbox in the sidebar to enable/disable pattern recognition
2. **Combine with Volume**: Patterns are more reliable when confirmed by volume spikes
3. **Check Support/Resistance**: Patterns near key levels are more significant
4. **Consider Market Context**: Patterns work better in certain market conditions

## 🆘 Need Help?

If you're still having issues:
1. Check your Python version compatibility
2. Try installing in a fresh virtual environment
3. Ensure you have the latest pip version: `pip install --upgrade pip`
4. Check the official TA-Lib documentation: https://ta-lib.org/

## 🔄 Alternative Solutions

If you can't install TA-Lib, the application will continue to work without candlestick pattern recognition. Other technical analysis features will still be available:

- Price jump detection
- Support/resistance levels
- Volume analysis
- Moving averages
- MACD, RSI, Stochastic indicators
- Ichimoku analysis

---

**Note**: The candlestick pattern functionality will be automatically disabled if TA-Lib is not available, and the application will show a helpful message guiding you to this installation guide. 