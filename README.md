# 🚀 Crypto AI Prediction App

Enhanced cryptocurrency prediction application with SPOT and FUTURES market support, now featuring **CryptoCompare Market Intelligence** for comprehensive analysis of ALL cryptocurrencies.

## 📋 Features

- 📈 **SPOT Markets** - Regular cryptocurrency trading analysis
- 🚀 **FUTURES Markets** - Futures trading analysis and predictions  
- 🤖 **AI Predictions** - LSTM-based predictions for 1h, 4h, 8h, 24h horizons
- 📊 **Technical Analysis** - MACD, RSI, Bollinger Bands, and more
- 🕯️ **Candlestick Pattern Recognition** - AI-powered detection of 61+ candlestick patterns
- 🌐 **CryptoCompare Integration** - Enhanced predictions for ALL cryptocurrencies with comprehensive market data
- 🧠 **Advanced Sentiment Analysis** - Multi-factor sentiment scoring for every cryptocurrency
- 🔍 **Multi-Source Validation** - Price comparison and cross-validation across data sources
- 🌍 **Multi-language** - Russian and English support
- 🌐 **Public Access** - ngrok tunnel for sharing

## 🏃‍♂️ Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Configuration**
   - Create `.streamlit/secrets.toml` with your Bybit API credentials
   
3. **Run Application**
   ```bash
   python3 run_enhanced_app.py
   ```

## 📁 Core Files

- `run_enhanced_app.py` - Main application runner
- `app_enhanced.py` - Streamlit web application
- `data_loader.py` - SPOT market data handling
- `futures_data_loader.py` - FUTURES market data handling
- `cryptocompare_data_loader.py` - CryptoCompare API integration for universal cryptocurrency analysis
- `trading_pattern_analyzer.py` - **NEW**: Advanced pattern recognition with 61+ candlestick patterns
- `predictor.py` - LSTM predictor for SPOT markets
- `futures_predictor.py` - Enhanced LSTM predictor for FUTURES
- `translations.py` - Multi-language support
- `TALIB_INSTALLATION.md` - **NEW**: TA-Lib installation guide for pattern recognition

## 🔧 Configuration

Create `.streamlit/secrets.toml`:
```toml
[bybit]
api_key = "your_api_key"
api_secret = "your_api_secret"
testnet = true

# CryptoCompare API key for enhanced market intelligence
CRYPTOCOMPARE_API_KEY = "your_cryptocompare_api_key"
```

## 🌐 Access

- **Local**: http://localhost:8501
- **Public**: Generated ngrok URL (displayed on startup)

## 📊 Supported Markets

- **SPOT**: BTC, ETH, and 600+ cryptocurrency pairs
- **FUTURES**: Major cryptocurrency futures contracts

## 🆕 CryptoCompare Integration Features

### For ALL Cryptocurrencies (BTC, ETH, ADA, SOL, and 1000+ others):
- 🌐 **Universal Coverage** - Works with every cryptocurrency in spot and futures markets
- 📊 **Real-time Market Metrics** - Price, volume, market cap, 24h changes for all coins
- 🧠 **Advanced Sentiment Analysis** - Multi-factor sentiment scoring (0-100 scale)
- 📈 **Social Sentiment Tracking** - Reddit, Twitter, Facebook activity monitoring
- 📰 **News Sentiment Analysis** - Real-time news sentiment with ML scoring
- 🔍 **Price Validation** - Cross-validation between exchange and CryptoCompare data
- 🎯 **Enhanced Prediction Confidence** - Sentiment-adjusted confidence scoring
- ⚠️ **Risk & Opportunity Detection** - Automated market risk and opportunity identification

### Multi-Factor Sentiment Analysis:
- **Price Momentum** (40% weight) - 24h price movement and trend direction
- **Volume Strength** (20% weight) - Trading volume and liquidity indicators
- **Social Activity** (15% weight) - Reddit posts, Twitter mentions, social engagement
- **News Sentiment** (15% weight) - Recent news articles with sentiment scoring
- **Technical Indicators** (10% weight) - RSI, moving averages, trend analysis

### Technical Features:
- Professional CryptoCompare API integration with authentication
- Supports 1000+ cryptocurrencies across all major exchanges
- Real-time social media and news sentiment analysis
- Advanced sentiment scoring with confidence levels
- Historical data analysis for trend identification
- Multi-source data validation for improved accuracy

### Setup:
Add your CryptoCompare API key to environment variables:
```bash
export CRYPTOCOMPARE_API_KEY="your_api_key_here"
```

## 🕯️ Candlestick Pattern Recognition

Advanced AI-powered candlestick pattern recognition system using TA-Lib for professional technical analysis.

### Features:
- **61+ Pattern Detection** - Complete library of candlestick patterns including:
  - **Reversal Patterns**: Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star
  - **Continuation Patterns**: Three White Soldiers, Three Black Crows, Rising/Falling Three Methods
  - **Indecision Patterns**: Spinning Top, High-Wave Candle, Long-Legged Doji
- **Visual Pattern Markers** - Patterns highlighted directly on price charts with color coding
- **Market Sentiment Analysis** - AI-powered sentiment scoring based on detected patterns
- **Pattern Classification** - Automatic categorization by bullish/bearish and reversal/continuation
- **Educational Insights** - Learn about each pattern as it's detected
- **Trading Signals** - Enhanced signals combining patterns with volume and support/resistance

### Pattern Categories:
- 🟢 **Bullish Reversal** - Patterns suggesting upward price reversals
- 🔴 **Bearish Reversal** - Patterns suggesting downward price reversals  
- ➡️ **Continuation** - Patterns suggesting trend continuation
- ⚪ **Indecision** - Patterns showing market uncertainty
- 🔄 **General Reversal** - Patterns that can be bullish or bearish

### Technical Implementation:
- **TA-Lib Integration** - Professional-grade pattern recognition library
- **Real-time Detection** - Patterns detected on every data update
- **Historical Analysis** - Pattern frequency and reliability tracking
- **Chart Integration** - Seamlessly integrated with existing technical analysis charts
- **Performance Optimized** - Efficient pattern scanning with caching

### Installation:
The candlestick pattern recognition requires TA-Lib installation. See `TALIB_INSTALLATION.md` for detailed setup instructions.

```bash
# Quick install (macOS with Homebrew)
brew install ta-lib
pip install TA-Lib

# For other systems, see TALIB_INSTALLATION.md
```

### Usage:
1. Enable "Candlestick Pattern Analysis" in the sidebar
2. Patterns will be automatically detected and displayed
3. View pattern categories in organized tabs
4. Check the enhanced chart with pattern markers
5. Read educational insights about detected patterns

## Language Support

This application now supports **full internationalization** with proper language separation:

### Supported Languages
- **English** (`en`) - Complete interface in English
- **Russian** (`ru`) - Complete interface in Russian

### Language Features
- Dynamic language switching in the app interface
- Console output language can be configured via environment variable
- No mixed language content - clean separation between English and Russian
- All user-facing text is properly translated

### Console Language Configuration
You can set the console language for the runner script:
```bash
# For English console output
export CONSOLE_LANGUAGE=en
python3 run_enhanced_app.py

# For Russian console output (default)
export CONSOLE_LANGUAGE=ru
python3 run_enhanced_app.py
```

## Development

The application uses Streamlit for the web interface and includes comprehensive language support through the `translations.py` module.

## Deployment

### Automated Deployment (Recommended)

This setup provides:
- **High Availability**: 3 application instances with load balancing
- **Zero Downtime**: Rolling deployments
- **Auto-scaling**: Handle high concurrent loads
- **CI/CD**: Automatic deployment on git push
- **Security**: Rate limiting, security headers, firewall

#### 1. Server Setup
```bash
# Run this on your server as root
sudo bash deploy.sh
```

#### 2. GitHub Actions Setup
Add these secrets to your GitHub repository:
- `HOST`: Your server IP address
- `USERNAME`: Server username (usually 'appuser')  
- `SSH_KEY`: Private SSH key for server access
- `PORT`: SSH port (usually 22)
- `CRYPTOCOMPARE_API_KEY`: Your CryptoCompare API key

#### 3. Environment Configuration
```bash
# On your server, edit the environment file
sudo nano /opt/ai-crypto-app/.env

# Add your actual API keys:
CRYPTOCOMPARE_API_KEY=your_actual_api_key_here
CONSOLE_LANGUAGE=en
```

#### 4. SSL Certificate (Optional but Recommended)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### Manual Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Scale for high load
docker-compose up -d --scale app1=3 --scale app2=3 --scale app3=3

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Architecture 🏗️

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │     (nginx)     │
                    └─────────┬───────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
        ┌───────▼──┐  ┌───────▼──┐  ┌───────▼──┐
        │   App 1  │  │   App 2  │  │   App 3  │
        │Streamlit │  │Streamlit │  │Streamlit │
        └──────────┘  └──────────┘  └──────────┘
```

### Key Components:
- **Load Balancer**: Nginx with round-robin distribution
- **Application Instances**: Multiple Streamlit containers
- **Health Checks**: Automatic failure detection and recovery
- **WebSocket Support**: Real-time updates and interactions
- **Rate Limiting**: Protection against abuse

## Performance Optimization 🚄

### High Load Configuration
The deployment is configured to handle high concurrent loads:

1. **Multiple Instances**: 3 app instances by default
2. **Load Balancing**: Least-connection algorithm
3. **Connection Pooling**: Efficient resource usage
4. **Caching**: Static file caching and API response caching
5. **Rate Limiting**: 10 requests/second per IP (configurable)

### Scaling Up
```bash
# Scale to more instances
docker-compose up -d --scale app1=5 --scale app2=5 --scale app3=5

# Or use Docker Swarm for auto-scaling
docker swarm init
docker stack deploy -c docker-compose.yml ai-crypto
```

## Monitoring 📊

### Health Checks
```bash
# Check application health
python health_check.py

# JSON output for monitoring systems
python health_check.py --json

# Check specific services
curl http://your-server/health
```

### Logs
```bash
# View application logs
docker-compose logs -f

# View nginx logs
docker-compose logs nginx

# System monitoring
htop
docker stats
```

## Security 🔒

### Built-in Security Features:
- **Rate Limiting**: Prevents abuse
- **Security Headers**: XSS, CSRF protection
- **Firewall**: UFW configured for essential ports only
- **Non-root User**: Application runs as non-privileged user
- **Container Isolation**: Each service in separate container

### Additional Security:
```bash
# Enable fail2ban
sudo apt install fail2ban

# Configure SSH key-only authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no

# Regular updates
sudo apt update && sudo apt upgrade -y
```

## Troubleshooting 🔧

### Common Issues:

#### Application Won't Start
```bash
# Check logs
docker-compose logs app1

# Check environment variables
docker-compose exec app1 env | grep API

# Rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Optimize Python memory
export PYTHONOPTIMIZE=1

# Scale down if needed
docker-compose up -d --scale app1=1 --scale app2=1
```

#### SSL Certificate Issues
```bash
# Renew certificate
sudo certbot renew

# Test certificate
sudo nginx -t
sudo systemctl reload nginx
```

## API Keys Setup 🔑

### Required APIs:
1. **CryptoCompare**: Get from https://cryptocompare.com/cryptopian/api-keys
2. **Binance**: Optional, for additional data sources

### Configuration:
```bash
# Method 1: Environment variables
export CRYPTOCOMPARE_API_KEY="your_key_here"

# Method 2: .env file
echo "CRYPTOCOMPARE_API_KEY=your_key_here" >> .env

# Method 3: Streamlit secrets
echo 'CRYPTOCOMPARE_API_KEY = "your_key_here"' >> .streamlit/secrets.toml
```

## Development 👨‍💻

### Adding New Features:
1. Make changes locally
2. Test thoroughly
3. Push to `main` branch
4. GitHub Actions will automatically deploy

### Local Testing:
```bash
# Run tests
python -c "import app_enhanced; print('✅ App imports successfully')"

# Check dependencies
python health_check.py

# Load test (optional)
pip install locust
locust -f load_test.py --host=http://localhost:8501
```

## Support 💬

### Logs Location:
- Application: `docker-compose logs`
- Nginx: `/var/log/nginx/`
- System: `/var/log/syslog`

### Performance Monitoring:
- CPU/Memory: `htop`, `docker stats`
- Network: `iftop`, `netstat`
- Disk: `df -h`, `du -sh`

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Happy Trading! 📈🚀**

For support or questions, please open an issue or contact the development team. 