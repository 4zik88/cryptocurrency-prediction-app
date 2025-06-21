# 🚀 Local Deployment with Public Access

This guide will help you run the enhanced cryptocurrency prediction app locally with internet access.

## 📋 Prerequisites

### 1. API Keys from Bybit
1. Go to [Bybit API Management](https://www.bybit.com/app/user/api-management)
2. Create **READ-ONLY** API keys (for security)
3. Copy your API Key and Secret

### 2. Configuration Setup
```bash
# Check if secrets file exists
ls -la .streamlit/secrets.toml

# If not exists, copy template
cp secrets.toml.template .streamlit/secrets.toml

# Edit the file with your API keys
nano .streamlit/secrets.toml
```

In `.streamlit/secrets.toml` add:
```toml
BYBIT_API_KEY = "your_api_key_here"
BYBIT_API_SECRET = "your_api_secret_here"
```

## 🚀 Launch Methods

### Method 1: Enhanced App (Recommended)
```bash
# Activate virtual environment
source venv_py310/bin/activate

# Run the enhanced app with SPOT & FUTURES
python3 run_enhanced_app.py
```

### Method 2: Quick Launch Script
```bash
./start_app.sh
```
Choose option 1 (ngrok) when prompted.

### Method 3: Manual Launch
```bash
# Activate environment
source venv_py310/bin/activate

# Run Streamlit directly
streamlit run app_enhanced.py --server.port=8501
```
Then in another terminal:
```bash
ngrok http 8501
```

## 🌐 Result

After successful launch you'll get:

### Local URLs:
- **Application**: http://localhost:8501
- **ngrok Dashboard**: http://localhost:4040

### Public Access:
- **Public URL**: `https://xxxxx.ngrok-free.app`

## 📱 App Features

✅ **SPOT Markets** - Regular cryptocurrency trading:
- Bitcoin, Ethereum, and other popular pairs
- LSTM neural network price predictions
- Technical indicators

✅ **FUTURES Markets** - Futures trading:
- Futures contracts
- Additional metrics (leverage, risk/reward)
- Specialized indicators

✅ **AI Functionality:**
- Predictions for 1 hour, 4 hours, 8 hours, 1 day
- Trading signals (Long/Short/Hold)
- Volatility and trend analysis

✅ **Multi-language Support:**
- English and Ukrainian interface

## 🎯 How to Use

1. **Open the public URL**: `https://xxxxx.ngrok-free.app`
2. **Select market type**: SPOT or FUTURES
3. **Choose cryptocurrency** (e.g., BTCUSDT)
4. **Configure prediction parameters**
5. **Get predictions and trading signals**

## ⚠️ Important Notes

### Security:
- Use only **READ-ONLY** API keys
- Never share your API keys publicly
- `.streamlit/secrets.toml` should not be committed to Git

### Free Tunnel Limitations:
- **Free ngrok**: URL changes on restart
- For production use, consider paid ngrok or VPS

### Performance:
- First launch may take 1-2 minutes
- Model training happens automatically
- Models are saved for reuse

## 🆘 Troubleshooting

### "Module not found" error:
```bash
source venv_py310/bin/activate
pip install -r requirements.txt
```

### API key errors:
- Check keys in `.streamlit/secrets.toml`
- Ensure keys have read permissions

### Tunnel issues:
- Try restarting the application
- Check internet connection
- Verify ngrok is properly configured

### Slow performance:
- Increase prediction timeframe
- Reduce historical data days
- Close other resource-intensive apps

## 📞 Share the Link

After launch, you'll get a public link like:
- `https://abc123.ngrok-free.app`

Share this link with anyone for access to your application!

## 🛑 Stop Application

Press `Ctrl+C` in terminal to properly stop all processes.

---

**Ready!** 🎉 Your cryptocurrency prediction app is running and accessible via the internet!

## 🔧 Advanced Configuration

### Custom Domain (ngrok Pro):
```bash
ngrok http 8501 --hostname=your-domain.ngrok.io
```

### Background Running:
```bash
nohup python3 run_enhanced_app.py &
```

### Docker Alternative:
```bash
docker-compose up -d
```

## 📊 Application Structure

- `app_enhanced.py` - Main application with SPOT & FUTURES
- `run_enhanced_app.py` - Launch script with ngrok
- `data_loader.py` - SPOT market data handling
- `futures_data_loader.py` - FUTURES market data handling
- `predictor.py` - SPOT market predictions
- `futures_predictor.py` - FUTURES market predictions 