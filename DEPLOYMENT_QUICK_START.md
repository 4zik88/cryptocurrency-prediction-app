# 🚀 Quick Deployment to Streamlit Cloud

## ⚡ 5-Minute Setup

### 1. Test Locally (Optional)
```bash
python deploy_streamlit.py
```

### 2. Push to GitHub
```bash
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

### 3. Deploy on Streamlit Cloud

1. **Go to**: [share.streamlit.io](https://share.streamlit.io)
2. **Click**: "New app"
3. **Repository**: Select your GitHub repo
4. **Main file**: `app_enhanced.py`
5. **Click**: "Deploy"

### 4. Configure API Keys

In your Streamlit Cloud app dashboard:

1. **Settings** → **Secrets**
2. **Add**:
```toml
BYBIT_API_KEY = "your_actual_api_key_here"
BYBIT_API_SECRET = "your_actual_secret_here"
```

### 5. Get Your Bybit API Keys

1. Go to [Bybit API Management](https://www.bybit.com/app/user/api-management)
2. Create **Read-Only** API keys
3. Copy API Key and Secret

## ✅ That's it!

Your crypto prediction app is now live on Streamlit Cloud!

## 🔗 Features Available

- ✅ Real-time crypto price predictions
- ✅ Spot and Futures market analysis
- ✅ Technical indicators (MACD, Stochastic, RSI, etc.)
- ✅ Multi-language support
- ✅ Interactive charts with Plotly
- ✅ AI-powered trading signals

## 🚨 Important Notes

- Use **READ-ONLY** API keys only
- Monitor your API usage limits
- App may take 1-2 minutes to start initially
- Free Streamlit Cloud has resource limits

## 🆘 Need Help?

Check the full guide: `STREAMLIT_CLOUD_DEPLOYMENT.md` 