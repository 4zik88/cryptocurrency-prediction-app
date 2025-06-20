# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. **GitHub Repository**: Your code must be in a GitHub repository
2. **Bybit API Keys**: Get your API keys from [Bybit](https://www.bybit.com/app/user/api-management)
3. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## 🔧 Deployment Steps

### 1. Prepare Your Repository

Make sure your repository has these files:
- ✅ `app_enhanced.py` (main application)
- ✅ `requirements.txt` (dependencies)
- ✅ `.streamlit/config.toml` (configuration)
- ✅ All model files (`.h5`, `.pkl`)
- ✅ All Python modules (`data_loader.py`, `futures_data_loader.py`, etc.)

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set the main file path: `app_enhanced.py`
6. Click "Deploy!"

### 3. Configure Secrets

In your Streamlit Cloud dashboard:

1. Go to your app settings
2. Click on "Secrets"
3. Add your API credentials:

```toml
BYBIT_API_KEY = "your_actual_api_key"
BYBIT_API_SECRET = "your_actual_api_secret"
```

### 4. Update Code for Streamlit Secrets

Your code is already configured to use Streamlit secrets! The data loaders check for:
1. Streamlit secrets first (`st.secrets`)
2. Environment variables second (`.env` file)

## 🔐 Security Best Practices

### ✅ DO:
- Use Streamlit Cloud secrets for API keys
- Add `secrets.toml` to `.gitignore`
- Use read-only API keys when possible
- Monitor your API usage

### ❌ DON'T:
- Commit API keys to GitHub
- Use API keys with trading permissions
- Share your secrets.toml file

## 📁 File Structure

```
your-repo/
├── app_enhanced.py              # Main Streamlit app
├── requirements.txt             # Dependencies
├── .streamlit/
│   └── config.toml             # Streamlit config
├── secrets.toml.template       # Template for secrets
├── data_loader.py              # Spot data loader
├── futures_data_loader.py      # Futures data loader
├── enhanced_predictor.py       # ML predictor
├── translations.py             # Multi-language support
├── *.h5                        # LSTM models
├── *.pkl                       # Random Forest models
└── README.md                   # Documentation
```

## 🚨 Common Issues & Solutions

### Issue 1: Import Errors
**Problem**: Module not found errors
**Solution**: Ensure all Python files are in the root directory

### Issue 2: Model Loading Errors
**Problem**: Cannot load `.h5` or `.pkl` files
**Solution**: Ensure model files are committed to GitHub (check file size limits)

### Issue 3: API Connection Errors
**Problem**: Bybit API authentication fails
**Solution**: Double-check your secrets configuration in Streamlit Cloud

### Issue 4: Memory Issues
**Problem**: App crashes due to memory limits
**Solution**: Streamlit Cloud has memory limits. Consider:
- Using smaller models
- Implementing model caching
- Reducing data lookback periods

## 🔧 Performance Optimization

1. **Caching**: Use `@st.cache_resource` for models
2. **Data Loading**: Limit historical data to essential periods
3. **Model Size**: Use compressed models when possible
4. **API Calls**: Implement rate limiting and caching

## 📊 Monitoring

After deployment, monitor:
- App performance and load times
- API usage and rate limits
- Error logs in Streamlit Cloud dashboard
- Memory and CPU usage

## 🆘 Troubleshooting

If your app fails to deploy:

1. Check the logs in Streamlit Cloud dashboard
2. Verify all dependencies in `requirements.txt`
3. Test locally with `streamlit run app_enhanced.py`
4. Check GitHub repository permissions
5. Ensure all files are properly committed

## 🎯 Next Steps

After successful deployment:
1. Test all features thoroughly
2. Share your app URL with users
3. Monitor performance and usage
4. Consider upgrading to Streamlit Cloud Pro for better performance

## 📞 Support

- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

---

🎉 **Your crypto prediction app is now ready for the cloud!** 