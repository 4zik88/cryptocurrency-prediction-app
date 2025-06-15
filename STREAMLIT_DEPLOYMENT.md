# Streamlit Cloud Deployment Guide

## File Structure for Deployment

Due to Python 3.13 compatibility issues with TensorFlow on Streamlit Cloud, we've organized the files as follows:

### Main Files (Used by Streamlit Cloud)
- `app_enhanced.py` - **Demo version** (no TensorFlow, works on Streamlit Cloud)
- `requirements.txt` - **Lightweight dependencies** (no TensorFlow/scikit-learn)

### Full Version Files (For local/server deployment)
- `app_full.py` - **Full version** with TensorFlow and ML models
- `requirements-full.txt` - **Complete dependencies** including TensorFlow

## Deployment Strategy

### Streamlit Cloud (Public Demo)
- Uses `app_enhanced.py` (demo version)
- Uses `requirements.txt` (lightweight)
- No API keys required
- Shows all UI features with demo data
- Compatible with Python 3.13

### Local/Server Deployment (Full Features)
- Use `app_full.py` 
- Use `requirements-full.txt`
- Requires API keys in `.env`
- Real-time data and ML predictions
- Requires Python 3.10-3.12

## Features Comparison

| Feature | Demo Version | Full Version |
|---------|-------------|--------------|
| UI Interface | ✅ Complete | ✅ Complete |
| Ukrainian Translation | ✅ Yes | ✅ Yes |
| Spot Market | ✅ Demo data | ✅ Real API |
| Futures Market | ✅ Demo data | ✅ Real API |
| Technical Indicators | ✅ All indicators | ✅ All indicators |
| ML Predictions | ✅ Demo predictions | ✅ Real LSTM models |
| Trading Signals | ✅ Demo signals | ✅ Real signals |
| Risk Analysis | ✅ Demo metrics | ✅ Real metrics |

## Quick Start

### For Streamlit Cloud
1. Fork the repository
2. Deploy on Streamlit Cloud using default settings
3. Main file: `app_enhanced.py` (auto-detected)
4. Requirements: `requirements.txt` (auto-detected)

### For Full Local Setup
```bash
# Use full version files
cp app_full.py app.py
pip install -r requirements-full.txt

# Add your API keys to .env
cp .env.example .env
# Edit .env with your Bybit API credentials

# Run the app
streamlit run app.py
```

## Troubleshooting

### TensorFlow Issues on Streamlit Cloud
- **Problem**: Python 3.13 doesn't support TensorFlow yet
- **Solution**: Use the demo version (already configured)

### Missing API Data
- **Demo version**: Uses generated demo data (normal behavior)
- **Full version**: Check your `.env` file and API credentials

### Import Errors
- **Demo version**: Should work with minimal dependencies
- **Full version**: Ensure all packages in `requirements-full.txt` are installed 