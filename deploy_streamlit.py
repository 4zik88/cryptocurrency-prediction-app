#!/usr/bin/env python3
"""
Streamlit Cloud Deployment Test Script
Run this locally to verify your app is ready for deployment
"""

import os
import sys
import subprocess
import importlib.util

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'streamlit',
        'tensorflow',
        'pandas',
        'numpy',
        'pybit',
        'ta',
        'sklearn',  # scikit-learn imports as sklearn
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def check_files():
    """Check if all required files exist."""
    required_files = [
        'app_enhanced.py',
        'requirements.txt',
        'data_loader.py',
        'futures_data_loader.py',
        'enhanced_predictor.py',
        'translations.py',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files exist")
        return True

def check_secrets():
    """Check if secrets are configured."""
    secrets_file = '.streamlit/secrets.toml'
    
    if os.path.exists(secrets_file):
        print("✅ Streamlit secrets file exists")
        print("⚠️  Make sure your API keys are configured in secrets.toml")
        return True
    else:
        print("⚠️  No local secrets.toml found")
        print("📝 You'll need to configure secrets in Streamlit Cloud dashboard")
        return True

def test_app():
    """Test if the app modules can be imported."""
    print("\n🧪 Testing app modules...")
    try:
        # Test individual module imports instead of full app
        import data_loader
        import futures_data_loader
        import enhanced_predictor
        import translations
        print("✅ All app modules can be imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing modules: {str(e)}")
        return False

def main():
    """Main deployment check function."""
    print("🚀 Streamlit Cloud Deployment Check")
    print("=" * 40)
    
    checks = [
        ("📦 Checking required packages", check_requirements),
        ("📁 Checking required files", check_files),
        ("🔐 Checking secrets configuration", check_secrets),
        ("🧪 Testing module imports", test_app)
    ]
    
    all_passed = True
    
    for description, check_func in checks:
        print(f"\n{description}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    
    if all_passed:
        print("🎉 Your app is ready for Streamlit Cloud deployment!")
        print("\n📋 Next steps:")
        print("1. Push your code to GitHub")
        print("2. Go to https://share.streamlit.io")
        print("3. Connect your repository")
        print("4. Set main file to: app_enhanced.py")
        print("5. Configure secrets in the dashboard")
        print("6. Deploy!")
    else:
        print("❌ Please fix the issues above before deploying")
        sys.exit(1)

if __name__ == "__main__":
    main() 