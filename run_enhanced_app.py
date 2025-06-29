#!/usr/bin/env python3
"""
Enhanced App Runner with Multi-language Support
Запуск app_enhanced.py с SPOT и FUTURES рынками
"""

import subprocess
import sys
import time
import requests
from pathlib import Path
import os

# Set up language for console output
# You can change this to "en" for English or "ru" for Russian
CONSOLE_LANGUAGE = os.getenv('CONSOLE_LANGUAGE', 'ru')  # Default to Russian for backwards compatibility

# Import translation system
try:
    from translations import get_text
except ImportError:
    # Fallback if translations module is not available
    def get_text(key, lang="en"):
        translations = {
            "en": {
                "secrets_file_not_found": "❌ File .streamlit/secrets.toml not found!",
                "app_file_not_found": "❌ File app_enhanced.py not found!",
                "launching_app": "🚀 Launching app_enhanced.py with SPOT and FUTURES markets...",
                "waiting_for_startup": "⏳ Waiting for application startup...",
                "app_started_successfully": "✅ Application started successfully!",
                "attempt": "⏳ Attempt",
                "creating_public_tunnel": "🌐 Creating public tunnel...",
                "ngrok_not_found": "❌ ngrok not found!",
                "stopping_processes": "🛑 Stopping processes...",
                "streamlit_stopped": "✅ Streamlit stopped",
                "ngrok_stopped": "✅ ngrok stopped",
                "app_launch_title": "🚀 LAUNCHING app_enhanced.py",
                "functionality": "📊 Functionality:",
                "spot_markets": "   • SPOT markets",
                "futures_markets": "   • FUTURES markets",
                "ai_predictions": "   • AI predictions",
                "technical_analysis": "   • Technical analysis",
                "failed_to_start": "❌ Failed to start application",
                "app_running": "✅ APPLICATION RUNNING!",
                "public_url": "🌐 Public URL:",
                "local_url": "📱 Local URL: http://localhost:8501",
                "check_url": "Check http://localhost:4040",
                "available_features": "🎯 Available in the app:",
                "spot_trading": "   📈 SPOT - regular trading",
                "futures_trading": "   🚀 FUTURES - futures",
                "ai_forecasts": "   🤖 AI forecasts (1h, 4h, 8h, 24h)",
                "languages": "   🌍 Russian/English",
                "share_link": "💡 Share the link for access!",
                "stop_instruction": "🛑 Ctrl+C to stop",
                "stopping": "⏹️  Stopping...",
                "error_message": "❌ Error:",
            },
            "ru": {
                "secrets_file_not_found": "❌ Файл .streamlit/secrets.toml не найден!",
                "app_file_not_found": "❌ Файл app_enhanced.py не найден!",
                "launching_app": "🚀 Запускаю app_enhanced.py с SPOT и FUTURES рынками...",
                "waiting_for_startup": "⏳ Ожидаю запуска приложения...",
                "app_started_successfully": "✅ Приложение запущено успешно!",
                "attempt": "⏳ Попытка",
                "creating_public_tunnel": "🌐 Создаю публичный туннель...",
                "ngrok_not_found": "❌ ngrok не найден!",
                "stopping_processes": "🛑 Останавливаю процессы...",
                "streamlit_stopped": "✅ Streamlit остановлен",
                "ngrok_stopped": "✅ ngrok остановлен",
                "app_launch_title": "🚀 ЗАПУСК app_enhanced.py",
                "functionality": "📊 Функционал:",
                "spot_markets": "   • SPOT рынки",
                "futures_markets": "   • FUTURES рынки",
                "ai_predictions": "   • AI прогнозы",
                "technical_analysis": "   • Технический анализ",
                "failed_to_start": "❌ Не удалось запустить приложение",
                "app_running": "✅ ПРИЛОЖЕНИЕ ЗАПУЩЕНО!",
                "public_url": "🌐 Публичный URL:",
                "local_url": "📱 Локальный URL: http://localhost:8501",
                "check_url": "Проверьте http://localhost:4040",
                "available_features": "🎯 В приложении доступно:",
                "spot_trading": "   📈 SPOT - обычная торговля",
                "futures_trading": "   🚀 FUTURES - фьючерсы",
                "ai_forecasts": "   🤖 AI прогнозы (1ч, 4ч, 8ч, 24ч)",
                "languages": "   🌍 Русский/Английский",
                "share_link": "💡 Поделитесь ссылкой для доступа!",
                "stop_instruction": "🛑 Ctrl+C для остановки",
                "stopping": "⏹️  Остановка...",
                "error_message": "❌ Ошибка:",
            }
        }
        return translations.get(lang, {}).get(key, key)

def check_config():
    """Проверка конфигурации / Configuration check"""
    secrets_file = Path(".streamlit/secrets.toml")
    if not secrets_file.exists():
        print(get_text("secrets_file_not_found", CONSOLE_LANGUAGE))
        return False
    
    if not Path("app_enhanced.py").exists():
        print(get_text("app_file_not_found", CONSOLE_LANGUAGE))
        return False
    
    return True

def start_streamlit():
    """Запуск app_enhanced.py / Start app_enhanced.py"""
    print(get_text("launching_app", CONSOLE_LANGUAGE))
    
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app_enhanced.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])
    
    return process

def wait_for_streamlit():
    """Ожидание запуска / Wait for startup"""
    print(get_text("waiting_for_startup", CONSOLE_LANGUAGE))
    for i in range(30):
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=2)
            if response.status_code == 200:
                print(get_text("app_started_successfully", CONSOLE_LANGUAGE))
                return True
        except:
            time.sleep(1)
            if i % 5 == 0:
                print(f"{get_text('attempt', CONSOLE_LANGUAGE)} {i+1}/30...")
    
    return False

def start_ngrok():
    """Запуск ngrok / Start ngrok"""
    print(get_text("creating_public_tunnel", CONSOLE_LANGUAGE))
    
    try:
        subprocess.run(["ngrok", "version"], capture_output=True, check=True)
    except:
        print(get_text("ngrok_not_found", CONSOLE_LANGUAGE))
        return None
    
    process = subprocess.Popen([
        "ngrok", "http", "8501", "--log=stdout"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    time.sleep(3)
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        if response.status_code == 200:
            tunnels = response.json()
            if tunnels["tunnels"]:
                public_url = tunnels["tunnels"][0]["public_url"]
                return process, public_url
    except:
        pass
    
    return process, None

def cleanup(streamlit_process, ngrok_process):
    """Очистка / Cleanup"""
    print(f"\n{get_text('stopping_processes', CONSOLE_LANGUAGE)}")
    
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()
        print(get_text("streamlit_stopped", CONSOLE_LANGUAGE))
    
    if ngrok_process:
        ngrok_process.terminate() 
        ngrok_process.wait()
        print(get_text("ngrok_stopped", CONSOLE_LANGUAGE))

def main():
    """Главная функция / Main function"""
    print(get_text("app_launch_title", CONSOLE_LANGUAGE))
    print("=" * 40)
    print(get_text("functionality", CONSOLE_LANGUAGE))
    print(get_text("spot_markets", CONSOLE_LANGUAGE))
    print(get_text("futures_markets", CONSOLE_LANGUAGE))
    print(get_text("ai_predictions", CONSOLE_LANGUAGE))
    print(get_text("technical_analysis", CONSOLE_LANGUAGE))
    print("=" * 40)
    
    if not check_config():
        return 1
    
    streamlit_process = None
    ngrok_process = None
    
    try:
        streamlit_process = start_streamlit()
        
        if not wait_for_streamlit():
            print(get_text("failed_to_start", CONSOLE_LANGUAGE))
            return 1
        
        result = start_ngrok()
        if result:
            ngrok_process, public_url = result
        
        print("\n" + "=" * 40)
        print(get_text("app_running", CONSOLE_LANGUAGE))
        print("=" * 40)
        print(f"{get_text('public_url', CONSOLE_LANGUAGE)} {public_url or get_text('check_url', CONSOLE_LANGUAGE)}")
        print(get_text("local_url", CONSOLE_LANGUAGE))
        
        print(f"\n{get_text('available_features', CONSOLE_LANGUAGE)}")
        print(get_text("spot_trading", CONSOLE_LANGUAGE))
        print(get_text("futures_trading", CONSOLE_LANGUAGE))
        print(get_text("ai_forecasts", CONSOLE_LANGUAGE))
        print(get_text("languages", CONSOLE_LANGUAGE))
        
        print(f"\n{get_text('share_link', CONSOLE_LANGUAGE)}")
        print(get_text("stop_instruction", CONSOLE_LANGUAGE))
        print("=" * 40)
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n{get_text('stopping', CONSOLE_LANGUAGE)}")
    except Exception as e:
        print(f"{get_text('error_message', CONSOLE_LANGUAGE)} {e}")
    finally:
        cleanup(streamlit_process, ngrok_process)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 