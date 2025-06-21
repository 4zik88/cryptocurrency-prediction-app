#!/usr/bin/env python3
"""
Запуск app_enhanced.py с SPOT и FUTURES рынками
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_config():
    """Проверка конфигурации"""
    secrets_file = Path(".streamlit/secrets.toml")
    if not secrets_file.exists():
        print("❌ Файл .streamlit/secrets.toml не найден!")
        return False
    
    if not Path("app_enhanced.py").exists():
        print("❌ Файл app_enhanced.py не найден!")
        return False
    
    return True

def start_streamlit():
    """Запуск app_enhanced.py"""
    print("🚀 Запускаю app_enhanced.py с SPOT и FUTURES рынками...")
    
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app_enhanced.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])
    
    return process

def wait_for_streamlit():
    """Ожидание запуска"""
    print("⏳ Ожидаю запуска приложения...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=2)
            if response.status_code == 200:
                print("✅ Приложение запущено успешно!")
                return True
        except:
            time.sleep(1)
            if i % 5 == 0:
                print(f"⏳ Попытка {i+1}/30...")
    
    return False

def start_ngrok():
    """Запуск ngrok"""
    print("🌐 Создаю публичный туннель...")
    
    try:
        subprocess.run(["ngrok", "version"], capture_output=True, check=True)
    except:
        print("❌ ngrok не найден!")
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
    """Очистка"""
    print("\n🛑 Останавливаю процессы...")
    
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()
        print("✅ Streamlit остановлен")
    
    if ngrok_process:
        ngrok_process.terminate() 
        ngrok_process.wait()
        print("✅ ngrok остановлен")

def main():
    """Главная функция"""
    print("🚀 ЗАПУСК app_enhanced.py")
    print("=" * 40)
    print("📊 Функционал:")
    print("   • SPOT рынки")
    print("   • FUTURES рынки") 
    print("   • AI прогнозы")
    print("   • Технический анализ")
    print("=" * 40)
    
    if not check_config():
        return 1
    
    streamlit_process = None
    ngrok_process = None
    
    try:
        streamlit_process = start_streamlit()
        
        if not wait_for_streamlit():
            print("❌ Не удалось запустить приложение")
            return 1
        
        result = start_ngrok()
        if result:
            ngrok_process, public_url = result
        
        print("\n" + "=" * 40)
        print("✅ ПРИЛОЖЕНИЕ ЗАПУЩЕНО!")
        print("=" * 40)
        print(f"🌐 Публичный URL: {public_url or 'Проверьте http://localhost:4040'}")
        print(f"📱 Локальный URL: http://localhost:8501")
        
        print("\n🎯 В приложении доступно:")
        print("   📈 SPOT - обычная торговля")
        print("   🚀 FUTURES - фьючерсы")
        print("   🤖 AI прогнозы (1ч, 4ч, 8ч, 24ч)")
        print("   🌍 Украинский/Английский")
        
        print("\n💡 Поделитесь ссылкой для доступа!")
        print("🛑 Ctrl+C для остановки")
        print("=" * 40)
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Остановка...")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        cleanup(streamlit_process, ngrok_process)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 