#!/usr/bin/env python3
"""
Запуск полной версии криптовалютного приложения с SPOT и FUTURES рынками
Версия с ngrok туннелем
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
        print("📝 Создайте файл с вашими API ключами:")
        print("   cp secrets.toml.template .streamlit/secrets.toml")
        print("   # Затем отредактируйте файл")
        return False
    
    # Проверяем что файл app_full.py существует
    if not Path("app_full.py").exists():
        print("❌ Файл app_full.py не найден!")
        return False
    
    return True

def start_streamlit():
    """Запуск полной версии Streamlit с SPOT и FUTURES"""
    print("🚀 Запускаю ПОЛНУЮ версию с SPOT и FUTURES рынками...")
    print("📱 Файл: app_full.py")
    
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app_full.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])
    
    return process

def wait_for_streamlit():
    """Ожидание запуска Streamlit"""
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
    """Запуск ngrok туннеля"""
    print("🌐 Создаю публичный туннель через ngrok...")
    
    # Проверяем ngrok
    try:
        subprocess.run(["ngrok", "version"], capture_output=True, check=True)
    except:
        print("❌ ngrok не найден! Установите: brew install ngrok")
        return None
    
    # Запускаем ngrok
    process = subprocess.Popen([
        "ngrok", "http", "8501", "--log=stdout"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Получаем URL
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
    """Очистка процессов"""
    print("\n🛑 Останавливаю процессы...")
    
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()
        print("✅ Streamlit остановлен")
    
    if ngrok_process:
        ngrok_process.terminate() 
        ngrok_process.wait()
        print("✅ ngrok туннель закрыт")

def main():
    """Главная функция"""
    print("🚀 ПОЛНАЯ ВЕРСИЯ КРИПТОВАЛЮТНОГО ПРИЛОЖЕНИЯ")
    print("=" * 50)
    print("📊 Включает:")
    print("   • SPOT рынки (обычная торговля)")
    print("   • FUTURES рынки (фьючерсы)")
    print("   • Все технические индикаторы")
    print("   • Расширенные метрики")
    print("=" * 50)
    
    if not check_config():
        return 1
    
    streamlit_process = None
    ngrok_process = None
    
    try:
        # Запускаем Streamlit
        streamlit_process = start_streamlit()
        
        # Ждем запуска
        if not wait_for_streamlit():
            print("❌ Не удалось запустить приложение")
            return 1
        
        # Создаем туннель
        result = start_ngrok()
        if result:
            ngrok_process, public_url = result
        
        print("\n" + "=" * 50)
        print("✅ ПРИЛОЖЕНИЕ ЗАПУЩЕНО УСПЕШНО!")
        print("=" * 50)
        print(f"🌐 Публичный URL: {public_url or 'Проверьте http://localhost:4040'}")
        print(f"📱 Локальный URL: http://localhost:8501")
        print(f"🔧 ngrok панель: http://localhost:4040")
        
        print("\n🎯 ФУНКЦИОНАЛ ПРИЛОЖЕНИЯ:")
        print("   📈 SPOT рынки - обычная торговля криптовалютами")
        print("   🚀 FUTURES рынки - торговля фьючерсами")
        print("   📊 Технический анализ (MACD, RSI, Bollinger Bands)")
        print("   🤖 AI прогнозы на 1ч, 4ч, 8ч, 24ч")
        print("   🌍 Украинский и английский языки")
        
        print("\n💡 ПОДЕЛИТЕСЬ ССЫЛКОЙ с кем угодно для доступа!")
        print("🛑 Нажмите Ctrl+C для остановки")
        print("=" * 50)
        
        # Ожидаем завершения
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Получен сигнал остановки...")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        cleanup(streamlit_process, ngrok_process)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 