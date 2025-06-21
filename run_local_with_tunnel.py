#!/usr/bin/env python3
"""
Локальный запуск криптовалютного приложения с доступом извне через ngrok туннель
"""

import subprocess
import sys
import time
import threading
import os
import signal
import requests
from pathlib import Path

def install_package(package):
    """Установка пакета если он не установлен"""
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Устанавливаю {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_streamlit_config():
    """Проверка конфигурации Streamlit"""
    secrets_file = Path(".streamlit/secrets.toml")
    if not secrets_file.exists():
        print("❌ Файл .streamlit/secrets.toml не найден!")
        print("📝 Пожалуйста, создайте файл с вашими API ключами:")
        print("   cp secrets.toml.template .streamlit/secrets.toml")
        print("   # Затем отредактируйте файл и добавьте ваши API ключи")
        return False
    return True

def start_streamlit():
    """Запуск Streamlit приложения"""
    print("🚀 Запускаю Streamlit приложение...")
    
    # Определяем какое приложение запускать
    if Path("app_enhanced_v2.py").exists():
        app_file = "app_enhanced_v2.py"
    elif Path("app_enhanced.py").exists():
        app_file = "app_enhanced.py"
    else:
        app_file = "app.py"
    
    print(f"📱 Запускаю {app_file}...")
    
    # Запускаем Streamlit
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])
    
    return process

def wait_for_streamlit():
    """Ожидание запуска Streamlit"""
    print("⏳ Ожидаю запуска Streamlit...")
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=2)
            if response.status_code == 200:
                print("✅ Streamlit запущен успешно!")
                return True
        except:
            time.sleep(1)
            if i % 5 == 0:
                print(f"⏳ Попытка {i+1}/{max_attempts}...")
    
    print("❌ Не удалось дождаться запуска Streamlit")
    return False

def start_ngrok():
    """Запуск ngrok туннеля"""
    print("🌐 Создаю ngrok туннель...")
    
    # Проверяем установлен ли ngrok
    try:
        subprocess.run(["ngrok", "version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ngrok не установлен!")
        print("📥 Установите ngrok:")
        print("   1. Скачайте с https://ngrok.com/download")
        print("   2. Или установите через Homebrew: brew install ngrok")
        print("   3. Зарегистрируйтесь и добавьте authtoken: ngrok authtoken YOUR_TOKEN")
        return None
    
    # Запускаем ngrok
    process = subprocess.Popen([
        "ngrok", "http", "8501", "--log=stdout"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Ждем получения URL
    time.sleep(3)
    
    try:
        # Получаем URL через ngrok API
        response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        if response.status_code == 200:
            tunnels = response.json()
            if tunnels["tunnels"]:
                public_url = tunnels["tunnels"][0]["public_url"]
                print(f"🌐 Ваше приложение доступно по адресу: {public_url}")
                print(f"📱 Локальный адрес: http://localhost:8501")
                return process, public_url
    except:
        pass
    
    print("⚠️  Не удалось получить публичный URL автоматически")
    print("🌐 Проверьте ngrok веб-интерфейс: http://localhost:4040")
    return process, None

def cleanup_processes(streamlit_process, ngrok_process):
    """Очистка процессов при завершении"""
    print("\n🛑 Завершаю процессы...")
    
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
    print("🚀 Локальное развертывание криптовалютного приложения")
    print("=" * 50)
    
    # Проверяем конфигурацию
    if not check_streamlit_config():
        return 1
    
    # Устанавливаем зависимости
    install_package("streamlit")
    install_package("requests")
    
    streamlit_process = None
    ngrok_process = None
    
    try:
        # Запускаем Streamlit
        streamlit_process = start_streamlit()
        
        # Ждем запуска
        if not wait_for_streamlit():
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
        print("\n💡 Поделитесь публичным URL с другими для доступа к приложению")
        print("⚠️  ВНИМАНИЕ: При использовании бесплатного ngrok URL может измениться при перезапуске")
        print("\n🛑 Нажмите Ctrl+C для остановки")
        print("=" * 50)
        
        # Ожидаем завершения
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Получен сигнал остановки...")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        cleanup_processes(streamlit_process, ngrok_process)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 