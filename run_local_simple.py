#!/usr/bin/env python3
"""
Простой локальный запуск с localtunnel (не требует регистрации)
"""

import subprocess
import sys
import time
import threading
import os
from pathlib import Path

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
    
    # Запускаем Streamlit в фоне
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port=8501",
        "--server.address=localhost"
    ])
    
    return process

def start_localtunnel():
    """Запуск localtunnel"""
    print("🌐 Создаю публичный туннель...")
    print("📥 Устанавливаю localtunnel (если нужно)...")
    
    # Проверяем node.js
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except:
        print("❌ Node.js не установлен!")
        print("📥 Установите Node.js: https://nodejs.org/")
        return None
    
    # Устанавливаем localtunnel
    try:
        subprocess.run(["npm", "install", "-g", "localtunnel"], check=True)
    except:
        print("⚠️  Не удалось установить localtunnel через npm")
    
    # Запускаем туннель
    process = subprocess.Popen([
        "lt", "--port", "8501"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    return process

def main():
    """Главная функция"""
    print("🚀 Простое локальное развертывание")
    print("=" * 40)
    
    # Проверяем конфигурацию
    secrets_file = Path(".streamlit/secrets.toml")
    if not secrets_file.exists():
        print("❌ Файл .streamlit/secrets.toml не найден!")
        print("📝 Создайте файл с API ключами:")
        print("   cp secrets.toml.template .streamlit/secrets.toml")
        return 1
    
    streamlit_process = None
    tunnel_process = None
    
    try:
        # Запускаем Streamlit
        streamlit_process = start_streamlit()
        
        # Ждем немного
        print("⏳ Ожидаю запуска Streamlit...")
        time.sleep(5)
        
        # Запускаем туннель
        tunnel_process = start_localtunnel()
        
        print("\n" + "=" * 40)
        print("✅ ПРИЛОЖЕНИЕ ЗАПУЩЕНО!")
        print("=" * 40)
        print("📱 Локальный адрес: http://localhost:8501")
        print("🌐 Публичный URL будет показан в терминале")
        print("\n🛑 Нажмите Ctrl+C для остановки")
        print("=" * 40)
        
        # Показываем вывод туннеля
        if tunnel_process:
            for line in tunnel_process.stdout:
                print(f"🌐 {line.strip()}")
                if "https://" in line:
                    break
        
        # Ожидаем завершения
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Останавливаю...")
    finally:
        if streamlit_process:
            streamlit_process.terminate()
        if tunnel_process:
            tunnel_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 