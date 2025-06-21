#!/bin/bash

# Скрипт для быстрого запуска криптовалютного приложения локально с доступом извне

echo "🚀 Локальный запуск криптовалютного приложения"
echo "============================================="

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден!"
    exit 1
fi

# Проверяем конфигурацию
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "❌ Файл .streamlit/secrets.toml не найден!"
    echo "📝 Создайте файл с API ключами:"
    echo "   cp secrets.toml.template .streamlit/secrets.toml"
    echo "   # Затем отредактируйте и добавьте ваши Bybit API ключи"
    exit 1
fi

# Выбираем метод туннелирования
echo ""
echo "Выберите способ создания публичного доступа:"
echo "1) ngrok (требует регистрацию, но более стабильный)"
echo "2) localtunnel (не требует регистрацию, но URL может измениться)"
echo "3) Только локально (без публичного доступа)"
read -p "Ваш выбор (1-3): " choice

case $choice in
    1)
        echo "🌐 Запуск с ngrok..."
        python3 run_local_with_tunnel.py
        ;;
    2)
        echo "🌐 Запуск с localtunnel..."
        python3 run_local_simple.py
        ;;
    3)
        echo "📱 Запуск только локально..."
        # Определяем какой файл запускать
        if [ -f "app_enhanced_v2.py" ]; then
            APP_FILE="app_enhanced_v2.py"
        elif [ -f "app_enhanced.py" ]; then
            APP_FILE="app_enhanced.py"
        else
            APP_FILE="app.py"
        fi
        
        echo "🚀 Запускаю $APP_FILE..."
        python3 -m streamlit run $APP_FILE --server.port=8501
        ;;
    *)
        echo "❌ Неверный выбор!"
        exit 1
        ;;
esac 