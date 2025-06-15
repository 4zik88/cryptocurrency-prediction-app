#!/bin/bash

# Cryptocurrency Prediction App Deployment Script
echo "🚀 Starting deployment of Crypto Prediction App..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "📝 Please copy .env.example to .env and add your API credentials"
    echo "   cp .env.example .env"
    echo "   nano .env  # Edit with your API keys"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start the application
echo "🔨 Building and starting the application..."
docker-compose up -d --build

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
    echo "✅ Application deployed successfully!"
    echo "🌐 Access your app at: http://localhost:8501"
    echo "📊 Features available:"
    echo "   • Spot Market Predictions"
    echo "   • Futures Market Predictions"
    echo "   • Ukrainian Language Support"
    echo "   • Advanced Technical Analysis"
else
    echo "❌ Application failed to start. Check logs:"
    docker-compose logs
fi

echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop app:  docker-compose down"
echo "   Restart:   docker-compose restart" 