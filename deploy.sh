#!/bin/bash

# AI Crypto App Deployment Script
# This script sets up and deploys the application on a production server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="ai-crypto-app"
APP_DIR="/opt/${APP_NAME}"
SERVICE_USER="appuser"

echo -e "${GREEN}🚀 AI Crypto App Deployment Script${NC}"
echo "=================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
apt-get update && apt-get upgrade -y

# Install Docker and Docker Compose
print_status "Installing Docker and Docker Compose..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Install nginx (as backup/additional proxy)
print_status "Installing nginx..."
apt-get install -y nginx

# Create application user
print_status "Creating application user..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d $APP_DIR $SERVICE_USER
fi

# Create application directory
print_status "Setting up application directory..."
mkdir -p $APP_DIR
chown $SERVICE_USER:$SERVICE_USER $APP_DIR

# Install Git if not present
print_status "Installing Git..."
apt-get install -y git

# Clone or update repository
print_status "Setting up application code..."
if [ -d "$APP_DIR/.git" ]; then
    print_status "Updating existing repository..."
    cd $APP_DIR
    sudo -u $SERVICE_USER git pull origin main
else
    print_status "Cloning repository..."
    print_warning "Please provide your repository URL:"
    read -p "Repository URL: " REPO_URL
    sudo -u $SERVICE_USER git clone $REPO_URL $APP_DIR
    cd $APP_DIR
fi

# Create .env file
print_status "Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << EOF
CRYPTOCOMPARE_API_KEY=your_api_key_here
CONSOLE_LANGUAGE=en
EOF
    chown $SERVICE_USER:$SERVICE_USER .env
    print_warning "Please edit .env file with your actual API keys"
fi

# Create .streamlit directory and config
print_status "Setting up Streamlit configuration..."
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOF

# Create secrets.toml template
cat > .streamlit/secrets.toml << EOF
CRYPTOCOMPARE_API_KEY = "your_api_key_here"
EOF

chown -R $SERVICE_USER:$SERVICE_USER .streamlit

# Add user to docker group
print_status "Adding user to docker group..."
usermod -aG docker $SERVICE_USER

# Create systemd service for automatic startup
print_status "Creating systemd service..."
cat > /etc/systemd/system/${APP_NAME}.service << EOF
[Unit]
Description=AI Crypto App
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=$SERVICE_USER
Group=$SERVICE_USER

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl daemon-reload
systemctl enable ${APP_NAME}.service

# Set up log rotation
print_status "Setting up log rotation..."
cat > /etc/logrotate.d/${APP_NAME} << EOF
/var/lib/docker/containers/*/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    copytruncate
}
EOF

# Configure firewall
print_status "Configuring firewall..."
ufw --force enable
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp

# Build and start the application
print_status "Building and starting the application..."
sudo -u $SERVICE_USER docker-compose build
sudo -u $SERVICE_USER docker-compose up -d

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 30

# Check service status
print_status "Checking service status..."
docker-compose ps

print_status "Setting up SSL certificate (optional)..."
print_warning "To set up SSL certificate with Let's Encrypt, run:"
print_warning "sudo apt install certbot python3-certbot-nginx"
print_warning "sudo certbot --nginx -d your-domain.com"

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo "=================================="
echo "🌐 Your application should be available at: http://your-server-ip"
echo "📊 To check logs: docker-compose logs -f"
echo "🔄 To restart: sudo systemctl restart ${APP_NAME}"
echo "📈 To monitor: docker-compose ps"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit $APP_DIR/.env with your actual API keys"
echo "2. Edit $APP_DIR/.streamlit/secrets.toml with your secrets"
echo "3. Set up SSL certificate for HTTPS"
echo "4. Configure your domain DNS to point to this server"
echo "5. Set up monitoring and backups"
echo ""
echo -e "${GREEN}Happy trading! 🚀📈${NC}" 