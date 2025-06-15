# 🚀 Deployment Guide

This guide covers deploying your Cryptocurrency Prediction App to GitHub and various server platforms.

## 📋 Prerequisites

- Git installed
- GitHub account
- Bybit API credentials
- Server with Docker (for server deployment)

## 🐙 GitHub Deployment

### Step 1: Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Crypto prediction app with spot and futures markets"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it: `cryptocurrency-prediction-app`
3. Make it public or private as needed
4. Don't initialize with README (we already have one)

### Step 3: Push to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/cryptocurrency-prediction-app.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 🖥️ Server Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Requirements:
- Ubuntu/Debian server
- Docker and Docker Compose installed

#### Steps:

1. **Clone repository on server:**
```bash
git clone https://github.com/YOUR_USERNAME/cryptocurrency-prediction-app.git
cd cryptocurrency-prediction-app
```

2. **Set up environment:**
```bash
# Copy environment template
cp .env.example .env

# Edit with your API credentials
nano .env
```

3. **Deploy with Docker:**
```bash
# Make deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

4. **Access application:**
- Open browser: `http://your-server-ip:8501`

### Option 2: VPS Deployment (DigitalOcean, Linode, etc.)

#### 1. Create VPS:
- Choose Ubuntu 20.04 or 22.04
- Minimum 2GB RAM, 1 CPU
- 25GB storage

#### 2. Install Docker:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
```

#### 3. Deploy Application:
```bash
# Clone and deploy
git clone https://github.com/YOUR_USERNAME/cryptocurrency-prediction-app.git
cd cryptocurrency-prediction-app
cp .env.example .env
nano .env  # Add your API keys
./deploy.sh
```

### Option 3: Cloud Platform Deployment

#### Heroku Deployment:

1. **Install Heroku CLI**
2. **Create Heroku app:**
```bash
heroku create your-crypto-app
```

3. **Set environment variables:**
```bash
heroku config:set BYBIT_API_KEY=your_key
heroku config:set BYBIT_API_SECRET=your_secret
```

4. **Deploy:**
```bash
git push heroku main
```

#### Railway Deployment:

1. Connect GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push

#### Streamlit Cloud:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub repository
3. Set secrets in Streamlit Cloud dashboard
4. Deploy `app_enhanced.py`

## 🔧 Configuration

### Environment Variables

Create `.env` file with:
```env
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
```

### Port Configuration

- Default port: `8501`
- To change port, modify `docker-compose.yml`:
```yaml
ports:
  - "YOUR_PORT:8501"
```

## 🔒 Security Considerations

### 1. API Key Security:
- Never commit `.env` file to Git
- Use read-only API permissions
- Rotate API keys regularly

### 2. Server Security:
```bash
# Enable firewall
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 8501  # Application

# Keep system updated
sudo apt update && sudo apt upgrade -y
```

### 3. SSL/HTTPS (Production):
```bash
# Install Nginx
sudo apt install nginx

# Configure reverse proxy
sudo nano /etc/nginx/sites-available/crypto-app
```

Example Nginx config:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring

### Docker Logs:
```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f crypto-predictor
```

### Health Checks:
```bash
# Check application health
curl http://localhost:8501/_stcore/health

# Check container status
docker-compose ps
```

## 🔄 Updates and Maintenance

### Update Application:
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Backup Models:
```bash
# Backup trained models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/
```

## 🆘 Troubleshooting

### Common Issues:

1. **Port already in use:**
```bash
sudo lsof -i :8501
sudo kill -9 PID
```

2. **Memory issues:**
```bash
# Check memory usage
free -h
# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

3. **API connection issues:**
- Verify API credentials in `.env`
- Check API rate limits
- Ensure server has internet access

### Support:
- Check GitHub Issues
- Review application logs
- Verify API credentials

## 🎯 Production Checklist

- [ ] API credentials configured
- [ ] SSL certificate installed
- [ ] Firewall configured
- [ ] Monitoring set up
- [ ] Backup strategy implemented
- [ ] Domain name configured
- [ ] Health checks working
- [ ] Error logging enabled

## 📈 Scaling

For high traffic:
- Use load balancer (Nginx)
- Multiple container instances
- Database for model storage
- Redis for caching
- CDN for static assets

---

🎉 **Your cryptocurrency prediction app is now ready for deployment!**

Choose the deployment option that best fits your needs and follow the respective steps above. 