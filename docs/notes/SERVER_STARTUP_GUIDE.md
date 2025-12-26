# 🚀 Crypto Trading Dashboard - Server Startup Guide

## Available Startup Options

### 1. **START_LOCAL_SERVER.bat** (Local Testing)
Starts the dashboard for local testing only.

```
Location: C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system\START_LOCAL_SERVER.bat

Accessible at:
  ✅ http://127.0.0.1:5000
  ✅ http://localhost:5000
  ✅ http://192.168.1.11:5000 (Network IP)
```

**Use this when:**
- Testing locally on your machine
- Developing new features
- No internet sharing needed

---

### 2. **START_PUBLIC_SERVER.bat** (Public Access with Cloudflare)
Starts the dashboard AND creates a public URL via Cloudflare Tunnel.

```
Location: C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system\START_PUBLIC_SERVER.bat

Accessible at:
  ✅ http://127.0.0.1:5000 (Local)
  ✅ https://xxx.trycloudflare.com (Public - generated when you run it)
```

**Use this when:**
- Sharing with friends/colleagues
- Accessing from outside your network
- Running on cloud server (VPS/Laptop)

---

## 📋 Requirements

### For Public Server (Cloudflare Tunnel)

1. **Download Cloudflared:**
   - Go to: https://github.com/cloudflare/cloudflared/releases
   - Download: `cloudflared-windows-amd64.exe`
   - Or use: `choco install cloudflare-warp` (if you have Chocolatey)

2. **Install Cloudflared:**
   ```powershell
   # Option 1: Download and add to PATH
   # Option 2: Or download ZIP and extract to system PATH
   ```

3. **Verify Installation:**
   ```cmd
   cloudflared --version
   ```

---

## 🎯 How to Use

### Option 1: Quick Start (Local)
```cmd
1. Double-click: START_LOCAL_SERVER.bat
2. Wait 5 seconds for server to start
3. Open browser: http://127.0.0.1:5000
4. Done! Dashboard is ready
```

### Option 2: Public Access (Cloud)
```cmd
1. Double-click: START_PUBLIC_SERVER.bat
2. Wait 15 seconds for Cloudflare tunnel setup
3. Look for: "https://xxx.trycloudflare.com"
4. Share this URL with anyone!
```

---

## 🔑 API Endpoints Available

```
PUBLIC ENDPOINTS (No Auth):
  GET  /api/price/<symbol>          - Get current price (e.g., /api/price/BTCUSDT)
  GET  /api/stats/<symbol>          - Get 24h stats (e.g., /api/stats/ETH)
  GET  /api/chart/<symbol>          - Get candlesticks (e.g., /api/chart/SOL?interval=1h)
  GET  /api/signals                 - Get trading signals
  GET  /api/statistics              - Get system statistics

AUTHENTICATED ENDPOINTS (with Testnet API Keys):
  GET  /api/account/balance         - Get account balance (testnet)
  POST /api/order/test              - Place test order (testnet - no real money)
  POST /api/discord-notify          - Send Discord notification

NOTIFICATIONS:
  POST /api/discord-notify          - Send signal to Discord
```

---

## 📱 Dashboard Features

✅ **Real-time Price Charts** (Lightweight Charts)
✅ **Live Trading Signals** (ML-powered)
✅ **24h Statistics** (High, Low, Volume, Change%)
✅ **Multiple Timeframes** (1m, 5m, 15m, 1h, 4h, 1d)
✅ **Symbol Selection** (BTC, ETH, SOL, etc.)
✅ **Discord Notifications** (Automatic signal alerts)
✅ **Binance Testnet** (Risk-free order testing)
✅ **Responsive Design** (Mobile-friendly)

---

## 🛠️ Troubleshooting

### Server Won't Start
```cmd
# Check if port 5000 is already in use
netstat -ano | findstr :5000

# Kill existing process
taskkill /F /IM python.exe

# Try again
START_LOCAL_SERVER.bat
```

### Cloudflare Tunnel Issues
```cmd
# Verify cloudflared is installed
cloudflared --version

# If not found, install via Chocolatey
choco install cloudflare-warp

# Or download from: https://github.com/cloudflare/cloudflared/releases
```

### Dashboard Loads But API Fails
1. Check server console for `[BINANCE]` or `[ERROR]` logs
2. Verify internet connection (for Binance API)
3. Check Binance API key configuration in `config/settings.py`

---

## 📊 Configuration Files

- **Server Config:** `config/settings.py`
  - Binance API Keys
  - Discord Webhook
  - Server settings

- **Dashboard:** `templates/index.html`
  - UI/UX customization
  - Chart settings

- **API Routes:** `server/web_server.py`
  - All endpoints defined here

- **Binance Integration:** `server/binance_ws.py`
  - Real-time data fetching
  - Authenticated requests

---

## 🔒 Security Notes

⚠️ **WARNING:** 
- API Keys are in `config/settings.py`
- Consider moving to `.env` file for production
- Never share webhook URLs publicly
- Testnet API keys are safe (no real money)

---

## 📞 Support

If server won't start:
1. Check Windows Defender/Firewall (allow Python)
2. Make sure you're in the correct directory
3. Check if port 5000 is free
4. Verify Python installation: `python --version`

---

**Created:** 2025-12-21
**Version:** 1.0
**Status:** ✅ Ready to Use
