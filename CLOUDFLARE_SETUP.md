# Cloudflare Tunnel Setup Guide

## Problem: 502 Bad Gateway

If you see **"502 Bad Gateway"** error, it means Cloudflare cannot reach your local server.

### ✅ Solutions

#### 1. **Verify Server is Running**
```powershell
# Check if Flask server is on port 5000
netstat -ano | findstr :5000
```

Expected output should show python.exe listening on port 5000.

#### 2. **Test Local Server First**
```powershell
# Start the local server
cd "C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system"
python run.py
```

Then in another terminal:
```powershell
# Test if server responds
curl http://127.0.0.1:5000/api/price/BTCUSDT
```

#### 3. **Check Cloudflare Configuration**
Make sure your cloudflared command uses **PORT 5000** (NOT 8080):
```bat
cloudflared tunnel --url http://127.0.0.1:5000
```

#### 4. **Windows Firewall**
Allow Python through Windows Firewall:
1. Open Windows Defender Firewall
2. Click "Allow an app through firewall"
3. Find Python and check both Private and Public boxes

#### 5. **Restart Everything**
```bat
@REM Kill all Python processes
taskkill /F /IM python.exe

@REM Kill all cloudflared processes  
taskkill /F /IM cloudflared.exe

@REM Wait
timeout /t 3

@REM Start fresh - use START_PUBLIC_SERVER.bat
START_PUBLIC_SERVER.bat
```

#### 6. **Check Server Logs**
If using START_PUBLIC_SERVER.bat, check `server.log`:
```powershell
Get-Content server.log -Tail 50
```

## Expected Output

When working correctly:
```
[2/3] Starting Web Server...
Dashboard will be at: http://127.0.0.1:5000

[3/3] Creating Public URL (Cloudflare Tunnel)...

================================================
    📡 PUBLIC URL BELOW
    Is URL ko apne dosto ko share kar sakte ho!
================================================

Your quick Tunnel has been created! Visit it at (it may take a few moments to be available):

https://something-random-1234.trycloudflare.com
```

## Troubleshooting Steps

### Step 1: Verify Server is Online
```powershell
curl -Uri "http://127.0.0.1:5000/" -UseBasicParsing
```

### Step 2: Verify Specific Endpoint
```powershell
curl -Uri "http://127.0.0.1:5000/api/stats/BTCUSDT" -UseBasicParsing
```

### Step 3: Check for Port Conflicts
```powershell
# See all listening ports
netstat -ano | findstr LISTENING

# Kill process on port 5000 (if not Flask)
netstat -ano | findstr :5000 | findstr /V python
```

### Step 4: Reinstall Cloudflared
```powershell
# Download latest cloudflared
scoop install cloudflared

# Or using winget
winget install Cloudflare.cloudflared
```

### Step 5: Run with Full Debugging
```bat
@echo off
cd /d "C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system"

echo Starting server with logs...
python run.py > full_debug.log 2>&1

echo Waiting for server...
timeout /t 5

echo Starting cloudflared with verbose logging...
cloudflared tunnel --url http://127.0.0.1:5000 -v
```

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| 502 Bad Gateway | Server not running | Start server first |
| Connection timeout | Firewall blocking | Allow Python in Windows Firewall |
| "Connection refused" | Server crashed | Check server.log for errors |
| Port already in use | Another app on port 5000 | Change port or kill other app |

## API Endpoints to Test

```
GET  /api/price/BTCUSDT
GET  /api/stats/BTCUSDT  
GET  /api/chart/BTCUSDT?interval=1h&limit=100
GET  /api/account/balance
POST /api/order/test
POST /api/discord-notify
```

## Quick Restart Commands

```powershell
# Kill everything
taskkill /F /IM python.exe
taskkill /F /IM cloudflared.exe

# Start public server
cd "C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system"
.\START_PUBLIC_SERVER.bat
```

---

**Still having issues?** Check the server logs in `server.log` or `full_debug.log`
