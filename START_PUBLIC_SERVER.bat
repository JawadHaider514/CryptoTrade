@echo off
title Crypto Trading Dashboard - Public Access
color 0A
echo ================================================
echo    CRYPTO TRADING DASHBOARD
echo    Live Server + Public URL
echo ================================================
echo.
cd /d "C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system"

echo [0/3] Installing/Updating required packages...
python -m pip install -U pip >nul 2>&1
python -m pip install -U eventlet flask-socketio >nul 2>&1
echo.

echo [1/3] Killing Previous Processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM cloudflared.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo.

echo [2/3] Starting Web Server (Flask/SocketIO)...
echo API will be at: http://127.0.0.1:5000
start "Flask Server" /min cmd /c "python main.py"
timeout /t 5 /nobreak >nul
echo.

echo [3/3] Creating Public URL (Cloudflare Tunnel)...
echo ================================================
echo    PUBLIC URL BELOW
echo ================================================
echo.
cloudflared tunnel --url http://127.0.0.1:5000
pause
