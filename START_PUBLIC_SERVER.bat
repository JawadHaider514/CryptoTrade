@echo off
title Crypto Trading Dashboard - Public Access
color 0A

echo ================================================
echo    CRYPTO TRADING DASHBOARD
echo    Live Server + Public URL
echo ================================================
echo.

cd /d "C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system"

echo [1/3] Killing Previous Processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo.
echo [2/3] Starting Web Server (Flask)...
echo API will be at: http://127.0.0.1:5000

REM Start python server in a new minimized window (non-blocking)
start "Flask Server" /min cmd /c "python run.py > server.log 2>&1"

REM Give it a few seconds to boot
timeout /t 3 /nobreak >nul

echo.
echo [3/3] Creating Public URL (Cloudflare Tunnel)...
echo ================================================
echo    PUBLIC URL BELOW
echo ================================================
echo.

cloudflared tunnel --url http://127.0.0.1:5000

pause
