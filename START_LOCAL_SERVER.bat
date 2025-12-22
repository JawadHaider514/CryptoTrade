@echo off
title Crypto Trading Dashboard - Local
color 0B

echo ================================================
echo    🚀 CRYPTO TRADING DASHBOARD
echo    Local Testing Mode
echo ================================================
echo.

cd /d "C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system"

echo [1/2] Killing Previous Processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak

echo.
echo [2/2] Starting Web Server...
echo.
echo ================================================
echo    📊 DASHBOARD ADDRESSES
echo    Local:        http://127.0.0.1:3000
echo    Network:      http://192.168.1.11:3000
echo    Localhost:    http://localhost:3000
echo ================================================
echo.
echo 🔗 API Endpoints:
echo    /api/price/BTCUSDT
echo    /api/stats/BTCUSDT
echo    /api/chart/BTCUSDT?interval=1h
echo    /api/account/balance
echo    /api/order/test
echo.

python run.py

pause
