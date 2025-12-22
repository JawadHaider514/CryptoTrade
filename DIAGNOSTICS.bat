@echo off
title Crypto Dashboard - Diagnostics
color 0E

echo ================================================
echo    🔍 CRYPTO DASHBOARD DIAGNOSTICS
echo    Troubleshooting Tool
echo ================================================
echo.

cd /d "C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system"

echo [1/6] Checking Python...
python --version
echo.

echo [2/6] Checking installed packages...
pip list | findstr /I "flask requests"
echo.

echo [3/6] Checking for running processes...
echo Killing any existing Python/Cloudflare processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM cloudflared.exe >nul 2>&1
timeout /t 2 /nobreak
echo.

echo [4/6] Checking port 5000...
netstat -ano | findstr :5000
if errorlevel 1 (
  echo ✓ Port 5000 is free
) else (
  echo ⚠ Port 5000 is in use
)
echo.

echo [5/6] Testing server startup...
echo Starting Flask server...
python run.py > server.log 2>&1 &

echo Waiting 10 seconds for server to start...
timeout /t 10 /nobreak
echo.

echo [6/6] Testing server connectivity...
python -c "import requests; r = requests.get('http://127.0.0.1:5000/api/stats/BTCUSDT', timeout=5); print('✓ Server response:', r.status_code)" 2>&1
if errorlevel 1 (
  echo ✗ Server not responding
  echo Checking logs...
  type server.log | findstr /I "error"
) else (
  echo ✓ Server is responding correctly
)
echo.

echo ================================================
echo    DIAGNOSTICS COMPLETE
echo ================================================
echo.
echo Next steps:
echo   1. If all tests passed, you can use START_PUBLIC_SERVER.bat
echo   2. If tests failed, check the logs above
echo   3. Make sure Discord webhook and API keys are configured
echo.

pause
