@echo off
title Crypto Trading Dashboard - Cloud Setup
color 0A

echo ================================================
echo    🚀 CRYPTO DASHBOARD - CLOUD SETUP
echo    Automatic Configuration
echo ================================================
echo.

setlocal enabledelayedexpansion

cd /d "C:\Users\Jawad\AI BOT\crypto-dashboard-project\crypto_trading_system"

REM Step 1: Kill existing processes
echo [Step 1/4] Cleaning up existing processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM cloudflared.exe >nul 2>&1
timeout /t 2 /nobreak
echo ✓ Cleanup complete
echo.

REM Step 2: Start Flask server
echo [Step 2/4] Starting Flask server on port 5000...
echo Dashboard will be served from: http://127.0.0.1:5000
echo.

REM Start server and capture PID
python run.py > server.log 2>&1 &
set SERVER_PID=%ERRORLEVEL%

echo Waiting 15 seconds for server to initialize...
timeout /t 15 /nobreak
echo.

REM Step 3: Verify server is running
echo [Step 3/4] Verifying server connectivity...
python -c "import requests, time; attempts=0; 
while attempts < 5:
    try:
        r = requests.get('http://127.0.0.1:3000/health', timeout=3)
        print('✓ Server is online and responding')
        break
    except:
        attempts += 1
        if attempts < 5:
            print(f'  Attempting connection {attempts}/5...')
            time.sleep(2)
        else:
            print('✗ Server did not respond after 5 attempts')
            print('  Check server.log for errors')
" 2>&1
echo.

REM Step 4: Start Cloudflare tunnel
echo [Step 4/4] Starting Cloudflare tunnel...
echo.
echo ================================================
echo    📡 YOUR PUBLIC URL
echo    Share this with anyone!
echo ================================================
echo.
echo Note: The tunnel may take 10-30 seconds to activate
echo If you see errors, check CLOUDFLARE_SETUP.md
echo.

cloudflared tunnel --url http://127.0.0.1:3000

REM Cleanup on exit
echo.
echo Shutting down...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM cloudflared.exe >nul 2>&1

echo Done. Goodbye!
pause
