#!/usr/bin/env python3
"""
Verify Crypto Dashboard Setup
Tests all components before running
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(message, status='INFO'):
    if status == 'OK':
        print(f"{Colors.GREEN}✓{Colors.END} {message}")
    elif status == 'ERROR':
        print(f"{Colors.RED}✗{Colors.END} {message}")
    elif status == 'WARN':
        print(f"{Colors.YELLOW}⚠{Colors.END} {message}")
    else:
        print(f"{Colors.BLUE}ℹ{Colors.END} {message}")

def check_python():
    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True)
        print_status(f"Python: {result.stdout.strip()}", 'OK')
        return True
    except Exception as e:
        print_status(f"Python not found: {e}", 'ERROR')
        return False

def check_packages():
    required = ['flask', 'requests', 'hmac', 'hashlib']
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        for package in required:
            if package.lower() in result.stdout.lower():
                print_status(f"Package {package}: OK", 'OK')
            else:
                print_status(f"Package {package}: Missing", 'WARN')
        return True
    except:
        return False

def check_server_files():
    required_files = [
        'run.py',
        'server/web_server.py',
        'server/binance_ws.py',
        'config/settings.py',
        'templates/index.html'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print_status(f"File {file}: Found", 'OK')
        else:
            print_status(f"File {file}: Missing", 'ERROR')
            missing.append(file)
    
    return len(missing) == 0

def check_port():
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        if ':5000' in result.stdout:
            # Check if it's Python
            if 'python' in result.stdout.lower():
                print_status("Port 5000: In use by Python (server running)", 'WARN')
            else:
                print_status("Port 5000: In use by another app", 'ERROR')
                return False
        else:
            print_status("Port 5000: Available", 'OK')
        return True
    except:
        print_status("Could not check port status", 'WARN')
        return True

def check_api_keys():
    from config.settings import APP_CONFIG
    
    api_key = APP_CONFIG.get('BINANCE_API_KEY')
    secret_key = APP_CONFIG.get('BINANCE_SECRET_KEY')
    discord_webhook = APP_CONFIG.get('DISCORD_WEBHOOK')
    
    if api_key and api_key != 'your-api-key':
        print_status("Binance API Key: Configured", 'OK')
    else:
        print_status("Binance API Key: Not configured", 'WARN')
    
    if secret_key and secret_key != 'your-secret':
        print_status("Binance Secret Key: Configured", 'OK')
    else:
        print_status("Binance Secret Key: Not configured", 'WARN')
    
    if discord_webhook and 'discord.com' in discord_webhook:
        print_status("Discord Webhook: Configured", 'OK')
    else:
        print_status("Discord Webhook: Not configured", 'WARN')

def main():
    print("\n" + "="*60)
    print(" CRYPTO DASHBOARD - SETUP VERIFICATION")
    print("="*60 + "\n")
    
    checks_passed = 0
    checks_total = 5
    
    print("Checking environment...\n")
    
    # Check 1: Python
    if check_python():
        checks_passed += 1
    
    # Check 2: Packages
    check_packages()
    
    # Check 3: Server files
    if check_server_files():
        checks_passed += 1
    
    # Check 4: Port availability
    if check_port():
        checks_passed += 1
    
    # Check 5: API Keys
    print()
    check_api_keys()
    checks_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f" RESULTS: {checks_passed}/{checks_total} checks passed")
    print("="*60 + "\n")
    
    if checks_passed == checks_total:
        print_status("✅ All systems ready! You can run START_CLOUD_SIMPLE.bat", 'OK')
        return 0
    else:
        print_status("⚠️  Some issues found. See details above.", 'WARN')
        return 1

if __name__ == '__main__':
    sys.exit(main())
