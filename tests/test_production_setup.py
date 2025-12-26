"""
Quick test: Testnet adapter + health monitor + Discord alerts
"""

import os
import sys
from pathlib import Path

# Load environment from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, use direct env vars
    pass

# Set env vars directly if not loaded from .env
os.environ.setdefault('API_KEY_TESTNET', 'xH14gQvS1VbGpluvmPYJzuzMCtEtBjmI0vAUqsXoe5oJVQrYYziVPArjWeDTTUFk')
os.environ.setdefault('API_SECRET_TESTNET', 'ura98tCA6uXQMAWvg6lDiIg6DnnNrIs7ZbUnU67YAEhjAUyavxuIXO4547emRkBN')
os.environ.setdefault('DISCORD_WEBHOOK_URL', 'https://discord.com/api/webhooks/1447651247749337089/tajiT4cIfvOrAUxVxHyR2lQT3S6wxMb_iPJ2PCkshPoeH7g6UoxW-FPVIEQMfC70BblV')

from crypto_bot.core.exchange_adapter import get_adapter
from scripts.health_monitor import HealthMonitor

print("=" * 70)
print("FULL INTEGRATION TEST: Testnet + Health Monitor + Discord Alerts")
print("=" * 70)

# Test 1: Testnet Adapter
print("\n[1] Testing Binance Testnet Adapter...")
adapter = get_adapter('testnet')
if adapter:
    print("    [OK] Adapter created")
    connected = adapter.connect()
    if connected:
        print("    [SUCCESS] Connected to Binance testnet!")
        
        # Fetch balance
        balance = adapter.get_balance()
        if balance:
            print(f"    [OK] Account fetched: {len(balance)} currencies")
        else:
            print("    [WARN] Could not fetch balance")
    else:
        print("    [ERROR] Connection failed")
else:
    print("    [ERROR] Adapter creation failed")

# Test 2: Health Monitor
print("\n[2] Testing Health Monitor...")
monitor = HealthMonitor()
checks = monitor.run_all_checks()
report = monitor.report()
print(report)

# Test 3: Discord Alert
print("\n[3] Testing Discord Webhook Alert...")
webhook = os.getenv('DISCORD_WEBHOOK_URL')
if webhook:
    test_message = "[TEST] Production trading system is online and operational!"
    alert_sent = monitor.send_alert(test_message)
    if alert_sent:
        print("    [SUCCESS] Discord alert sent!")
    else:
        print("    [WARN] Failed to send Discord alert")
else:
    print("    [WARN] Discord webhook not configured")

print("\n" + "=" * 70)
print("[OK] Full integration test complete!")
print("=" * 70)
