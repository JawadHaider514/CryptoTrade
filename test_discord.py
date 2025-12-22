#!/usr/bin/env python3
"""Test Discord webhook"""

import requests
import json

webhook_url = "https://discord.com/api/webhooks/1447651247749337089/tajiT4cIfvOrAUxVxHyR2lQT3S6wxMb_iPJ2PCkshPoeH7g6UoxW-FPVIEQMfC70BblV"

print("\n" + "="*60)
print("Testing Discord Webhook Integration")
print("="*60 + "\n")

# Test the webhook directly
embed = {
    "title": "🚀 LONG Signal - BTCUSDT",
    "description": "**Price**: $88,050.00\n**Confidence**: 97%",
    "color": 3066993,  # Green
    "fields": [
        {
            "name": "Symbol",
            "value": "BTCUSDT",
            "inline": True
        },
        {
            "name": "Signal",
            "value": "LONG",
            "inline": True
        },
        {
            "name": "Confidence",
            "value": "97%",
            "inline": True
        },
        {
            "name": "Time",
            "value": "2025-12-21 19:50:00",
            "inline": False
        }
    ],
    "footer": {
        "text": "Crypto Trading System"
    }
}

payload = {
    "embeds": [embed]
}

try:
    print("Sending test message to Discord...")
    response = requests.post(webhook_url, json=payload, timeout=10)
    
    if response.status_code in [200, 204]:
        print(f"✅ SUCCESS! Message sent to Discord")
        print(f"   Status Code: {response.status_code}")
    else:
        print(f"❌ FAILED!")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60 + "\n")
