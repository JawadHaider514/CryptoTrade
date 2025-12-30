#!/usr/bin/env python3
"""Test /api/predictions response format"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
from datetime import datetime
from crypto_bot.server.advanced_web_server import app

# Create test client
client = app.test_client()

print("Testing /api/predictions endpoint response...\n")

# Call the endpoint
response = client.get('/api/predictions')
data = response.get_json()

print(f"Status Code: {response.status_code}")
print(f"Success: {data.get('success')}")
print(f"Predictions Count: {data.get('count')}")
print(f"Warming Up: {data.get('warming_up')}")

# Show first prediction if available
if data.get('predictions'):
    first_symbol = list(data['predictions'].keys())[0]
    pred = data['predictions'][first_symbol]
    print(f"\nFirst Prediction ({first_symbol}):")
    print(f"  Direction: {pred.get('direction')}")
    print(f"  Entry: {pred.get('entry_price')}")
    print(f"  Stop Loss: {pred.get('stop_loss')}")
    print(f"  TP1: {pred.get('tp1')}")
    print(f"  TP2: {pred.get('tp2')}")
    print(f"  TP3: {pred.get('tp3')}")
    print(f"  TP1_ETA: {pred.get('tp1_eta')}")
    if pred.get('take_profits'):
        print(f"  Take Profits Array: {pred['take_profits']}")
    print(f"  Confidence: {pred.get('confidence')}")
    print(f"  Accuracy: {pred.get('accuracy')}")
else:
    print("\nNo predictions in response (cache may be empty)")
    print("This is expected on first run")

print("\n✅ Response format test complete!")
