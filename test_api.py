#!/usr/bin/env python3
"""Test API endpoints"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

# Give server time to start
time.sleep(2)

print("\n" + "="*60)
print("TESTING API ENDPOINTS")
print("="*60)

# Test 1: /api/stats/BTCUSDT
print("\n1. GET /api/stats/BTCUSDT")
try:
    r = requests.get(f"{BASE_URL}/api/stats/BTCUSDT", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   ✓ Response: {json.dumps(data, indent=6)}")
    else:
        print(f"   ✗ Error: {r.text}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: /api/stats/BTC (symbol normalization)
print("\n2. GET /api/stats/BTC (symbol normalization test)")
try:
    r = requests.get(f"{BASE_URL}/api/stats/BTC", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   ✓ Response: success={data['success']}, symbol={data['symbol']}")
    else:
        print(f"   ✗ Error: {r.text}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: /api/price/BTCUSDT
print("\n3. GET /api/price/BTCUSDT")
try:
    r = requests.get(f"{BASE_URL}/api/price/BTCUSDT", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   ✓ Response: {json.dumps(data, indent=6)}")
    else:
        print(f"   ✗ Error: {r.text}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: /api/price/ETH (symbol normalization)
print("\n4. GET /api/price/ETH (symbol normalization test)")
try:
    r = requests.get(f"{BASE_URL}/api/price/ETH", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   ✓ Response: success={data['success']}, symbol={data['symbol']}, price={data['price']}")
    else:
        print(f"   ✗ Error: {r.text}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 5: /api/chart/BTCUSDT
print("\n5. GET /api/chart/BTCUSDT?interval=1h&limit=3")
try:
    r = requests.get(f"{BASE_URL}/api/chart/BTCUSDT?interval=1h&limit=3", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   ✓ Response: success={data['success']}, symbol={data['symbol']}, candles={len(data['candles'])}")
        if data['candles']:
            print(f"     First candle close: {data['candles'][0]['close']}")
    else:
        print(f"   ✗ Error: {r.text}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "="*60)
print("TESTS COMPLETE")
print("="*60 + "\n")
