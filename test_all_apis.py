#!/usr/bin/env python3
"""
Complete API Testing Suite
Tests: Binance API keys, Account balance, Test orders, Discord webhook
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

print("\n" + "="*70)
print(" CRYPTO DASHBOARD - COMPLETE API TEST SUITE")
print("="*70)

print("\n⏳ Waiting for server to start...")
time.sleep(3)

tests_passed = 0
tests_failed = 0

# =============================================================================
# PUBLIC API TESTS
# =============================================================================

print("\n" + "="*70)
print(" 1. PUBLIC API ENDPOINTS (Binance)")
print("="*70)

# Test 1: Get Price
print("\n1️⃣  GET /api/price/BTCUSDT")
try:
    r = requests.get(f"{BASE_URL}/api/price/BTCUSDT", timeout=5)
    if r.status_code == 200:
        data = r.json()
        if data['success']:
            print(f"   ✅ PASS - Price: {data['price']} USDT")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL - {data.get('error')}")
            tests_failed += 1
    else:
        print(f"   ❌ FAIL - Status {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    tests_failed += 1

# Test 2: Get Stats
print("\n2️⃣  GET /api/stats/BTCUSDT")
try:
    r = requests.get(f"{BASE_URL}/api/stats/BTCUSDT", timeout=5)
    if r.status_code == 200:
        data = r.json()
        if data['success']:
            stats = data['stats']
            print(f"   ✅ PASS - 24h Stats retrieved")
            print(f"      Price: ${stats['price']:,.2f}")
            print(f"      24h High: ${stats['high24h']:,.2f}")
            print(f"      24h Low: ${stats['low24h']:,.2f}")
            print(f"      24h Change: {stats['changePercent24h']}%")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL - {data.get('error')}")
            tests_failed += 1
    else:
        print(f"   ❌ FAIL - Status {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    tests_failed += 1

# Test 3: Get Chart
print("\n3️⃣  GET /api/chart/BTCUSDT?interval=1h&limit=3")
try:
    r = requests.get(f"{BASE_URL}/api/chart/BTCUSDT?interval=1h&limit=3", timeout=5)
    if r.status_code == 200:
        data = r.json()
        if data['success']:
            print(f"   ✅ PASS - {len(data['candles'])} candles retrieved")
            if data['candles']:
                candle = data['candles'][0]
                print(f"      First candle - Close: {candle['close']}")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL - {data.get('error')}")
            tests_failed += 1
    else:
        print(f"   ❌ FAIL - Status {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    tests_failed += 1

# =============================================================================
# AUTHENTICATED API TESTS
# =============================================================================

print("\n" + "="*70)
print(" 2. AUTHENTICATED API ENDPOINTS (Testnet API Keys)")
print("="*70)

# Test 4: Get Account Balance
print("\n4️⃣  GET /api/account/balance")
try:
    r = requests.get(f"{BASE_URL}/api/account/balance", timeout=10)
    if r.status_code == 200:
        data = r.json()
        if data['success']:
            balances = data['balances']
            print(f"   ✅ PASS - Account balance retrieved")
            print(f"      Assets: {len(balances)}")
            for asset, amounts in list(balances.items())[:5]:
                print(f"      {asset}: {amounts['total']} (Free: {amounts['free']}, Locked: {amounts['locked']})")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL - {data.get('error')}")
            tests_failed += 1
    else:
        print(f"   ❌ FAIL - Status {r.status_code}: {r.text}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    tests_failed += 1

# Test 5: Place Test Order
print("\n5️⃣  POST /api/order/test (Testnet - No real money)")
try:
    payload = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 0.001,
        "price": 50000
    }
    r = requests.post(f"{BASE_URL}/api/order/test", json=payload, timeout=10)
    if r.status_code == 200:
        data = r.json()
        if data['success']:
            print(f"   ✅ PASS - Test order placed successfully")
            print(f"      Symbol: {payload['symbol']}")
            print(f"      Side: {payload['side']}")
            print(f"      Quantity: {payload['quantity']}")
            print(f"      Price: {payload['price']}")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL - {data.get('error')}")
            tests_failed += 1
    else:
        print(f"   ❌ FAIL - Status {r.status_code}: {r.text}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    tests_failed += 1

# =============================================================================
# DISCORD WEBHOOK TEST
# =============================================================================

print("\n" + "="*70)
print(" 3. DISCORD NOTIFICATION")
print("="*70)

# Test 6: Discord Notification
print("\n6️⃣  POST /api/discord-notify")
try:
    payload = {
        "symbol": "BTCUSDT",
        "signal": "LONG",
        "confidence": 97,
        "price": 88050.00
    }
    r = requests.post(f"{BASE_URL}/api/discord-notify", json=payload, timeout=10)
    if r.status_code == 200:
        data = r.json()
        if data['success']:
            print(f"   ✅ PASS - Discord notification sent")
            print(f"      Signal: {payload['signal']}")
            print(f"      Symbol: {payload['symbol']}")
            print(f"      Confidence: {payload['confidence']}%")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL - {data.get('error')}")
            tests_failed += 1
    else:
        print(f"   ❌ FAIL - Status {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    tests_failed += 1

# =============================================================================
# TEST SUMMARY
# =============================================================================

print("\n" + "="*70)
print(" TEST SUMMARY")
print("="*70)
print(f"\n✅ PASSED: {tests_passed}")
print(f"❌ FAILED: {tests_failed}")
print(f"📊 TOTAL:  {tests_passed + tests_failed}")
print(f"📈 SUCCESS RATE: {(tests_passed/(tests_passed+tests_failed)*100):.1f}%")
print("\n" + "="*70 + "\n")
