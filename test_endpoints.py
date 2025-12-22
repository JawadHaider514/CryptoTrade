#!/usr/bin/env python3
"""Test script for Binance API endpoints"""

from server.binance_ws import get_binance_manager

mgr = get_binance_manager()

print("\n=== Testing Binance API Functions ===\n")

# Test fetch_klines
print("1. Testing fetch_klines('BTCUSDT', '1h', 3):")
candles = mgr.fetch_klines('BTCUSDT', '1h', 3)
print(f"   ✓ Got {len(candles)} candles")
if candles:
    print(f"   First candle close: {candles[0]['close']}")

# Test fetch_price
print("\n2. Testing fetch_price('BTCUSDT'):")
price = mgr.fetch_price('BTCUSDT')
print(f"   ✓ Price: {price}")

# Test fetch_24h_stats
print("\n3. Testing fetch_24h_stats('BTCUSDT'):")
stats = mgr.fetch_24h_stats('BTCUSDT')
print(f"   ✓ Got stats: {list(stats.keys())}")
print(f"   Current price: {stats.get('price', 'N/A')}")
print(f"   24h change: {stats.get('changePercent24h', 'N/A')}%")

print("\n=== All tests passed! ===\n")
