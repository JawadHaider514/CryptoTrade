#!/usr/bin/env python
"""Regenerate all 32 coin datasets with corrected class distribution mapping."""

import subprocess
import sys
import json
from pathlib import Path

# Load coins from config
try:
    with open(Path(__file__).parent / "config" / "coins.json") as f:
        config = json.load(f)
        coins = config.get("symbols", [])
except Exception:
    # Fallback: 32 verified trading symbols
    coins = [
        'ADAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BNBUSDT', 'BTCUSDT', 'CHZUSDT', 'DASHUSDT',
        'DOGEUSDT', 'DOTUSDT', 'ETHUSDT', 'FILUSDT', 'FLOKIUSDT', 'FLOWUSDT', 'GMTUSDT',
        'ICPUSDT', 'LINKUSDT', 'LUNCUSDT', 'MANAUSDT', 'NEARUSDT',
        'OPUSDT', 'PEOPLEUSDT', 'PEPEUSDT', 'QTUMUSDT', 'SANDUSDT', 'SHIBUSDT', 'SNXUSDT',
        'SOLUSDT', 'UNIUSDT', 'VETUSDT', 'WIFUSDT', 'XLMUSDT', 'XRPUSDT'
    ]

print("=" * 80)
print(f"REGENERATING ALL {len(coins)} COIN DATASETS WITH CORRECTED CLASS DISTRIBUTION")
print("=" * 80)
print(f"Total coins: {len(coins)}")
print()

success_count = 0
failed_count = 0
failed_coins = []

for i, coin in enumerate(coins, 1):
    print(f"[{i:2d}/{len(coins)}] {coin:10s} ...", end=" ", flush=True)
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'crypto_bot.features.dataset_builder',
             '--symbol', coin, '--timeframe', '15m'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("OK")
            success_count += 1
        else:
            print("FAILED")
            failed_coins.append(coin)
            failed_count += 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        failed_coins.append(coin)
        failed_count += 1

print()
print("=" * 80)
print(f"SUCCESS: {success_count}/{len(coins)}")
if failed_coins:
    print(f"FAILED: {failed_count}/{len(coins)} - {failed_coins}")
print("=" * 80)

sys.exit(0 if failed_count == 0 else 1)
