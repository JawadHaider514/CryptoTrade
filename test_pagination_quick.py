#!/usr/bin/env python3
"""Quick test of pagination - fetch 7 days instead of 365"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crypto_bot.data_pipeline.binance_ohlcv import fetch_klines
from datetime import datetime, timezone, timedelta

print("=" * 70)
print("QUICK TEST: BTCUSDT 15m (7 days)")
print("=" * 70)

start = datetime.now(timezone.utc) - timedelta(days=7)
end = datetime.now(timezone.utc)

print(f"\nFetching from {start.date()} to {end.date()}...")
print(f"Expected: ~672 rows (7 days * 96 candles/day for 15m)\n")

df = fetch_klines("BTCUSDT", "15m", start=start, end=end)

print(f"\n{'='*70}")
print(f"Total rows: {len(df)}")
print(f"Min timestamp: {df['timestamp'].min()}")
print(f"Max timestamp: {df['timestamp'].max()}")

if len(df) > 600:
    print("\n✅ SUCCESS: Pagination working correctly!")
else:
    print("\n❌ WARNING: Expected more rows")
