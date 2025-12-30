#!/usr/bin/env python3
"""Test pagination for historical OHLCV data"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crypto_bot.data_pipeline.binance_ohlcv import fetch_klines
from datetime import datetime, timezone, timedelta

print("=" * 70)
print("TESTING PAGINATION - BTCUSDT 15m (365 days)")
print("=" * 70)

start = datetime.now(timezone.utc) - timedelta(days=365)
end = datetime.now(timezone.utc)

print(f"\nFetching from {start.date()} to {end.date()}...")

df = fetch_klines("BTCUSDT", "15m", start=start, end=end)

print(f"\n{'='*70}")
print("RESULTS")
print("=" * 70)
print(f"Total rows: {len(df)}")
print(f"Min timestamp: {df['timestamp'].min()}")
print(f"Max timestamp: {df['timestamp'].max()}")
print(f"Date range: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"\nExpected: ~35k+ rows for 15m over 365 days")
print(f"Actual: {len(df)} rows")

if len(df) > 30000:
    print("\n✅ SUCCESS: Got sufficient historical data!")
    print(f"{'='*70}")
else:
    print("\n❌ WARNING: Data seems limited")
