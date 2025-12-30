#!/usr/bin/env python3
"""Debug pagination logic"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import datetime, timezone, timedelta
from crypto_bot.data_pipeline.binance_ohlcv import fetch_klines_chunk
import logging

logging.basicConfig(level=logging.DEBUG)

print("=" * 70)
print("DEBUG: Testing _fetch_klines_chunk")
print("=" * 70)

start = datetime.now(timezone.utc) - timedelta(days=365)
end = datetime.now(timezone.utc)

start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)

print(f"\nStart: {start} ({start_ms})")
print(f"End:   {end} ({end_ms})")
print(f"Range: {(end_ms - start_ms) / (1000 * 60 * 60 * 24):.1f} days")

# First request
print(f"\nFetching chunk 1...")
chunk1 = _fetch_klines_chunk("BTCUSDT", "15m", start_time=start_ms, end_time=end_ms)

if chunk1:
    print(f"  Got {len(chunk1)} candles")
    first_time = chunk1[0][6]  # close_time
    last_time = chunk1[-1][6]
    
    first_dt = datetime.fromtimestamp(first_time / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(last_time / 1000, tz=timezone.utc)
    
    print(f"  First: {first_dt} ({first_time})")
    print(f"  Last:  {last_dt} ({last_time})")
    print(f"  Range: {(last_time - first_time) / (1000 * 60 * 15):.0f} candles (15m)")
    
    # Try second request
    new_end_ms = first_time - 1
    print(f"\nFetching chunk 2 (end_ms={new_end_ms})...")
    chunk2 = _fetch_klines_chunk("BTCUSDT", "15m", start_time=start_ms, end_time=new_end_ms)
    
    if chunk2:
        print(f"  Got {len(chunk2)} candles")
        first_time2 = chunk2[0][6]
        last_time2 = chunk2[-1][6]
        
        first_dt2 = datetime.fromtimestamp(first_time2 / 1000, tz=timezone.utc)
        last_dt2 = datetime.fromtimestamp(last_time2 / 1000, tz=timezone.utc)
        
        print(f"  First: {first_dt2} ({first_time2})")
        print(f"  Last:  {last_dt2} ({last_time2})")
    else:
        print(f"  No candles returned!")
else:
    print(f"  ERROR: First chunk returned nothing!")
