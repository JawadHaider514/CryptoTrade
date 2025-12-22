#!/usr/bin/env python3
"""Generate a CSV listing data gaps per symbol from crypto_historical.db

CSV columns:
- symbol
- gap_start_utc (ISO)
- gap_end_utc (ISO)
- gap_hours (float)
- gap_candles_missing (approx)

Usage: python scripts/generate_gap_csv.py
"""
import os
import csv
import sqlite3
from datetime import datetime
from pathlib import Path

DB = Path(__file__).parent.parent / 'data' / 'crypto_historical.db'
REPORT_DIR = Path(__file__).parent.parent / 'reports'
EXPECTED_INTERVAL_HOURS = 1
TOLERANCE = 1.5  # 150% of expected


def find_gaps():
    if not DB.exists():
        print('Database not found:', DB)
        return []

    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()

    # get symbols
    cur.execute("SELECT DISTINCT symbol FROM historical_klines ORDER BY symbol")
    symbols = [r[0] for r in cur.fetchall()]

    gaps = []
    expected_diff = EXPECTED_INTERVAL_HOURS * 3600 * 1000
    threshold = expected_diff * TOLERANCE

    for symbol in symbols:
        cur.execute("SELECT timestamp FROM historical_klines WHERE symbol=? ORDER BY timestamp", (symbol,))
        rows = cur.fetchall()
        timestamps = [r[0] for r in rows]
        if len(timestamps) < 2:
            continue
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i - 1]
            if diff > threshold:
                gap_hours = diff / (3600 * 1000)
                # approximate missing candles count (rounded)
                missing = int(round(diff / expected_diff)) - 1
                gap_start = datetime.utcfromtimestamp(timestamps[i-1] / 1000)
                gap_end = datetime.utcfromtimestamp(timestamps[i] / 1000)
                gaps.append({
                    'symbol': symbol,
                    'gap_start_utc': gap_start.isoformat(sep=' '),
                    'gap_end_utc': gap_end.isoformat(sep=' '),
                    'gap_hours': f"{gap_hours:.2f}",
                    'missing_candles': missing,
                })
    conn.close()
    return gaps


def write_csv(gaps):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    path = REPORT_DIR / f'gaps_{ts}.csv'
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['symbol','gap_start_utc','gap_end_utc','gap_hours','missing_candles'])
        writer.writeheader()
        for r in gaps:
            writer.writerow(r)
    return path


if __name__ == '__main__':
    print('Scanning DB for gaps...')
    gaps = find_gaps()
    if not gaps:
        print('No gaps found.')
    else:
        path = write_csv(gaps)
        print(f'Gaps found: {len(gaps)} -> {path}')
        # print summary top 5
        from collections import Counter
        c = Counter([g['symbol'] for g in gaps])
        print('Top symbols by gap count:')
        for sym, cnt in c.most_common()[:10]:
            print(f'  {sym}: {cnt} gaps')
