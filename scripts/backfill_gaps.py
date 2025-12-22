#!/usr/bin/env python3
"""
Backfill missing historical candles (targeted gaps) using ccxt (Binance).

Usage (dry-run):
  python scripts/backfill_gaps.py --dry-run

Options:
  --symbols SYMBOLS   comma-separated list of symbols (e.g., BTC/USDT,XRP/USDT)
  --timeframe TF      timeframe (default: 1h)
  --max-days N        only attempt to backfill gaps <= N days (default: 7)
  --csv PATH          use gaps CSV (reports/gaps_*.csv) instead of scanning DB
  --limit N           limit number of gaps to show/attempt (default: 50)
  --dry-run           print actions only, do not write to DB
  --force             override safety checks

Notes:
- This script performs targeted re-downloads for missing windows only.
- By default it will NOT perform writes unless `--dry-run` is omitted and `--force` used when gap > max-days.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import time
import math
import sys
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    import ccxt
except Exception:
    ccxt = None

REPORTS_DIR = Path(__file__).parent.parent / 'reports'
DB_PATH = Path(__file__).parent.parent / 'data' / 'crypto_historical.db'

# Mapping for timeframe string to milliseconds
_TIMEFRAME_MS = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000
}


def timeframe_to_ms(tf: str) -> int:
    return _TIMEFRAME_MS.get(tf, 60 * 60 * 1000)


def detect_gaps_from_db(db_path: Path, timeframe: str, gap_multiplier: float = 1.5, limit: Optional[int] = None) -> List[Dict]:
    """Scan DB for gaps; return list of gap dicts: {symbol, start_ms, end_ms, missing_count} """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT symbol FROM historical_klines WHERE timeframe = ?", (timeframe,))
    symbols = [r[0] for r in cur.fetchall()]

    expected_diff = timeframe_to_ms(timeframe)
    gaps: List[Dict] = []

    for sym in symbols:
        cur.execute("SELECT timestamp FROM historical_klines WHERE symbol = ? AND timeframe = ? ORDER BY timestamp ASC", (sym, timeframe))
        rows = [r[0] for r in cur.fetchall()]
        if not rows:
            continue
        prev = rows[0]
        for ts in rows[1:]:
            diff = ts - prev
            if diff > expected_diff * gap_multiplier:
                gaps.append({
                    'symbol': sym,
                    'start_ms': prev + expected_diff,
                    'end_ms': ts - expected_diff,
                    'missing_count': int(round(diff / expected_diff)) - 1
                })
            prev = ts

    conn.close()

    gaps_sorted = sorted(gaps, key=lambda x: (x['symbol'], x['start_ms']))
    if limit:
        return gaps_sorted[:limit]
    return gaps_sorted


def read_gaps_csv(path: Path, limit: Optional[int] = None) -> List[Dict]:
    gaps = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Expect columns: symbol, gap_start, gap_end, missing_count
            try:
                start_dt = r.get('gap_start') or r.get('gap_start_utc')
                end_dt = r.get('gap_end') or r.get('gap_end_utc')
                # Parse as UTC if no tz information is provided to avoid local-time offsets
                try:
                    dt_start = datetime.fromisoformat(start_dt)
                except Exception:
                    dt_start = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
                if dt_start.tzinfo is None:
                    from datetime import timezone
                    dt_start = dt_start.replace(tzinfo=timezone.utc)
                start_ms = int(dt_start.timestamp() * 1000)

                try:
                    dt_end = datetime.fromisoformat(end_dt)
                except Exception:
                    dt_end = datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S")
                if dt_end.tzinfo is None:
                    from datetime import timezone
                    dt_end = dt_end.replace(tzinfo=timezone.utc)
                end_ms = int(dt_end.timestamp() * 1000)

                gaps.append({
                    'symbol': r['symbol'],
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'missing_count': int(r.get('missing_count', 0))
                })
            except Exception:
                continue
    gaps_sorted = sorted(gaps, key=lambda x: (x['symbol'], x['start_ms']))
    if limit:
        return gaps_sorted[:limit]
    return gaps_sorted


class GapBackfiller:
    def __init__(self, db_path: Path = DB_PATH, timeframe: str = '1h', dry_run: bool = True, max_days: int = 7, force: bool = False):
        self.db_path = db_path
        self.timeframe = timeframe
        self.dry_run = dry_run
        self.max_days = max_days
        self.force = force
        self.exchange = None

        if ccxt is None:
            raise RuntimeError('ccxt is required. Install with `pip install ccxt`')

        self.exchange = ccxt.binance({'enableRateLimit': True})

    def _save_candles(self, symbol: str, candles: List[List], timeframe: str):
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        data = []
        for c in candles:
            ts, o, h, l, close, vol = c[0], c[1], c[2], c[3], c[4], c[5]
            data.append((symbol, ts, o, h, l, close, vol, timeframe))
        cur.executemany('''
            INSERT OR REPLACE INTO historical_klines
            (symbol, timestamp, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', data)
        conn.commit()
        conn.close()

    def _fetch_range(self, symbol: str, start_ms: int, end_ms: int) -> List[List]:
        """Fetch OHLCV in [start_ms, end_ms] (inclusive)."""
        limit = 1000
        interval_ms = timeframe_to_ms(self.timeframe)
        since = start_ms
        fetched = []
        attempts = 0
        max_attempts = 6

        while since <= end_ms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since, limit)
                if not ohlcv:
                    break
                # Keep only candles within requested window
                for c in ohlcv:
                    if c[0] < start_ms:
                        continue
                    if c[0] > end_ms:
                        break
                    fetched.append(c)

                last_ts = ohlcv[-1][0]
                if last_ts >= end_ms - interval_ms:
                    break

                since = last_ts + interval_ms
                attempts = 0
                # Respect rate limit
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                attempts += 1
                if attempts > max_attempts:
                    raise
                backoff = min(30, 2 ** attempts)
                print(f"⚠️  Fetch error (attempt {attempts}), sleeping {backoff}s: {e}")
                time.sleep(backoff)
        # After bulk fetch, check for any remaining single-timestamp holes and try fetching them individually
        interval_ms = timeframe_to_ms(self.timeframe)
        expected_ts = list(range(start_ms, end_ms + interval_ms, interval_ms))
        fetched_ts = {c[0] for c in fetched}
        missing_ts = [ts for ts in expected_ts if ts >= start_ms and ts <= end_ms and ts not in fetched_ts]

        if missing_ts:
            print(f"⚠️  Detected {len(missing_ts)} missing timestamps; attempting single-candle fetches...")
            for m_ts in missing_ts:
                try:
                    single = self.exchange.fetch_ohlcv(symbol, self.timeframe, m_ts, 1)
                    if single:
                        # Only add if timestamp within window and not already present
                        if single[0][0] >= start_ms and single[0][0] <= end_ms and single[0][0] not in fetched_ts:
                            fetched.append(single[0])
                            fetched_ts.add(single[0][0])
                    time.sleep(self.exchange.rateLimit / 1000)
                except Exception as e:
                    # Ignore single fetch failures but log briefly
                    print(f"   ⚠️ single fetch failed for {symbol} @ {datetime.utcfromtimestamp(m_ts/1000)}: {e}")

        # Sort fetched candles by timestamp before returning
        fetched_sorted = sorted(fetched, key=lambda x: x[0])
        return fetched_sorted

    def backfill_gap(self, gap: Dict) -> Tuple[int, int]:
        symbol = gap['symbol']
        start_ms = gap['start_ms']
        end_ms = gap['end_ms']
        duration_ms = end_ms - start_ms
        duration_days = duration_ms / (24 * 3600 * 1000)

        if duration_days > self.max_days and not self.force:
            print(f"⚠️  Skipping gap for {symbol} ({duration_days:.2f} days) > max_days={self.max_days}. Use --force to override")
            return 0, 0

        if self.dry_run:
            print(f"DRY-RUN: Will attempt to fetch {symbol} {self.timeframe} from {datetime.utcfromtimestamp(start_ms/1000)} to {datetime.utcfromtimestamp(end_ms/1000)} ({duration_days:.2f} days)")
            return 0, 0

        print(f"⏳ Backfilling {symbol}: {datetime.utcfromtimestamp(start_ms/1000)} -> {datetime.utcfromtimestamp(end_ms/1000)}")
        candles = self._fetch_range(symbol, start_ms, end_ms)
        if not candles:
            print(f"❌ No candles fetched for {symbol} in range")
            return 0, 0

        self._save_candles(symbol, candles, self.timeframe)
        print(f"✅ Saved {len(candles)} candles for {symbol}")
        return len(candles), 1

    def run(self, gaps: List[Dict], limit: Optional[int] = None) -> Dict:
        attempted = 0
        filled = 0
        saved_candles = 0
        for gap in gaps[:limit] if limit else gaps:
            attempted += 1
            try:
                cnt, groups = self.backfill_gap(gap)
                saved_candles += cnt
                filled += groups
            except Exception as e:
                print(f"❌ Error backfilling gap {gap}: {e}")
        return {'attempted': attempted, 'filled_gaps': filled, 'saved_candles': saved_candles}


def parse_args():
    p = argparse.ArgumentParser(description='Backfill missing historical candles')
    p.add_argument('--symbols', type=str, help='Comma-separated symbols to restrict (e.g., BTC/USDT,XRP/USDT)')
    p.add_argument('--timeframe', type=str, default='1h', help='Timeframe to consider (default 1h)')
    p.add_argument('--max-days', type=int, default=7, help='Max gap length to backfill (days)')
    p.add_argument('--csv', type=str, help='Path to gaps CSV (reports/gaps_*.csv)')
    p.add_argument('--limit', type=int, default=50, help='Limit number of gaps to process/list')
    p.add_argument('--dry-run', action='store_true', help='Do not write to DB; only print actions')
    p.add_argument('--force', action='store_true', help='Override safety checks (use with caution)')
    return p.parse_args()


def main():
    args = parse_args()
    timeframe = args.timeframe
    dry_run = args.dry_run
    max_days = args.max_days
    force = args.force

    print(f"🔍 Backfill starting (timeframe={timeframe}, dry_run={dry_run}, max_days={max_days}, limit={args.limit})")

    if args.csv:
        path = Path(args.csv)
        if not path.exists():
            print(f"❌ CSV not found: {path}")
            sys.exit(1)
        gaps = read_gaps_csv(path, limit=args.limit)
    else:
        gaps = detect_gaps_from_db(DB_PATH, timeframe, limit=args.limit)

    if args.symbols:
        allowed = {s.strip().upper() for s in args.symbols.split(',')}
        gaps = [g for g in gaps if g['symbol'].upper() in allowed]

    if not gaps:
        print("✅ No gaps found matching criteria")
        return

    print(f"Found {len(gaps)} gap(s) to consider (showing up to {args.limit}).")
    for g in gaps:
        s = g['symbol']; st = datetime.utcfromtimestamp(g['start_ms']/1000); en = datetime.utcfromtimestamp(g['end_ms']/1000)
        dur = (g['end_ms'] - g['start_ms']) / 1000 / 3600
        print(f" - {s}: {st} -> {en} ({dur:.2f} hours, missing ~{g['missing_count']})")

    # Confirm action if not dry-run
    if not dry_run:
        confirm = input('Proceed with backfill for the above gaps? [y/N]: ')
        if confirm.lower() != 'y':
            print('Aborted by user')
            return

    try:
        backfiller = GapBackfiller(DB_PATH, timeframe=timeframe, dry_run=dry_run, max_days=max_days, force=force)
    except Exception as e:
        print(f"❌ Cannot initialize backfiller: {e}")
        sys.exit(1)

    result = backfiller.run(gaps, limit=args.limit)
    print(f"\nSummary: attempted={result['attempted']}, filled_gaps={result['filled_gaps']}, saved_candles={result['saved_candles']}")


if __name__ == '__main__':
    main()
