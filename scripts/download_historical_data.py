#!/usr/bin/env python3
"""Bulk historical data downloader using ccxt (Binance).

Usage:
  python scripts/download_historical_data.py
"""
import os
import time
from datetime import datetime

try:
    import ccxt
except Exception:
    ccxt = None

import sqlite3


class HistoricalDataDownloader:
    def __init__(self, db_path='data/crypto_historical.db'):
        if ccxt is None:
            raise RuntimeError('ccxt is required. Install with `pip install ccxt`')
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timeframe TEXT DEFAULT '1h',
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, timeframe)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_time 
            ON historical_klines(symbol, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timeframe 
            ON historical_klines(timeframe)
        ''')
        conn.commit()
        conn.close()
        print("✅ Database setup complete")

    def download_symbol_data(self, symbol, start_date='2020-01-01', timeframe='1h'):
        print(f"\n📊 Downloading {symbol} from {start_date} ({timeframe})...")
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        since = int(start_dt.timestamp() * 1000)

        all_candles = []
        batch_count = 0

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not ohlcv:
                    break
                all_candles.extend(ohlcv)
                batch_count += 1
                last_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                print(f"  Batch {batch_count}: {len(all_candles)} candles | Last: {last_date.date()}")
                if ohlcv[-1][0] >= int(datetime.now().timestamp() * 1000):
                    break
                since = ohlcv[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                print(f"❌ Error downloading {symbol}: {e}")
                break

        if all_candles:
            self.save_to_database(symbol, all_candles, timeframe)
            print(f"✅ {symbol}: {len(all_candles)} candles saved")

        return len(all_candles)

    def save_to_database(self, symbol, candles, timeframe):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        data_to_insert = [
            (symbol, candle[0], candle[1], candle[2], candle[3], candle[4], candle[5], timeframe)
            for candle in candles
        ]
        cursor.executemany('''
            INSERT OR REPLACE INTO historical_klines 
            (symbol, timestamp, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        conn.commit()
        conn.close()

    def download_all_symbols(self, symbols, start_date='2020-01-01'):
        total_candles = 0
        failed_symbols = []
        print(f"\n🚀 Starting download for {len(symbols)} symbols...")
        print(f"📅 Date range: {start_date} to {datetime.now().date()}")
        print("=" * 60)
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
            try:
                count = self.download_symbol_data(symbol, start_date)
                total_candles += count
            except Exception as e:
                print(f"❌ Failed {symbol}: {e}")
                failed_symbols.append(symbol)
            print(f"\n📈 Progress: {i}/{len(symbols)} | Total candles: {total_candles:,}")

        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE!")
        print(f"Total symbols processed: {len(symbols)}")
        print(f"Total candles downloaded: {total_candles:,}")
        print(f"Failed symbols: {len(failed_symbols)}")
        if failed_symbols:
            print(f"Failed list: {failed_symbols}")

        # Run final data quality verify if available (file-based import to avoid package import issues)
        try:
            import importlib.util, os
            verify_path = os.path.join(os.path.dirname(__file__), 'verify_data_quality.py')
            if os.path.exists(verify_path):
                spec = importlib.util.spec_from_file_location('verify_data_quality', verify_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    print('\n🔎 Running final data quality verify...')
                    verify_fn = getattr(mod, 'verify_downloaded_data', None)
                    if callable(verify_fn):
                        verify_fn()
                        print('✅ Final verify complete')
                    else:
                        print('⚠️ verify_data_quality.verify_downloaded_data not found; skipping')
                else:
                    print('⚠️ Could not load spec for verify_data_quality; skipping')
            else:
                print('⚠️ verify_data_quality.py not found; skipping final verify')
        except Exception as e:
            print(f"⚠️ Final verify failed: {e}")


if __name__ == '__main__':
    # Load symbols from config
    try:
        import json as _json
        from pathlib import Path as _Path
        _config_path = _Path(__file__).parent.parent / "config" / "coins.json"
        _coins_config = _json.load(open(_config_path))
        symbols = [f"{s.replace('USDT', '')}/USDT" for s in _coins_config.get("symbols", [])]
    except Exception:
        # Fallback: basic symbols list
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'ADA/USDT',
            'DOGE/USDT', 'SOL/USDT', 'DOT/USDT'
        ]
    downloader = HistoricalDataDownloader()
    downloader.download_all_symbols(symbols, start_date='2020-01-01')
