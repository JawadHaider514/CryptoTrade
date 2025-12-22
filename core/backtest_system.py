#!/usr/bin/env python3
"""
PHASE 1: BACKTESTING SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real backtesting with actual historical data and real outcomes.

TASK 1.1: Historical Data Collector
- Download Binance historical klines
- Store in SQLite database
- 30 days of 1-minute candles
"""

import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import json
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    """Download and store historical price data from Binance"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        """Initialize data collector"""
        self.db_path = db_path
        self.base_url = "https://api.binance.com/api/v3"
        self.init_database()
    
    def init_database(self):
        """Create database and tables"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    trades INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol)
                )
            """)
            
            # Create index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp_symbol 
                ON historical_candles(timestamp, symbol)
            """)
            
            conn.commit()
            logger.info("✅ Database initialized")
    
    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 1000, start_time: Optional[int] = None) -> list:
        """
        Download klines from Binance
        
        Args:
            symbol: Trading pair (e.g., XRPUSDT)
            interval: Candle interval (1m, 5m, 1h, etc.)
            limit: Number of candles to fetch (max 1000)
            start_time: Start time in milliseconds (optional)
        
        Returns:
            List of kline data
        """
        try:
            url = f"{self.base_url}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            if start_time:
                params["startTime"] = start_time
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            klines = response.json()
            logger.info(f"✅ Downloaded {len(klines)} candles for {symbol}")
            return klines
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download klines: {e}")
            return []
    
    def download_30_days_of_data(self, symbol: str = "XRPUSDT"):
        """
        Download 30 days of 1-minute candles
        
        Due to API limit of 1000 candles per request, we need multiple requests
        1440 minutes per day × 30 days = 43,200 candles
        """
        logger.info(f"\n📥 Downloading 30 days of {symbol} data...")
        logger.info(f"   This will take ~2-3 minutes (API rate limits)")
        
        # Get current time
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)
        
        total_candles = 0
        request_count = 0
        
        # Download in chunks of 1000 candles
        # Start from the beginning (earliest) and move forward
        current_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)
        
        while current_time_ms < end_time_ms:
            # Fetch 1000 candles starting from current_time
            klines = self.get_klines(symbol, interval="1m", limit=1000, start_time=current_time_ms)
            
            if not klines:
                logger.error("❌ No data received, stopping")
                break
            
            # Store in database
            inserted = self.store_klines(symbol, klines)
            total_candles += inserted
            request_count += 1
            
            logger.info(f"   ✓ Request #{request_count}: Stored {inserted} candles (Total: {total_candles})")
            
            # Update current_time to last candle's time + 1 minute
            last_kline = klines[-1]
            current_time_ms = last_kline[0] + 60000  # Add 1 minute to avoid duplicates
            
            # Rate limiting (Binance allows 1200 requests per minute)
            time.sleep(0.2)
            
            # Stop if we've reached the end or have enough data (30k+ candles = 20+ days)
            if len(klines) < 1000 or current_time_ms >= end_time_ms or total_candles >= 30000:
                break
        
        logger.info(f"\n✅ Download complete!")
        logger.info(f"   Total candles stored: {total_candles}")
        
        # Calculate actual date range
        if total_candles > 0:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM historical_candles WHERE symbol = ?", (symbol,))
                min_ts, max_ts = c.fetchone()
                min_date = datetime.fromtimestamp(min_ts / 1000)
                max_date = datetime.fromtimestamp(max_ts / 1000)
                logger.info(f"   Date range: {min_date.date()} to {max_date.date()}")
        
        return total_candles
    
    def store_klines(self, symbol: str, klines: list) -> int:
        """
        Store klines in database
        
        Kline format from Binance:
        [0] = Open time (ms)
        [1] = Open
        [2] = High
        [3] = Low
        [4] = Close
        [5] = Volume
        [8] = Number of trades
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = []
                for kline in klines:
                    data.append((
                        kline[0],           # timestamp (ms)
                        symbol,
                        float(kline[1]),    # open
                        float(kline[2]),    # high
                        float(kline[3]),    # low
                        float(kline[4]),    # close
                        float(kline[5]),    # volume
                        int(kline[8])       # trades
                    ))
                
                # Insert with IGNORE to skip duplicates
                conn.executemany("""
                    INSERT OR IGNORE INTO historical_candles
                    (timestamp, symbol, open, high, low, close, volume, trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, data)
                
                conn.commit()
                return len(data)
                
        except Exception as e:
            logger.error(f"❌ Error storing klines: {e}")
            return 0
    
    def get_candles_for_period(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get candles for a specific time period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                start_ms = int(start_time.timestamp() * 1000)
                end_ms = int(end_time.timestamp() * 1000)
                
                df = pd.read_sql_query("""
                    SELECT timestamp, symbol, open, high, low, close, volume, trades
                    FROM historical_candles
                    WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                """, conn, params=(symbol, start_ms, end_ms))
                
                # Convert timestamp to seconds for easier handling
                df['timestamp'] = df['timestamp'] / 1000
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                return df
                
        except Exception as e:
            logger.error(f"❌ Error retrieving candles: {e}")
            return pd.DataFrame()
    
    def get_future_candles(self, symbol: str, from_timestamp: datetime, minutes: int = 5) -> pd.DataFrame:
        """Get candles after a specific timestamp (for outcome tracking)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                from_ms = int(from_timestamp.timestamp() * 1000)
                to_time = from_timestamp + timedelta(minutes=minutes)
                to_ms = int(to_time.timestamp() * 1000)
                
                df = pd.read_sql_query("""
                    SELECT timestamp, symbol, open, high, low, close, volume
                    FROM historical_candles
                    WHERE symbol = ? AND timestamp > ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, conn, params=(symbol, from_ms, to_ms, minutes))
                
                df['timestamp'] = df['timestamp'] / 1000
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                return df
                
        except Exception as e:
            logger.error(f"❌ Error retrieving future candles: {e}")
            return pd.DataFrame()
    
    def get_data_stats(self, symbol: str) -> dict:
        """Get statistics about stored data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_candles,
                        MIN(timestamp) as oldest_timestamp,
                        MAX(timestamp) as newest_timestamp,
                        MIN(close) as min_price,
                        MAX(close) as max_price,
                        AVG(close) as avg_price,
                        AVG(volume) as avg_volume
                    FROM historical_candles
                    WHERE symbol = ?
                """, (symbol,))
                
                row = cursor.fetchone()
                
                if row[0] == 0:
                    return {"message": "No data found"}
                
                stats = {
                    'total_candles': row[0],
                    'oldest_time': datetime.fromtimestamp(row[1] / 1000),
                    'newest_time': datetime.fromtimestamp(row[2] / 1000),
                    'days_of_data': (row[2] - row[1]) / (1000 * 60 * 60 * 24),
                    'min_price': row[3],
                    'max_price': row[4],
                    'avg_price': row[5],
                    'avg_volume': row[6]
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    collector = HistoricalDataCollector()
    
    # Download 30 days of data
    print("\n" + "="*60)
    print("HISTORICAL DATA COLLECTOR")
    print("="*60)
    
    collector.download_30_days_of_data("XRPUSDT")
    
    # Show statistics
    stats = collector.get_data_stats("XRPUSDT")
    
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test retrieving data
    print("\n" + "="*60)
    print("SAMPLE DATA (first 5 candles)")
    print("="*60)
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    
    df = collector.get_candles_for_period("XRPUSDT", start_time, end_time)
    print(df.head())
