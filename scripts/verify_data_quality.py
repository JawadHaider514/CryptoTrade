#!/usr/bin/env python3
"""
Historical Data Quality Verification Script
Analyzes crypto_historical.db for completeness, gaps, and data quality
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

# Configuration
DB_PATH = Path(__file__).parent.parent / 'data' / 'crypto_historical.db'
EXPECTED_INTERVAL_HOURS = 1  # 1-hour candles

class DataVerifier:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.issues: list[str] = []
        
    def get_cursor(self) -> sqlite3.Cursor:
        """Return a DB cursor, raising if not connected (helps static analysis)."""
        if self.conn is None:
            raise RuntimeError('Database not connected')
        return self.conn.cursor()
        
    def connect(self):
        """Connect to database"""
        if not self.db_path.exists():
            print(f"❌ Database not found: {self.db_path}")
            sys.exit(1)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        print(f"✅ Connected to: {self.db_path}")
        print(f"📊 Database size: {self.db_path.stat().st_size / (1024*1024):.2f} MB\n")
    
    def verify_schema(self):
        """Verify database schema"""
        print("=" * 80)
        print("1. SCHEMA VERIFICATION")
        print("=" * 80)
        
        cursor = self.get_cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"📋 Tables found: {', '.join(tables)}")
        
        if 'historical_klines' not in tables:
            print("❌ CRITICAL: historical_klines table not found!")
            self.issues.append("Missing historical_klines table")
            return False
        
        # Check columns
        cursor.execute("PRAGMA table_info(historical_klines)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        expected_columns = {
            'symbol': 'TEXT',
            'timestamp': 'INTEGER',
            'open': 'REAL',
            'high': 'REAL',
            'low': 'REAL',
            'close': 'REAL',
            'volume': 'REAL',
            'timeframe': 'TEXT'
        }
        
        print(f"📊 Columns found: {len(columns)}")
        for col, dtype in columns.items():
            status = "✅" if col in expected_columns else "⚠️"
            print(f"   {status} {col}: {dtype}")
        
        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        print(f"🔍 Indexes: {len(indexes)} - {', '.join(indexes) if indexes else 'None'}")
        
        if not indexes:
            self.issues.append("No indexes found - queries will be slow")
            print("   ⚠️  WARNING: No indexes - recommend adding for performance")
        
        print("✅ Schema verification complete\n")
        return True
    
    def get_basic_stats(self):
        """Get basic statistics"""
        print("=" * 80)
        print("2. BASIC STATISTICS")
        print("=" * 80)
        
        cursor = self.get_cursor()
        
        # Total candles
        cursor.execute("SELECT COUNT(*) FROM historical_klines")
        total_candles = cursor.fetchone()[0]
        print(f"📊 Total candles: {total_candles:,}")
        
        if total_candles == 0:
            print("❌ CRITICAL: No data in database!")
            self.issues.append("Database is empty")
            return
        
        # Unique symbols
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM historical_klines")
        symbol_count = cursor.fetchone()[0]
        print(f"💎 Unique symbols: {symbol_count}")
        
        # Timeframes
        cursor.execute("SELECT DISTINCT timeframe FROM historical_klines")
        timeframes = [row[0] for row in cursor.fetchall()]
        print(f"⏱️  Timeframes: {', '.join(timeframes)}")
        
        # Date range
        cursor.execute("""
            SELECT 
                MIN(datetime(timestamp/1000, 'unixepoch')) as earliest,
                MAX(datetime(timestamp/1000, 'unixepoch')) as latest
            FROM historical_klines
        """)
        row = cursor.fetchone()
        print(f"📅 Date range: {row[0]} → {row[1]}")
        
        # Calculate duration
        earliest = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        latest = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
        duration = latest - earliest
        print(f"⏳ Duration: {duration.days} days ({duration.days/365.25:.1f} years)")
        
        print(f"✅ Basic statistics complete\n")
    
    def analyze_per_symbol(self):
        """Analyze data per symbol"""
        print("=" * 80)
        print("3. PER-SYMBOL ANALYSIS")
        print("=" * 80)
        
        cursor = self.get_cursor()
        
        cursor.execute("""
            SELECT 
                symbol,
                COUNT(*) as candle_count,
                MIN(datetime(timestamp/1000, 'unixepoch')) as earliest,
                MAX(datetime(timestamp/1000, 'unixepoch')) as latest,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(volume) as avg_volume
            FROM historical_klines
            GROUP BY symbol
            ORDER BY candle_count DESC
        """)
        
        print(f"{'Symbol':<15} {'Candles':>10} {'Date Range':<42} {'Price Range':<20}")
        print("-" * 95)
        
        for row in cursor.fetchall():
            symbol = row[0]
            count = row[1]
            earliest = row[2][:10]
            latest = row[3][:10]
            min_price = row[4]
            max_price = row[5]
            
            print(f"{symbol:<15} {count:>10,} {earliest} → {latest}  ${min_price:>10.2f} - ${max_price:>10.2f}")
        
        print(f"✅ Per-symbol analysis complete\n")
    
    def detect_gaps(self):
        """Detect data gaps"""
        print("=" * 80)
        print("4. GAP DETECTION (Missing Data)")
        print("=" * 80)
        
        cursor = self.get_cursor()
        
        # Get all symbols
        cursor.execute("SELECT DISTINCT symbol FROM historical_klines ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        
        total_gaps = 0
        gap_details = []
        
        for symbol in symbols:
            # Get timestamps for this symbol
            cursor.execute("""
                SELECT timestamp
                FROM historical_klines
                WHERE symbol = ?
                ORDER BY timestamp
            """, (symbol,))
            
            timestamps = [row[0] for row in cursor.fetchall()]
            
            if len(timestamps) < 2:
                continue
            
            # Check for gaps (more than expected interval)
            expected_diff = EXPECTED_INTERVAL_HOURS * 3600 * 1000  # milliseconds
            gaps_found = 0
            large_gaps = []
            
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                
                # Allow 10% tolerance
                if diff > expected_diff * 1.5:
                    gaps_found += 1
                    total_gaps += 1
                    
                    # Track large gaps (> 24 hours)
                    if diff > 24 * 3600 * 1000:
                        gap_hours = diff / (3600 * 1000)
                        gap_start = datetime.fromtimestamp(timestamps[i-1] / 1000)
                        gap_end = datetime.fromtimestamp(timestamps[i] / 1000)
                        large_gaps.append((gap_start, gap_end, gap_hours))
            
            if gaps_found > 0:
                gap_details.append((symbol, gaps_found, large_gaps))
        
        if total_gaps == 0:
            print("✅ No significant gaps detected!")
        else:
            print(f"⚠️  Found {total_gaps} gaps across {len(gap_details)} symbols:\n")
            
            for symbol, gap_count, large_gaps in gap_details:
                print(f"   {symbol}: {gap_count} gaps")
                
                if large_gaps:
                    print(f"      Large gaps (>24h):")
                    for gap_start, gap_end, hours in large_gaps[:3]:  # Show first 3
                        print(f"         {gap_start.date()} → {gap_end.date()} ({hours:.0f} hours)")
                    
                    if len(large_gaps) > 3:
                        print(f"         ... and {len(large_gaps) - 3} more")
            
            self.issues.append(f"{total_gaps} data gaps detected")
        
        print(f"\n✅ Gap detection complete\n")
    
    def detect_anomalies(self):
        """Detect data anomalies"""
        print("=" * 80)
        print("5. ANOMALY DETECTION")
        print("=" * 80)
        
        cursor = self.get_cursor()
        
        anomalies_found = 0
        
        # Check for zero/null prices
        cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM historical_klines
            WHERE close = 0 OR close IS NULL OR open = 0 OR high = 0 OR low = 0
            GROUP BY symbol
        """)
        
        zero_prices = cursor.fetchall()
        if zero_prices:
            print("⚠️  Zero/null prices found:")
            for row in zero_prices:
                print(f"   {row[0]}: {row[1]} candles")
                anomalies_found += row[1]
            self.issues.append(f"{sum(r[1] for r in zero_prices)} candles with zero prices")
        else:
            print("✅ No zero/null prices")
        
        # Check for invalid OHLC (high < low, close > high, etc.)
        cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM historical_klines
            WHERE high < low 
               OR close > high 
               OR close < low
               OR open > high
               OR open < low
            GROUP BY symbol
        """)
        
        invalid_ohlc = cursor.fetchall()
        if invalid_ohlc:
            print("⚠️  Invalid OHLC relationships:")
            for row in invalid_ohlc:
                print(f"   {row[0]}: {row[1]} candles")
                anomalies_found += row[1]
            self.issues.append(f"{sum(r[1] for r in invalid_ohlc)} invalid OHLC candles")
        else:
            print("✅ No invalid OHLC relationships")
        
        # Check for duplicate timestamps
        cursor.execute("""
            SELECT symbol, timestamp, COUNT(*) as count
            FROM historical_klines
            GROUP BY symbol, timestamp
            HAVING count > 1
        """)
        
        duplicates = cursor.fetchall()
        if duplicates:
            print(f"⚠️  Duplicate timestamps: {len(duplicates)} instances")
            for row in duplicates[:5]:  # Show first 5
                dt = datetime.fromtimestamp(row[1] / 1000)
                print(f"   {row[0]} @ {dt}: {row[2]} duplicates")
            if len(duplicates) > 5:
                print(f"   ... and {len(duplicates) - 5} more")
            anomalies_found += len(duplicates)
            self.issues.append(f"{len(duplicates)} duplicate timestamps")
        else:
            print("✅ No duplicate timestamps")
        
        # Check for extreme price spikes (>50% in 1 hour)
        cursor.execute("""
            SELECT symbol, COUNT(*) as spike_count
            FROM (
                SELECT 
                    symbol,
                    close,
                    LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close
                FROM historical_klines
            )
            WHERE ABS((close - prev_close) / prev_close) > 0.5
            GROUP BY symbol
        """)
        
        spikes = cursor.fetchall()
        if spikes:
            print(f"⚠️  Extreme price movements (>50% in 1h):")
            for row in spikes:
                print(f"   {row[0]}: {row[1]} spikes")
            # This might be legitimate, so don't add to issues
        else:
            print("✅ No extreme price spikes")
        
        if anomalies_found == 0:
            print(f"\n✅ No critical anomalies detected")
        else:
            print(f"\n⚠️  Total anomalies: {anomalies_found}")
        
        print(f"\n✅ Anomaly detection complete\n")
    
    def calculate_completeness(self):
        """Calculate data completeness percentage"""
        print("=" * 80)
        print("6. DATA COMPLETENESS")
        print("=" * 80)
        
        cursor = self.get_cursor()
        
        cursor.execute("SELECT DISTINCT symbol FROM historical_klines")
        symbols = [row[0] for row in cursor.fetchall()]
        
        for symbol in symbols:
            # Get date range
            cursor.execute("""
                SELECT 
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts,
                    COUNT(*) as actual_count
                FROM historical_klines
                WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            min_ts = row[0]
            max_ts = row[1]
            actual_count = row[2]
            
            # Calculate expected count (1h intervals)
            expected_count = (max_ts - min_ts) / (3600 * 1000) + 1
            completeness = (actual_count / expected_count) * 100 if expected_count > 0 else 0
            
            status = "✅" if completeness > 95 else "⚠️" if completeness > 85 else "❌"
            print(f"{status} {symbol:<15} {completeness:>6.2f}% complete ({actual_count:,}/{int(expected_count):,} candles)")
            
            if completeness < 90:
                self.issues.append(f"{symbol} only {completeness:.1f}% complete")
        
        print(f"\n✅ Completeness calculation complete\n")
    
    def generate_summary(self):
        """Generate final summary"""
        print("=" * 80)
        print("7. VERIFICATION SUMMARY")
        print("=" * 80)
        
        cursor = self.get_cursor()
        cursor.execute("SELECT COUNT(*) FROM historical_klines")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM historical_klines")
        symbols = cursor.fetchone()[0]
        
        print(f"📊 Total Records: {total:,}")
        print(f"💎 Symbols: {symbols}")
        print(f"💾 Database Size: {self.db_path.stat().st_size / (1024*1024):.2f} MB")
        
        if not self.issues:
            print(f"\n✅ DATA QUALITY: EXCELLENT")
            print(f"   No critical issues found. Data is ready for backtesting!")
        else:
            print(f"\n⚠️  ISSUES FOUND: {len(self.issues)}")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
            print(f"\n💡 Recommendation: Review and fix issues before production use")
        
        print("\n" + "=" * 80)
        print(f"Verification completed at: {datetime.now()}")
        print("=" * 80)
    
    def run_full_verification(self):
        """Run complete verification suite"""
        try:
            self.connect()
            
            if not self.verify_schema():
                print("❌ Schema verification failed. Aborting.")
                return False
            
            self.get_basic_stats()
            self.analyze_per_symbol()
            self.detect_gaps()
            self.detect_anomalies()
            self.calculate_completeness()
            self.generate_summary()
            
            return len(self.issues) == 0
            
        except Exception as e:
            print(f"\n❌ VERIFICATION ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            if self.conn:
                self.conn.close()


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("HISTORICAL DATA QUALITY VERIFICATION")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Started: {datetime.now()}")
    print("=" * 80 + "\n")
    
    verifier = DataVerifier(DB_PATH)
    success = verifier.run_full_verification()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()