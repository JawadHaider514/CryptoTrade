#!/usr/bin/env python3
"""
TASK 1.3: Outcome Tracker
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Track what ACTUALLY happened after each signal.
Check next 5 minutes of price data.
Determine if TP1/TP2/TP3 or SL was hit.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutcomeTracker:
    """Track actual outcomes of trading signals"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        """Initialize outcome tracker"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create signal_outcomes table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    exit_time INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    tp_hit TEXT,
                    pnl_percentage REAL,
                    pnl_dollars REAL,
                    time_in_trade_seconds INTEGER,
                    max_price REAL,
                    min_price REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(signal_id) REFERENCES backtest_signals(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome_signal_id 
                ON signal_outcomes(signal_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome_result 
                ON signal_outcomes(result, symbol)
            """)
            
            conn.commit()
    
    def get_future_candles(self, symbol: str, from_timestamp: int, minutes: int = 5) -> pd.DataFrame:
        """
        Get candles after a specific timestamp.
        Maximum of 'minutes' minutes after the signal.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                to_ms = from_timestamp + (minutes * 60 * 1000)
                
                df = pd.read_sql_query("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM historical_candles
                    WHERE symbol = ? AND timestamp > ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                """, conn, params=(symbol, from_timestamp, to_ms))
                
                if df.empty:
                    return pd.DataFrame()
                
                df['datetime'] = pd.to_datetime(df['timestamp'] / 1000, unit='s')
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting future candles: {e}")
            return pd.DataFrame()
    
    def calculate_outcome(self, signal: Dict, future_prices: pd.DataFrame) -> Dict:
        """
        Check what ACTUALLY happened with a signal.
        Returns outcome data.
        """
        if future_prices.empty:
            return {
                'result': 'TIMEOUT',
                'exit_price': signal['entry_price'],
                'exit_time': signal['timestamp'] + (5 * 60 * 1000),
                'tp_hit': None,
                'pnl_percentage': 0,
                'pnl_dollars': 0,
                'time_in_trade': 0,
                'max_price': signal['entry_price'],
                'min_price': signal['entry_price']
            }
        
        direction = signal['direction']
        entry = signal['entry_price']
        sl = signal['stop_loss']
        tp1 = signal['take_profit_1']
        tp2 = signal['take_profit_2']
        tp3 = signal['take_profit_3']
        
        max_price = entry
        min_price = entry
        
        # Check each candle
        for idx, row in future_prices.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            timestamp = row['timestamp']
            
            # Track extremes
            max_price = max(max_price, high)
            min_price = min(min_price, low)
            
            # LONG trades
            if direction == "LONG":
                # Check if TP3 hit
                if high >= tp3:
                    return {
                        'result': 'WIN',
                        'exit_price': tp3,
                        'exit_time': timestamp,
                        'tp_hit': 'TP3',
                        'pnl_percentage': ((tp3 - entry) / entry) * 100,
                        'pnl_dollars': tp3 - entry,
                        'time_in_trade': int((timestamp - signal['timestamp']) / 1000),
                        'max_price': max_price,
                        'min_price': min_price
                    }
                
                # Check if TP2 hit
                if high >= tp2:
                    return {
                        'result': 'WIN',
                        'exit_price': tp2,
                        'exit_time': timestamp,
                        'tp_hit': 'TP2',
                        'pnl_percentage': ((tp2 - entry) / entry) * 100,
                        'pnl_dollars': tp2 - entry,
                        'time_in_trade': int((timestamp - signal['timestamp']) / 1000),
                        'max_price': max_price,
                        'min_price': min_price
                    }
                
                # Check if TP1 hit
                if high >= tp1:
                    return {
                        'result': 'WIN',
                        'exit_price': tp1,
                        'exit_time': timestamp,
                        'tp_hit': 'TP1',
                        'pnl_percentage': ((tp1 - entry) / entry) * 100,
                        'pnl_dollars': tp1 - entry,
                        'time_in_trade': int((timestamp - signal['timestamp']) / 1000),
                        'max_price': max_price,
                        'min_price': min_price
                    }
                
                # Check if SL hit
                if low <= sl:
                    return {
                        'result': 'LOSS',
                        'exit_price': sl,
                        'exit_time': timestamp,
                        'tp_hit': None,
                        'pnl_percentage': ((sl - entry) / entry) * 100,
                        'pnl_dollars': sl - entry,
                        'time_in_trade': int((timestamp - signal['timestamp']) / 1000),
                        'max_price': max_price,
                        'min_price': min_price
                    }
            
            # SHORT trades
            elif direction == "SHORT":
                # Check if TP3 hit
                if low <= tp3:
                    return {
                        'result': 'WIN',
                        'exit_price': tp3,
                        'exit_time': timestamp,
                        'tp_hit': 'TP3',
                        'pnl_percentage': ((entry - tp3) / entry) * 100,
                        'pnl_dollars': entry - tp3,
                        'time_in_trade': int((timestamp - signal['timestamp']) / 1000),
                        'max_price': max_price,
                        'min_price': min_price
                    }
                
                # Check if TP2 hit
                if low <= tp2:
                    return {
                        'result': 'WIN',
                        'exit_price': tp2,
                        'exit_time': timestamp,
                        'tp_hit': 'TP2',
                        'pnl_percentage': ((entry - tp2) / entry) * 100,
                        'pnl_dollars': entry - tp2,
                        'time_in_trade': int((timestamp - signal['timestamp']) / 1000),
                        'max_price': max_price,
                        'min_price': min_price
                    }
                
                # Check if TP1 hit
                if low <= tp1:
                    return {
                        'result': 'WIN',
                        'exit_price': tp1,
                        'exit_time': timestamp,
                        'tp_hit': 'TP1',
                        'pnl_percentage': ((entry - tp1) / entry) * 100,
                        'pnl_dollars': entry - tp1,
                        'time_in_trade': int((timestamp - signal['timestamp']) / 1000),
                        'max_price': max_price,
                        'min_price': min_price
                    }
                
                # Check if SL hit
                if high >= sl:
                    return {
                        'result': 'LOSS',
                        'exit_price': sl,
                        'exit_time': timestamp,
                        'tp_hit': None,
                        'pnl_percentage': ((entry - sl) / entry) * 100,
                        'pnl_dollars': entry - sl,
                        'time_in_trade': int((timestamp - signal['timestamp']) / 1000),
                        'max_price': max_price,
                        'min_price': min_price
                    }
        
        # If we get here, no TP or SL was hit (timeout)
        final_price = future_prices['close'].iloc[-1]
        
        if direction == "LONG":
            pnl_pct = ((final_price - entry) / entry) * 100
            pnl_dollars = final_price - entry
        else:  # SHORT
            pnl_pct = ((entry - final_price) / entry) * 100
            pnl_dollars = entry - final_price
        
        return {
            'result': 'TIMEOUT',
            'exit_price': final_price,
            'exit_time': future_prices['timestamp'].iloc[-1],
            'tp_hit': None,
            'pnl_percentage': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'time_in_trade': int((future_prices['timestamp'].iloc[-1] - signal['timestamp']) / 1000),
            'max_price': max_price,
            'min_price': min_price
        }
    
    def track_signal(self, signal: Dict) -> bool:
        """
        Track a signal and save its outcome to database.
        Returns True if successful.
        """
        try:
            # Get signal ID
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id FROM backtest_signals
                    WHERE timestamp = ? AND symbol = ? AND direction = ?
                """, (signal['timestamp'], signal['symbol'], signal['direction']))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Signal not found in database: {signal}")
                    return False
                
                signal_id = row[0]
            
            # Get future price data
            future_prices = self.get_future_candles(
                signal['symbol'],
                signal['timestamp'],
                minutes=5
            )
            
            # Calculate outcome
            outcome = self.calculate_outcome(signal, future_prices)
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO signal_outcomes
                    (signal_id, timestamp, symbol, direction, entry_price,
                     exit_price, exit_time, result, tp_hit, pnl_percentage,
                     pnl_dollars, time_in_trade_seconds, max_price, min_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_id,
                    signal['timestamp'],
                    signal['symbol'],
                    signal['direction'],
                    signal['entry_price'],
                    outcome['exit_price'],
                    outcome['exit_time'],
                    outcome['result'],
                    outcome['tp_hit'],
                    outcome['pnl_percentage'],
                    outcome['pnl_dollars'],
                    outcome['time_in_trade'],
                    outcome['max_price'],
                    outcome['min_price']
                ))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking signal: {e}")
            return False
    
    def track_all_signals(self, symbol: str) -> int:
        """
        Track all signals for a symbol that don't have outcomes yet.
        Returns count of tracked signals.
        """
        logger.info(f"📊 Tracking outcomes for {symbol}...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get signals without outcomes
                df = pd.read_sql_query("""
                    SELECT bs.* FROM backtest_signals bs
                    LEFT JOIN signal_outcomes so ON bs.id = so.signal_id
                    WHERE bs.symbol = ? AND so.id IS NULL
                    ORDER BY bs.timestamp ASC
                """, conn, params=(symbol,))
            
            if df.empty:
                logger.info("✅ All signals already tracked")
                return 0
            
            tracked_count = 0
            
            for idx, row in df.iterrows():
                signal = row.to_dict()
                signal['patterns'] = json.loads(signal['patterns'])
                
                if self.track_signal(signal):
                    tracked_count += 1
                    
                    if tracked_count % 10 == 0:
                        logger.info(f"   ✓ Tracked {tracked_count} signals")
            
            logger.info(f"✅ Tracked {tracked_count} signals")
            return tracked_count
            
        except Exception as e:
            logger.error(f"Error tracking all signals: {e}")
            return 0
    
    def get_outcomes(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get tracked outcomes from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM signal_outcomes
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, conn, params=(symbol, limit))
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting outcomes: {e}")
            return pd.DataFrame()
    
    def get_outcome_stats(self, symbol: str) -> Dict:
        """Get statistics about signal outcomes"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN result = 'TIMEOUT' THEN 1 ELSE 0 END) as timeouts,
                        AVG(pnl_percentage) as avg_pnl_pct,
                        SUM(pnl_dollars) as total_pnl,
                        AVG(time_in_trade_seconds) as avg_time_seconds
                    FROM signal_outcomes
                    WHERE symbol = ?
                """, (symbol,))
                
                row = cursor.fetchone()
                
                if not row or row[0] == 0:
                    return {'error': 'No outcomes found'}
                
                total = row[0]
                wins = row[1] or 0
                losses = row[2] or 0
                timeouts = row[3] or 0
                
                stats = {
                    'total_signals': total,
                    'wins': wins,
                    'losses': losses,
                    'timeouts': timeouts,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'loss_rate': (losses / total * 100) if total > 0 else 0,
                    'timeout_rate': (timeouts / total * 100) if total > 0 else 0,
                    'avg_pnl_percentage': row[4] or 0,
                    'total_pnl_dollars': row[5] or 0,
                    'avg_time_in_trade': int(row[6] or 0)
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    tracker = OutcomeTracker()
    
    print("\n" + "="*60)
    print("OUTCOME TRACKER")
    print("="*60)
    
    # Track all signals for XRPUSDT
    tracked = tracker.track_all_signals("XRPUSDT")
    
    # Get statistics
    stats = tracker.get_outcome_stats("XRPUSDT")
    
    print("\n" + "="*60)
    print("SIGNAL OUTCOME STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Show sample outcomes
    print("\n" + "="*60)
    print("SAMPLE OUTCOMES (most recent)")
    print("="*60)
    
    outcomes = tracker.get_outcomes("XRPUSDT", limit=5)
    print(outcomes[['timestamp', 'direction', 'entry_price', 'result', 'pnl_percentage']])
