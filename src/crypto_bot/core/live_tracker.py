#!/usr/bin/env python3
"""
PHASE 2: LIVE TRACKING SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 2.1: Real-Time Signal Tracker
Track live signals against real market prices.
Detect TP/SL hits and update P&L in real-time.
"""

import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
import threading
import time
from collections import deque
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveSignalTracker:
    """Track live trading signals against real market prices"""
    
    def __init__(self, db_path: str = "data/backtest.db", check_interval: int = 1):
        """
        Initialize live tracker
        
        Args:
            db_path: Path to database
            check_interval: Seconds between price checks
        """
        # Convert to absolute path if relative
        if not os.path.isabs(db_path):
            # Get project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            db_path = str(project_root / db_path)
        
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = db_path
        self.check_interval = check_interval
        self.base_url = "https://api.binance.com/api/v3"
        
        self.active_signals = {}
        self.completed_signals = deque(maxlen=1000)
        self.price_cache = {}
        
        self.init_database()
        self.running = False
        self.lock = threading.Lock()
    
    def init_database(self):
        """Create live tracking tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL NOT NULL,
                    take_profit_2 REAL NOT NULL,
                    take_profit_3 REAL NOT NULL,
                    confluence_score INTEGER,
                    start_time DATETIME NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    current_price REAL,
                    current_pnl REAL DEFAULT 0,
                    current_pnl_percentage REAL DEFAULT 0,
                    tp1_hit BOOLEAN DEFAULT 0,
                    tp2_hit BOOLEAN DEFAULT 0,
                    tp3_hit BOOLEAN DEFAULT 0,
                    sl_hit BOOLEAN DEFAULT 0,
                    exit_price REAL,
                    exit_time DATETIME,
                    final_result TEXT,
                    final_pnl REAL,
                    final_pnl_percentage REAL,
                    time_in_trade_seconds INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_live_symbol_status 
                ON live_signals(symbol, status)
            """)
            
            conn.commit()
    
    def get_real_time_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current price from Binance
        
        Returns: {symbol, price, timestamp, bid, ask}
        """
        try:
            # Check cache first (avoid hitting API too much)
            if symbol in self.price_cache:
                cached = self.price_cache[symbol]
                if time.time() - cached['timestamp'] < self.check_interval:
                    return cached
            
            # Get ticker
            url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            price_data = {
                'symbol': symbol,
                'price': float(data['lastPrice']),
                'bid': float(data['bidPrice']),
                'ask': float(data['askPrice']),
                'timestamp': time.time()
            }
            
            # Cache it
            self.price_cache[symbol] = price_data
            
            return price_data
            
        except Exception as e:
            logger.warning(f"Could not get price for {symbol}: {e}")
            return None
    
    def add_signal(self, signal: Dict) -> bool:
        """
        Start tracking a new live signal
        
        Args:
            signal: Signal dict with entry, TP levels, SL, etc.
        
        Returns: True if successfully added
        """
        try:
            with self.lock:
                signal_id = f"{signal['symbol']}_{signal['timestamp']}_{signal['direction']}"
                
                if signal_id in self.active_signals:
                    logger.warning(f"Signal {signal_id} already tracking")
                    return False
                
                # Get initial price
                price_data = self.get_real_time_price(signal['symbol'])
                if not price_data:
                    logger.error(f"Could not get price for {signal['symbol']}")
                    return False
                
                tracking_data = {
                    'signal_id': signal_id,
                    'symbol': signal['symbol'],
                    'direction': signal['direction'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'tp1': signal['take_profit_1'],
                    'tp2': signal['take_profit_2'],
                    'tp3': signal['take_profit_3'],
                    'confluence_score': signal.get('confluence_score', 0),
                    'start_time': datetime.now(),
                    'current_price': price_data['price'],
                    'current_pnl': 0,
                    'current_pnl_percentage': 0,
                    'tp1_hit': False,
                    'tp2_hit': False,
                    'tp3_hit': False,
                    'sl_hit': False,
                    'status': 'MONITORING'
                }
                
                self.active_signals[signal_id] = tracking_data
                self.save_to_database(tracking_data)
                
                logger.info(f"✅ Started tracking {signal_id}")
                logger.info(f"   Direction: {signal['direction']}")
                logger.info(f"   Entry: {signal['entry_price']} | SL: {signal['stop_loss']}")
                logger.info(f"   TP1: {signal['take_profit_1']} | TP2: {signal['take_profit_2']} | TP3: {signal['take_profit_3']}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding signal: {e}")
            return False
    
    def check_signal(self, signal_id: str, track: Dict) -> None:
        """
        Check if a signal's TP or SL was hit
        
        Args:
            signal_id: Signal ID
            track: Tracking data dict
        """
        try:
            # Get current price
            price_data = self.get_real_time_price(track['symbol'])
            if not price_data:
                return
            
            current_price = price_data['price']
            track['current_price'] = current_price
            
            # Calculate current PnL
            if track['direction'] == 'LONG':
                track['current_pnl'] = current_price - track['entry_price']
                track['current_pnl_percentage'] = (track['current_pnl'] / track['entry_price']) * 100
            else:  # SHORT
                track['current_pnl'] = track['entry_price'] - current_price
                track['current_pnl_percentage'] = (track['current_pnl'] / track['entry_price']) * 100
            
            # Check TPs and SL
            if track['direction'] == 'LONG':
                # TP3 has priority
                if current_price >= track['tp3'] and not track['tp3_hit']:
                    self.complete_signal(signal_id, track, 'WIN', 'TP3', track['tp3'])
                    return
                
                # TP2
                if current_price >= track['tp2'] and not track['tp2_hit']:
                    track['tp2_hit'] = True
                    logger.info(f"🎯 TP2 HIT: {signal_id} @ {current_price}")
                
                # TP1
                if current_price >= track['tp1'] and not track['tp1_hit']:
                    track['tp1_hit'] = True
                    logger.info(f"🎯 TP1 HIT: {signal_id} @ {current_price}")
                
                # Stop loss
                if current_price <= track['stop_loss']:
                    self.complete_signal(signal_id, track, 'LOSS', None, track['stop_loss'])
                    return
            
            else:  # SHORT
                # TP3 has priority
                if current_price <= track['tp3'] and not track['tp3_hit']:
                    self.complete_signal(signal_id, track, 'WIN', 'TP3', track['tp3'])
                    return
                
                # TP2
                if current_price <= track['tp2'] and not track['tp2_hit']:
                    track['tp2_hit'] = True
                    logger.info(f"🎯 TP2 HIT: {signal_id} @ {current_price}")
                
                # TP1
                if current_price <= track['tp1'] and not track['tp1_hit']:
                    track['tp1_hit'] = True
                    logger.info(f"🎯 TP1 HIT: {signal_id} @ {current_price}")
                
                # Stop loss
                if current_price >= track['stop_loss']:
                    self.complete_signal(signal_id, track, 'LOSS', None, track['stop_loss'])
                    return
            
            # Check timeout (5 minutes)
            time_elapsed = (datetime.now() - track['start_time']).total_seconds()
            if time_elapsed > 300:  # 5 minutes
                self.complete_signal(signal_id, track, 'TIMEOUT', None, current_price)
                return
            
            # Update database
            self.update_tracking_data(track)
            
        except Exception as e:
            logger.error(f"Error checking signal {signal_id}: {e}")
    
    def complete_signal(self, signal_id: str, track: Dict, result: str, 
                       tp_hit: Optional[str], exit_price: float) -> None:
        """
        Complete a signal (TP hit, SL hit, or timeout)
        
        Args:
            signal_id: Signal ID
            track: Tracking data
            result: WIN, LOSS, or TIMEOUT
            tp_hit: Which TP was hit (TP1, TP2, TP3) or None
            exit_price: Exit price
        """
        try:
            track['status'] = 'CLOSED'
            track['exit_price'] = exit_price
            track['exit_time'] = datetime.now()
            track['final_result'] = result
            track['time_in_trade_seconds'] = int(
                (datetime.now() - track['start_time']).total_seconds()
            )
            
            # Calculate final PnL
            if track['direction'] == 'LONG':
                track['final_pnl'] = exit_price - track['entry_price']
                track['final_pnl_percentage'] = (track['final_pnl'] / track['entry_price']) * 100
            else:  # SHORT
                track['final_pnl'] = track['entry_price'] - exit_price
                track['final_pnl_percentage'] = (track['final_pnl'] / track['entry_price']) * 100
            
            # Log result
            emoji = "✅" if result == "WIN" else ("❌" if result == "LOSS" else "⏱️ ")
            logger.info(f"{emoji} {result}: {signal_id}")
            logger.info(f"   Exit: {exit_price} | PnL: ${track['final_pnl']:.2f} ({track['final_pnl_percentage']:.2f}%)")
            
            # Save to database
            self.update_tracking_data(track, completed=True)
            
            # Move to completed
            with self.lock:
                if signal_id in self.active_signals:
                    del self.active_signals[signal_id]
                self.completed_signals.append(track)
            
        except Exception as e:
            logger.error(f"Error completing signal: {e}")
    
    def update_all_signals(self) -> None:
        """Check all active signals"""
        with self.lock:
            signal_ids = list(self.active_signals.keys())
        
        for signal_id in signal_ids:
            if signal_id in self.active_signals:
                track = self.active_signals[signal_id]
                self.check_signal(signal_id, track)
    
    def save_to_database(self, track: Dict) -> bool:
        """Save tracking data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO live_signals
                    (signal_id, symbol, direction, entry_price, stop_loss,
                     take_profit_1, take_profit_2, take_profit_3,
                     confluence_score, start_time, status, current_price,
                     current_pnl, current_pnl_percentage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    track['signal_id'],
                    track['symbol'],
                    track['direction'],
                    track['entry_price'],
                    track['stop_loss'],
                    track['tp1'],
                    track['tp2'],
                    track['tp3'],
                    track['confluence_score'],
                    track['start_time'],
                    track['status'],
                    track['current_price'],
                    track['current_pnl'],
                    track['current_pnl_percentage']
                ))
                
                conn.commit()
            
            return True
            
        except sqlite3.IntegrityError:
            # Already exists, update instead
            return self.update_tracking_data(track)
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            return False
    
    def update_tracking_data(self, track: Dict, completed: bool = False) -> bool:
        """Update tracking data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if completed:
                    conn.execute("""
                        UPDATE live_signals
                        SET status = ?, exit_price = ?, exit_time = ?,
                            final_result = ?, final_pnl = ?, 
                            final_pnl_percentage = ?, time_in_trade_seconds = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE signal_id = ?
                    """, (
                        track['status'],
                        track.get('exit_price'),
                        track.get('exit_time'),
                        track.get('final_result'),
                        track.get('final_pnl'),
                        track.get('final_pnl_percentage'),
                        track.get('time_in_trade_seconds'),
                        track['signal_id']
                    ))
                else:
                    conn.execute("""
                        UPDATE live_signals
                        SET current_price = ?, current_pnl = ?,
                            current_pnl_percentage = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE signal_id = ?
                    """, (
                        track['current_price'],
                        track['current_pnl'],
                        track['current_pnl_percentage'],
                        track['signal_id']
                    ))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating database: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get real-time statistics"""
        with self.lock:
            completed = list(self.completed_signals)
        
        if not completed:
            return {
                'total_completed': 0,
                'active_signals': len(self.active_signals),
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0
            }
        
        wins = len([s for s in completed if s.get('final_result') == 'WIN'])
        losses = len([s for s in completed if s.get('final_result') == 'LOSS'])
        total_pnl = sum(s.get('final_pnl', 0) for s in completed)
        
        return {
            'total_completed': len(completed),
            'active_signals': len(self.active_signals),
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / len(completed) * 100) if completed else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(completed) if completed else 0
        }
    
    def start(self) -> None:
        """Start tracking in background thread"""
        if self.running:
            logger.warning("Tracker already running")
            return
        
        self.running = True
        
        def track_loop():
            logger.info("🚀 Live signal tracker started")
            
            while self.running:
                try:
                    self.update_all_signals()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Error in tracking loop: {e}")
                    time.sleep(1)
            
            logger.info("⛔ Live signal tracker stopped")
        
        thread = threading.Thread(target=track_loop, daemon=True)
        thread.start()
    
    def stop(self) -> None:
        """Stop tracking"""
        self.running = False
        logger.info("Stopping tracker...")


if __name__ == "__main__":
    # Example usage
    tracker = LiveSignalTracker()
    
    # Start tracker
    tracker.start()
    
    # Simulate adding signals
    print("\n" + "="*60)
    print("LIVE SIGNAL TRACKER DEMO")
    print("="*60)
    
    # In real usage, signals would come from your signal generator
    print("\n(In production, signals would be added from signal generator)")
    print("(This is just to demonstrate the tracker structure)")
    
    # Keep running for demo
    try:
        while True:
            stats = tracker.get_statistics()
            print(f"\n📊 Live Stats: {stats['active_signals']} active, "
                  f"{stats['total_completed']} completed, "
                  f"Win Rate: {stats['win_rate']:.1f}%, "
                  f"Total PnL: ${stats['total_pnl']:.2f}")
            time.sleep(5)
    except KeyboardInterrupt:
        tracker.stop()
        print("\n✅ Tracker stopped")
