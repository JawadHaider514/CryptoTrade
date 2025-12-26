#!/usr/bin/env python3
"""
TASK 1.2: Historical Signal Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate signals using PAST data (not live).
Only use data BEFORE the signal timestamp (no peeking into future).
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalSignalGenerator:
    """Generate trading signals from historical data"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        """Initialize signal generator"""
        self.db_path = db_path
        self.init_database()
        
        # Configuration
        self.pattern_scores = {
            "doji": 8,
            "hammer": 12,
            "bullish_engulfing": 15,
            "bearish_engulfing": 12,
            "shooting_star": 10,
            "morning_star": 14
        }
        
        self.min_confluence_score = 15  # Lowered to allow EMA-only signals
        self.min_volume_surge = 0.9
        
        # Load ML model if available
        self.ml_model = None
        self.ml_scaler = None
        self.ml_features = None
        self.ml_threshold = 0.56  # Based on real model test accuracy
        
        self._load_ml_model()
        self.min_volume_surge = 0.9
    
    def init_database(self):
        """Create backtest_signals table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL NOT NULL,
                    take_profit_2 REAL NOT NULL,
                    take_profit_3 REAL NOT NULL,
                    confluence_score INTEGER NOT NULL,
                    patterns TEXT,
                    rsi REAL,
                    macd REAL,
                    volume_ratio REAL,
                    ema_9 REAL,
                    ema_21 REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_timestamp 
                ON backtest_signals(timestamp, symbol)
            """)
            
            conn.commit()
    
    def _load_ml_model(self):
        """Load trained ML model if available"""
        try:
            model_path = Path("models/signal_predictor.pkl")
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.ml_model = model_data.get('model')
                self.ml_scaler = model_data.get('scaler')
                self.ml_features = model_data.get('features', [])
                
                logger.info(f"✅ ML model loaded: {model_path.stat().st_size / 1024:.1f} KB")
                logger.info(f"   Features: {self.ml_features}")
                logger.info(f"   Threshold: {self.ml_threshold*100:.1f}% (test accuracy)")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load ML model: {e}")
            self.ml_model = None
    
    def _predict_ml_win_probability(self, signal: Dict) -> Optional[float]:
        """Predict win probability for signal using trained ML model"""
        if not self.ml_model or not self.ml_scaler:
            return None
        
        try:
            # Extract features in same order as training
            features = np.asarray([[
                signal.get('confluence_score', 0),
                signal.get('rsi', 50),
                signal.get('macd', 0),
                signal.get('volume_ratio', 1),
                signal.get('ema_9', 0)
            ]], dtype=float)
            
            # Scale features using trained scaler
            features_scaled = self.ml_scaler.transform(features)
            
            # Get probability of WIN class
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            win_prob = probabilities[1]  # Probability of class 1 (WIN)
            
            return win_prob
            
        except Exception as e:
            logger.debug(f"ML prediction error: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timestamp: int, lookback_minutes: int = 100) -> pd.DataFrame:
        """
        Get historical data BEFORE a specific timestamp.
        Never peek into the future!
        Uses caching to avoid repeated database queries.
        """
        # Pre-calculate start_ms so it's available even if DB queries fail early
        start_ms = timestamp - (lookback_minutes * 60 * 1000)
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # Set connection to readonly and fast mode
                conn.row_factory = sqlite3.Row
                
                # Use parameterized query with timeout
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM historical_candles
                    WHERE symbol = ? AND timestamp < ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                    LIMIT 10000
                """
                
                df = pd.read_sql_query(query, conn, params=(symbol, timestamp, start_ms))
                
                if df.empty:
                    return pd.DataFrame()
                
                # Convert timestamps
                df['datetime'] = pd.to_datetime(df['timestamp'] / 1000, unit='s')
                
                return df
                
        except sqlite3.OperationalError as e:
            logger.warning(f"Database query timeout or lock - retrying: {e}")
            # Retry with timeout increase
            try:
                with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                    query = """
                        SELECT timestamp, open, high, low, close, volume
                        FROM historical_candles
                        WHERE symbol = ? AND timestamp < ? AND timestamp >= ?
                        ORDER BY timestamp ASC
                        LIMIT 10000
                    """
                    df = pd.read_sql_query(query, conn, params=(symbol, timestamp, start_ms))
                    if not df.empty:
                        df['datetime'] = pd.to_datetime(df['timestamp'] / 1000, unit='s')
                    return df
            except Exception as retry_err:
                logger.error(f"Error getting historical data after retry: {retry_err}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate RSI, MACD, EMA, etc."""
        if df.empty or len(df) < 21:
            return {}
        
        close = np.asarray(df['close'].values, dtype=float)
        high = np.asarray(df['high'].values, dtype=float)
        low = np.asarray(df['low'].values, dtype=float)
        volume = np.asarray(df['volume'].values, dtype=float)
        
        indicators = {}
        
        # RSI (14 period)
        indicators['rsi'] = self.calculate_rsi(close, period=14)
        
        # MACD (12, 26, 9)
        macd, signal, hist = self.calculate_macd(close)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_hist'] = hist
        
        # EMA
        indicators['ema_9'] = self.calculate_ema(close, 9)
        indicators['ema_21'] = self.calculate_ema(close, 21)
        
        # Volume
        indicators['avg_volume'] = np.mean(volume[-20:])
        indicators['current_volume'] = volume[-1]
        indicators['volume_ratio'] = volume[-1] / indicators['avg_volume'] if indicators['avg_volume'] > 0 else 1
        
        # ATR (14 period)
        indicators['atr'] = self.calculate_atr(high, low, close, period=14)
        
        return indicators
    
    def calculate_rsi(self, prices, period: int = 14) -> float:
        """Calculate RSI indicator"""
        prices = np.asarray(prices, dtype=float)
        if prices.size < period:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs >= 0 else 0
        
        return float(rsi)
    
    def calculate_macd(self, prices) -> Tuple[float, float, float]:
        """Calculate MACD"""
        prices = np.asarray(prices, dtype=float)
        if prices.size < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        macd = ema_12 - ema_26
        
        # Signal line (9-period EMA of MACD)
        # Simplified: using approximate signal
        signal = macd * 0.7  # Simplified
        
        hist = macd - signal
        
        return float(macd), float(signal), float(hist)
    
    def calculate_ema(self, prices, period: int) -> float:
        """Calculate EMA"""
        prices = np.asarray(prices, dtype=float)
        if prices.size < period:
            return float(prices[-1])
        
        ema = prices[-period:].mean()
        multiplier = 2 / (period + 1)
        
        for price in prices[-period:]:
            ema = price * multiplier + ema * (1 - multiplier)
        
        return float(ema)
    
    def calculate_atr(self, high, low, close, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        if close.size < period:
            return float(high[-1] - low[-1])
        
        tr = []
        for i in range(len(close)):
            h_l = high[i] - low[i]
            h_c = abs(high[i] - close[i-1]) if i > 0 else h_l
            l_c = abs(low[i] - close[i-1]) if i > 0 else h_l
            tr.append(max(h_l, h_c, l_c))
        
        atr = np.mean(tr[-period:])
        return float(atr)
    
    def detect_patterns(self, df: pd.DataFrame) -> Tuple[List[str], int]:
        """
        Detect candlestick patterns
        Returns: (pattern_names, confluence_score)
        """
        if len(df) < 2:
            return [], 0
        
        patterns = []
        score = 0
        
        # Get recent candles
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        
        # Current candle
        open_curr = o[-1]
        high_curr = h[-1]
        low_curr = l[-1]
        close_curr = c[-1]
        
        # Previous candle
        open_prev = o[-2] if len(o) > 1 else open_curr
        high_prev = h[-2] if len(h) > 1 else high_curr
        low_prev = l[-2] if len(l) > 1 else low_curr
        close_prev = c[-2] if len(c) > 1 else close_curr
        
        body_curr = abs(close_curr - open_curr)
        body_prev = abs(close_prev - open_prev)
        range_curr = high_curr - low_curr
        
        # Doji pattern (small body, long wicks)
        if body_curr < range_curr * 0.1:
            patterns.append("doji")
            score += self.pattern_scores.get("doji", 8)
        
        # Hammer (small body at top, long lower wick)
        if (close_curr > open_curr and 
            body_curr < range_curr * 0.3 and 
            (low_curr - open_curr) > range_curr * 0.5):
            patterns.append("hammer")
            score += self.pattern_scores.get("hammer", 12)
        
        # Bullish engulfing
        if (close_prev < open_prev and 
            close_curr > open_curr and 
            close_curr > open_prev and 
            open_curr < close_prev):
            patterns.append("bullish_engulfing")
            score += self.pattern_scores.get("bullish_engulfing", 15)
        
        # Bearish engulfing
        if (close_prev > open_prev and 
            close_curr < open_curr and 
            close_curr < open_prev and 
            open_curr > close_prev):
            patterns.append("bearish_engulfing")
            score += self.pattern_scores.get("bearish_engulfing", 12)
        
        # Shooting star
        if (close_curr < open_curr and 
            (high_curr - close_curr) > range_curr * 0.5 and 
            body_curr < range_curr * 0.3):
            patterns.append("shooting_star")
            score += self.pattern_scores.get("shooting_star", 10)
        
        return patterns, score
    
    def generate_signal(self, symbol: str, timestamp: int) -> Optional[Dict]:
        """
        Generate a trading signal at a specific point in time.
        Only uses data from BEFORE that timestamp.
        """
        # Get historical data
        df = self.get_historical_data(symbol, timestamp, lookback_minutes=100)
        
        if df.empty or len(df) < 21:
            return None
        
        # Calculate indicators
        indicators = self.calculate_technical_indicators(df)
        
        if not indicators:
            return None
        
        # Detect patterns
        patterns, pattern_score = self.detect_patterns(df)
        
        # Calculate confluence score
        confluence_score = pattern_score
        
        # Add RSI confluence
        rsi = indicators.get('rsi', 50)
        if 30 < rsi < 70:
            confluence_score += 5
        
        # Add MACD confluence
        macd_hist = indicators.get('macd_hist', 0)
        if macd_hist != 0:
            confluence_score += 5
        
        # Volume surge
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > self.min_volume_surge:
            confluence_score += 10
        
        # Check minimum threshold
        if confluence_score < self.min_confluence_score:
            return None
        
        # Determine direction
        ema_9 = indicators.get('ema_9', df['close'].iloc[-1])
        ema_21 = indicators.get('ema_21', df['close'].iloc[-1])
        current_price = df['close'].iloc[-1]
        atr = indicators.get('atr', current_price * 0.01)
        
        if ema_9 > ema_21:
            direction = "LONG"
            entry = current_price
            stop_loss = current_price - (atr * 1.5)
            
            # Calculate take profits
            risk = entry - stop_loss
            take_profit_1 = entry + (risk * 0.5)
            take_profit_2 = entry + (risk * 1.0)
            take_profit_3 = entry + (risk * 1.5)
            
        elif ema_9 < ema_21:
            direction = "SHORT"
            entry = current_price
            stop_loss = current_price + (atr * 1.5)
            
            risk = stop_loss - entry
            take_profit_1 = entry - (risk * 0.5)
            take_profit_2 = entry - (risk * 1.0)
            take_profit_3 = entry - (risk * 1.5)
        else:
            return None
        
        # Create signal
        signal = {
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'entry_price': round(entry, 8),
            'stop_loss': round(stop_loss, 8),
            'take_profit_1': round(take_profit_1, 8),
            'take_profit_2': round(take_profit_2, 8),
            'take_profit_3': round(take_profit_3, 8),
            'confluence_score': confluence_score,
            'patterns': patterns,
            'rsi': round(indicators.get('rsi', 50), 2),
            'macd': round(indicators.get('macd', 0), 6),
            'volume_ratio': round(volume_ratio, 2),
            'ema_9': round(ema_9, 8),
            'ema_21': round(ema_21, 8),
            'atr': round(atr, 8)
        }
        
        # Apply ML filtering if model is available
        if self.ml_model:
            ml_win_prob = self._predict_ml_win_probability(signal)
            if ml_win_prob is not None:
                signal['ml_win_probability'] = round(ml_win_prob * 100, 2)
                
                # Filter out low-quality signals below threshold
                if ml_win_prob < self.ml_threshold:
                    # Signal filtered out - below ML threshold
                    return None
        
        return signal
    
    def save_signal(self, signal: Dict) -> bool:
        """Save signal to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO backtest_signals
                    (timestamp, symbol, direction, entry_price, stop_loss,
                     take_profit_1, take_profit_2, take_profit_3,
                     confluence_score, patterns, rsi, macd, volume_ratio,
                     ema_9, ema_21, ml_win_probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal['timestamp'],
                    signal['symbol'],
                    signal['direction'],
                    signal['entry_price'],
                    signal['stop_loss'],
                    signal['take_profit_1'],
                    signal['take_profit_2'],
                    signal['take_profit_3'],
                    signal['confluence_score'],
                    json.dumps(signal['patterns']),
                    signal['rsi'],
                    signal['macd'],
                    signal['volume_ratio'],
                    signal['ema_9'],
                    signal['ema_21'],
                    signal.get('ml_win_probability', None)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return False
    
    def generate_signals_for_period(self, symbol: str, start_time: datetime, 
                                   end_time: datetime, interval_minutes: int = 5) -> int:
        """
        Generate signals for a historical period.
        Useful for backtesting.
        """
        logger.info(f"📊 Generating signals for {symbol} from {start_time} to {end_time}")
        logger.info(f"   Interval: {interval_minutes} minutes")
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        interval_ms = interval_minutes * 60 * 1000
        
        signal_count = 0
        current_ms = start_ms
        total_iterations = (end_ms - start_ms) // interval_ms
        
        logger.info(f"   Estimated iterations: {total_iterations}")
        
        iteration = 0
        while current_ms < end_ms:
            iteration += 1
            
            # Log progress every 50 iterations
            if iteration % 50 == 0:
                progress_pct = (iteration / total_iterations) * 100 if total_iterations > 0 else 0
                logger.info(f"   Progress: {iteration}/{total_iterations} ({progress_pct:.0f}%) - Signals: {signal_count}")
            
            try:
                signal = self.generate_signal(symbol, current_ms)
                
                if signal:
                    self.save_signal(signal)
                    signal_count += 1
                    if signal_count <= 5 or signal_count % 50 == 0:
                        logger.info(f"   ✓ Signal #{signal_count}: {signal['direction']} @ {signal['entry_price']} (score: {signal['confluence_score']})")
            
            except Exception as e:
                logger.warning(f"   ⚠️ Error generating signal at iteration {iteration}: {str(e)[:100]}")
            
            current_ms += interval_ms
        
        logger.info(f"✅ Generated {signal_count} signals in {iteration} iterations")
        return signal_count
    
    def get_signals(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get generated signals from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM backtest_signals
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, conn, params=(symbol, limit))
                
                if not df.empty:
                    df['patterns'] = df['patterns'].apply(json.loads)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    generator = HistoricalSignalGenerator()
    
    print("\n" + "="*60)
    print("HISTORICAL SIGNAL GENERATOR")
    print("="*60)
    
    # Generate signals for last 7 days
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    signal_count = generator.generate_signals_for_period(
        "XRPUSDT",
        start_time,
        end_time,
        interval_minutes=5
    )
    
    print(f"\n✅ Total signals generated: {signal_count}")
    
    # Show sample signals
    print("\n" + "="*60)
    print("SAMPLE SIGNALS (most recent)")
    print("="*60)
    
    signals = generator.get_signals("XRPUSDT", limit=5)
    print(signals[['timestamp', 'direction', 'entry_price', 'confluence_score', 'patterns']])
