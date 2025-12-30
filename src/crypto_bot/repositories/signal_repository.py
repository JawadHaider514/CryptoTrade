#!/usr/bin/env python3
"""
Signal Repository
================
Stores and retrieves trading signals path:
"""

import sqlite3
import threading
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from crypto_bot.domain.signal_models import SignalModel

logger = logging.getLogger(__name__)


def _json_safe(obj):
    """Recursively convert datetime objects to ISO format strings for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


class SignalRepository:
    """Stores signals with in-memory cache + SQLite persistence"""
    
    def __init__(self, use_sqlite: bool = True, db_path: str = "data/signals.db"):
        """
        Args:
            use_sqlite: Enable SQLite persistence
            db_path: Path to SQLite database (will be converted to absolute path)
        """
        self.use_sqlite = use_sqlite
        
        # Convert to absolute path if relative
        if isinstance(db_path, str):
            db_path_obj = Path(db_path)
            if not db_path_obj.is_absolute():
                # Get PROJECT_ROOT
                try:
                    project_root = Path(__file__).resolve()
                    while project_root.name != "crypto_trading_system" and project_root.parent != project_root:
                        project_root = project_root.parent
                    db_path_obj = project_root / db_path
                except Exception:
                    db_path_obj = Path(db_path).resolve()
            self.db_path = str(db_path_obj)
        else:
            self.db_path = str(Path(db_path).resolve())
        
        logger.info(f"📁 Signal Repository DB: {self.db_path}")
        
        # In-memory cache (primary storage)
        self.cache: Dict[str, SignalModel] = {}
        self.history: Dict[str, List[SignalModel]] = {}
        self.lock = threading.Lock()
        
        # Initialize SQLite if enabled
        if self.use_sqlite:
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT,
                        timestamp DATETIME NOT NULL,
                        direction TEXT,
                        entry_price REAL,
                        stop_loss REAL,
                        take_profits TEXT,
                        confidence_score INTEGER,
                        accuracy_percent REAL,
                        leverage INTEGER,
                        valid_until DATETIME,
                        current_price REAL,
                        reasons TEXT,
                        patterns TEXT,
                        market_context TEXT,
                        signal_json TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Index for fast lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                    ON signals(symbol, timestamp DESC)
                """)
                conn.commit()
            
            logger.info(f"✅ SQLite initialized at {self.db_path}")
        
        except Exception as e:
            logger.error(f"❌ SQLite init error: {e}")
    
    def upsert_latest(self, signal: SignalModel) -> None:
        """Update or insert latest signal for a symbol (in cache + DB)"""
        
        if not signal:
            logger.warning("⚠️ Attempted to upsert_latest with None signal")
            return
        
        with self.lock:
            # Update cache
            old_signal = self.cache.get(signal.symbol)
            self.cache[signal.symbol] = signal
            logger.info(f"💾 CACHE UPSERTED: {signal.symbol} -> {signal.direction} @ ${signal.entry_price:.2f} (was: {old_signal.direction if old_signal else 'NEW'})")
            logger.info(f"   Cache size now: {len(self.cache)} symbols")
            
            # Add to history
            if signal.symbol not in self.history:
                self.history[signal.symbol] = []
            
            self.history[signal.symbol].append(signal)
            
            # Keep last 100 in history
            if len(self.history[signal.symbol]) > 100:
                self.history[signal.symbol] = self.history[signal.symbol][-100:]
        
        # Store in SQLite
        if self.use_sqlite:
            self._store_signal(signal)
    
    def _store_signal(self, signal: SignalModel):
        """Store signal in SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert Pydantic model to dict
                signal_dict = signal.dict() if hasattr(signal, 'dict') else signal.model_dump()
                
                # Make all objects JSON-safe (convert datetime to isoformat)
                take_profits_safe = _json_safe([tp.dict() if hasattr(tp, 'dict') else tp.model_dump() for tp in signal.take_profits])
                reasons_safe = _json_safe(signal.reasons or [])
                patterns_safe = _json_safe(signal.patterns or [])
                signal_dict_safe = _json_safe(signal_dict)
                
                conn.execute("""
                    INSERT INTO signals (
                        symbol, timeframe, timestamp, direction, entry_price,
                        stop_loss, take_profits, confidence_score, accuracy_percent,
                        leverage, valid_until, current_price, reasons, patterns,
                        market_context, signal_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.symbol,
                    signal.timeframe,
                    signal.timestamp.isoformat(),
                    signal.direction,
                    signal.entry_price,
                    signal.stop_loss,
                    json.dumps(take_profits_safe),
                    signal.confidence_score,
                    signal.accuracy_percent,
                    signal.leverage,
                    signal.valid_until.isoformat(),
                    signal.current_price,
                    json.dumps(reasons_safe),
                    json.dumps(patterns_safe),
                    signal.market_context,
                    json.dumps(signal_dict_safe)
                ))
                conn.commit()
                logger.info(f"✅ DB STORED: {signal.symbol} saved to SQLite")
        
        except Exception as e:
            logger.error(f"❌ Error storing signal for {signal.symbol}: {e}")
    
    def get_latest(self, symbol: str) -> Optional[SignalModel]:
        """Get latest signal for ONE symbol (from cache)"""
        with self.lock:
            return self.cache.get(symbol)
    
    def get_latest_all(self) -> Dict[str, SignalModel]:
        """Get latest signal for all symbols (from cache, reload from DB if empty)"""
        with self.lock:
            result = dict(self.cache)
            logger.debug(f"📊 CACHE READ: Retrieved {len(result)} signals from cache: {list(result.keys())}")
            
            # If cache empty, try loading from DB
            if not result and self.use_sqlite:
                logger.warning(f"⚠️ CACHE EMPTY: Attempting to reload from DB...")
                try:
                    self._load_from_db()
                    result = dict(self.cache)
                    if result:
                        logger.info(f"✅ Reloaded {len(result)} signals from DB")
                except Exception as e:
                    logger.error(f"❌ DB reload failed: {e}")
            
            if not result:
                logger.warning(f"⚠️ CACHE EMPTY: No signals in memory cache or DB")
            return result
    
    def _load_from_db(self):
        """Reload signals from SQLite database into cache"""
        if not self.use_sqlite:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, timeframe, timestamp, direction, entry_price, stop_loss, 
                           take_profits, confidence_score, accuracy_percent, leverage, valid_until,
                           current_price, reasons, patterns, market_context, signal_json
                    FROM signals
                    ORDER BY timestamp DESC
                """)
                
                signals_by_symbol = {}
                for row in cursor.fetchall():
                    try:
                        symbol = row[0]
                        
                        # Use signal_json if available, otherwise reconstruct
                        if row[15]:  # signal_json column
                            signal_dict = json.loads(row[15])
                            signal = SignalModel.model_validate(signal_dict)
                        else:
                            # Reconstruct from columns
                            signal = SignalModel(
                                symbol=symbol,
                                timeframe=row[1],
                                timestamp=datetime.fromisoformat(row[2].replace('Z', '+00:00')) if isinstance(row[2], str) else row[2],
                                direction=row[3],
                                entry_price=row[4],
                                stop_loss=row[5],
                                take_profits=json.loads(row[6]) if isinstance(row[6], str) else row[6] or [],
                                confidence_score=row[7],
                                accuracy_percent=row[8],
                                leverage=row[9],
                                valid_until=datetime.fromisoformat(row[10].replace('Z', '+00:00')) if isinstance(row[10], str) else row[10],
                                current_price=row[11],
                                reasons=json.loads(row[12]) if isinstance(row[12], str) else row[12] or [],
                                patterns=json.loads(row[13]) if isinstance(row[13], str) else row[13] or [],
                                market_context=row[14] or "",
                                source="DB"
                            )
                        
                        # Keep latest per symbol
                        if symbol not in signals_by_symbol:
                            signals_by_symbol[symbol] = signal
                    except Exception as parse_err:
                        logger.warning(f"Failed to parse signal from DB: {parse_err}")
                        continue
                
                # Update cache with loaded signals
                self.cache.update(signals_by_symbol)
                logger.info(f"✅ DB RELOAD: Loaded {len(signals_by_symbol)} signals into cache")
                
        except Exception as e:
            logger.error(f"❌ DB load failed: {e}")
            raise
    
    def get_history(self, symbol: str, limit: int = 100) -> List[SignalModel]:
        """Get past N signals for a symbol"""
        with self.lock:
            history = self.history.get(symbol, [])
            return history[-limit:] if limit else history
    
    def clear_cache(self):
        """Clear all in-memory cache"""
        with self.lock:
            self.cache.clear()
            self.history.clear()
    
    def get_stats(self, symbol: str) -> Dict:
        """Get statistics for a symbol"""
        history = self.get_history(symbol, limit=100)
        
        if not history:
            return {"total": 0, "avg_confidence": 0, "avg_accuracy": 0}
        
        return {
            "total": len(history),
            "avg_confidence": sum(s.confidence_score for s in history) / len(history),
            "avg_accuracy": sum(s.accuracy_percent for s in history) / len(history),
            "latest_timestamp": history[-1].timestamp.isoformat()
        }
