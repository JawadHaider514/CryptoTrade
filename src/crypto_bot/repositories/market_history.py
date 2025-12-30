#!/usr/bin/env python3
"""
Market History Repository
========================
Provides access to historical market data (OHLCV candles) from Parquet storage.

Functions:
    get_market_history() - Get singleton instance
    
Classes:
    MarketHistory - Manages OHLCV data access with caching
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketHistory:
    """Provides access to historical OHLCV market data."""
    
    def __init__(self, data_dir: str = "data/ohlcv"):
        """
        Initialize market history repository.
        
        Args:
            data_dir: Path to OHLCV data directory (relative to project root)
        """
        # Convert to absolute path if relative
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            # Get PROJECT_ROOT
            try:
                project_root = Path(__file__).resolve()
                while project_root.name != "crypto_trading_system" and project_root.parent != project_root:
                    project_root = project_root.parent
                data_path = project_root / data_dir
            except Exception:
                data_path = Path(data_dir).resolve()
        
        self.data_dir = data_path
        self._cache = {}  # Symbol -> {timeframe -> DataFrame}
        
        logger.info(f"🗂️ MarketHistory initialized with data_dir={self.data_dir}")
    
    def get_recent_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """
        Get recent candles for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Timeframe (e.g., 15m, 1h)
            limit: Maximum number of candles to return (latest first)
        
        Returns:
            List of dicts with keys: time, open, high, low, close, volume
            Returns None if data not found
        """
        symbol = symbol.upper()
        
        try:
            # Try to load from cache first
            if symbol not in self._cache:
                self._cache[symbol] = {}
            
            if timeframe not in self._cache[symbol]:
                # Load from parquet file
                parquet_path = self.data_dir / symbol / f"{timeframe}.parquet"
                
                if not parquet_path.exists():
                    logger.warning(f"Candle data not found for {symbol} {timeframe}: {parquet_path}")
                    return None
                
                # Load parquet
                df = pd.read_parquet(str(parquet_path))
                
                # Ensure required columns exist
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing required columns in {parquet_path}. Found: {list(df.columns)}")
                    return None
                
                # Cache the dataframe
                self._cache[symbol][timeframe] = df.copy()
                logger.debug(f"Loaded {len(df)} candles for {symbol} {timeframe}")
            
            # Get cached dataframe
            df = self._cache[symbol][timeframe]
            
            if df.empty:
                logger.debug(f"No candle data for {symbol} {timeframe}")
                return None
            
            # Sort by timestamp (ascending) and get latest records
            df_sorted = df.sort_values('timestamp', ascending=True)
            df_latest = df_sorted.tail(limit)
            
            # Convert to list of dicts
            candles = []
            for _, row in df_latest.iterrows():
                candle = {
                    "time": row['timestamp'] if isinstance(row['timestamp'], str) else row['timestamp'].isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                }
                candles.append(candle)
            
            logger.debug(f"✓ Retrieved {len(candles)} candles for {symbol} {timeframe}")
            return candles
        
        except Exception as e:
            logger.error(f"Error retrieving candles for {symbol} {timeframe}: {e}")
            return None
    
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """
        Get candles with optional time range filtering.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Timeframe (e.g., 15m, 1h)
            start_time: ISO format start time (inclusive)
            end_time: ISO format end time (inclusive)
            limit: Maximum number of candles (applied after filtering)
        
        Returns:
            List of dicts with keys: time, open, high, low, close, volume
            Returns None if data not found
        """
        symbol = symbol.upper()
        
        try:
            # Load candles using get_recent_candles (which handles caching)
            # For unlimited data, use a very large limit
            all_candles = self.get_recent_candles(symbol, timeframe, limit=999999)
            
            if not all_candles:
                return None
            
            # Filter by time range if provided
            if start_time or end_time:
                filtered = []
                for candle in all_candles:
                    candle_time = candle['time']
                    
                    # Parse ISO format timestamps
                    if isinstance(candle_time, str):
                        try:
                            candle_dt = datetime.fromisoformat(candle_time.replace('Z', '+00:00'))
                        except:
                            candle_dt = datetime.fromisoformat(candle_time)
                    else:
                        candle_dt = candle_time
                    
                    if start_time:
                        try:
                            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        except:
                            start_dt = datetime.fromisoformat(start_time)
                        if candle_dt < start_dt:
                            continue
                    
                    if end_time:
                        try:
                            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                        except:
                            end_dt = datetime.fromisoformat(end_time)
                        if candle_dt > end_dt:
                            continue
                    
                    filtered.append(candle)
                
                all_candles = filtered
            
            # Apply limit if specified
            if limit:
                all_candles = all_candles[-limit:]
            
            return all_candles
        
        except Exception as e:
            logger.error(f"Error retrieving filtered candles for {symbol} {timeframe}: {e}")
            return None
    
    def get_dataframe(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Get raw DataFrame for advanced analysis.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Timeframe (e.g., 15m, 1h)
        
        Returns:
            pandas DataFrame or None if not found
        """
        symbol = symbol.upper()
        
        try:
            # Try cache first
            if symbol in self._cache and timeframe in self._cache[symbol]:
                return self._cache[symbol][timeframe].copy()
            
            # Load from file
            parquet_path = self.data_dir / symbol / f"{timeframe}.parquet"
            
            if not parquet_path.exists():
                logger.warning(f"Candle data not found for {symbol} {timeframe}")
                return None
            
            df = pd.read_parquet(str(parquet_path))
            
            # Cache it
            if symbol not in self._cache:
                self._cache[symbol] = {}
            self._cache[symbol][timeframe] = df.copy()
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading DataFrame for {symbol} {timeframe}: {e}")
            return None
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Clear in-memory cache to free memory.
        
        Args:
            symbol: Clear only this symbol (None = clear all)
            timeframe: Clear only this timeframe (requires symbol)
        """
        if symbol is None:
            self._cache.clear()
            logger.info("Cleared all market history cache")
        elif symbol in self._cache:
            if timeframe is None:
                del self._cache[symbol]
                logger.info(f"Cleared cache for {symbol}")
            elif timeframe in self._cache[symbol]:
                del self._cache[symbol][timeframe]
                logger.info(f"Cleared cache for {symbol} {timeframe}")
    
    def available_symbols(self) -> List[str]:
        """Get list of symbols with available data."""
        try:
            if not self.data_dir.exists():
                return []
            
            symbols = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
            return sorted(symbols)
        except Exception as e:
            logger.error(f"Error listing available symbols: {e}")
            return []
    
    def available_timeframes(self, symbol: str) -> List[str]:
        """Get list of available timeframes for a symbol."""
        try:
            symbol = symbol.upper()
            symbol_dir = self.data_dir / symbol
            
            if not symbol_dir.exists():
                return []
            
            timeframes = [f.stem for f in symbol_dir.glob("*.parquet")]
            return sorted(timeframes)
        except Exception as e:
            logger.error(f"Error listing timeframes for {symbol}: {e}")
            return []


# Singleton instance
_market_history_instance: Optional[MarketHistory] = None


def get_market_history(data_dir: str = "data/ohlcv") -> MarketHistory:
    """
    Get or create the singleton MarketHistory instance.
    
    Args:
        data_dir: Path to OHLCV data directory
    
    Returns:
        MarketHistory instance
    """
    global _market_history_instance
    
    if _market_history_instance is None:
        _market_history_instance = MarketHistory(data_dir)
    
    return _market_history_instance
