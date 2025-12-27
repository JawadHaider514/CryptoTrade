#!/usr/bin/env python3
"""
Market History Service
======================
Fetch historical candlestick data from Binance with caching and rate limiting.

Responsibility:
- Fetch candles from Binance API
- Convert to pandas DataFrames
- Cache with TTL (30-60 sec per symbol+timeframe)
- Safe rate limit handling
"""

import pandas as pd
import requests
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
import threading


class MarketHistoryService:
    """Fetch and cache market history data from Binance"""
    
    def __init__(self, cache_ttl: int = 45):
        """
        Initialize MarketHistoryService
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 45 sec)
        """
        self.api_url = "https://api.binance.com/api/v3"
        self.cache_ttl = cache_ttl
        self.cache = {}  # {symbol+tf: {'data': df, 'timestamp': time}}
        self.cache_lock = threading.Lock()
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
    def get_dataframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get candlestick DataFrames for a symbol across multiple timeframes
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Dictionary with timeframes as keys and DataFrames as values:
            {
                '1m': pd.DataFrame,
                '5m': pd.DataFrame,
                '15m': pd.DataFrame,
                '1h': pd.DataFrame,
                '4h': pd.DataFrame,
                '1d': pd.DataFrame
            }
        """
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        result = {}
        
        for tf in timeframes:
            try:
                df = self._fetch_or_cache(symbol, tf)
                if df is not None and len(df) > 0:
                    result[tf] = df
                else:
                    result[tf] = pd.DataFrame()  # Empty DataFrame if fetch fails
            except Exception as e:
                print(f"  Error fetching {symbol} {tf}: {e}")
                result[tf] = pd.DataFrame()
        
        return result
    
    def _fetch_or_cache(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from cache or API (with fallback)
        
        Args:
            symbol: Trading pair
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            pandas DataFrame or None
        """
        cache_key = f"{symbol}_{interval}"
        
        # Check cache
        with self.cache_lock:
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                age = time.time() - cached['timestamp']
                
                if age < self.cache_ttl:
                    return cached['data']
        
        # Fetch from API
        df = self._fetch_from_binance(symbol, interval)
        
        # Update cache
        if df is not None and len(df) > 0:
            with self.cache_lock:
                self.cache[cache_key] = {
                    'data': df,
                    'timestamp': time.time()
                }
        
        return df
    
    def _fetch_from_binance(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch candlestick data from Binance REST API
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch (max 1000)
            
        Returns:
            pandas DataFrame or None on error
        """
        try:
            # Rate limiting
            self._rate_limit()
            
            url = f"{self.api_url}/klines"
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": min(limit, 1000)
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"  Binance API error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time',
                'open',
                'high',
                'low',
                'close',
                'volume',
                'close_time',
                'quote_asset_volume',
                'number_of_trades',
                'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume',
                'ignore'
            ])
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamps
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Drop unnecessary columns
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            df.rename(columns={'open_time': 'timestamp'}, inplace=True)
            
            return df
            
        except requests.exceptions.Timeout:
            print(f"  Timeout fetching {symbol} {interval}")
            return None
        except Exception as e:
            print(f"  Error fetching {symbol} {interval}: {e}")
            return None
    
    def _rate_limit(self):
        """
        Enforce rate limiting (100ms between requests)
        Binance allows ~1200 requests/minute
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        
        self.last_request_time = time.time()
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache for a symbol or all symbols
        
        Args:
            symbol: If provided, clear only this symbol's cache. Otherwise clear all.
        """
        with self.cache_lock:
            if symbol:
                # Clear specific symbol
                keys_to_remove = [k for k in self.cache.keys() if k.startswith(symbol)]
                for key in keys_to_remove:
                    del self.cache[key]
            else:
                # Clear all
                self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            total_entries = len(self.cache)
            
            # Count symbols
            symbols = set()
            for key in self.cache.keys():
                symbol = key.rsplit('_', 1)[0]
                symbols.add(symbol)
            
            return {
                'total_entries': total_entries,
                'unique_symbols': len(symbols),
                'symbols': list(symbols),
                'cache_ttl': self.cache_ttl
            }


# Singleton instance for use across application
_market_history_service: Optional[MarketHistoryService] = None

def get_market_history_service(cache_ttl: int = 45) -> MarketHistoryService:
    """
    Get or create singleton MarketHistoryService instance
    
    Args:
        cache_ttl: Cache time-to-live in seconds
        
    Returns:
        MarketHistoryService instance
    """
    global _market_history_service
    
    if _market_history_service is None:
        _market_history_service = MarketHistoryService(cache_ttl=cache_ttl)
    
    return _market_history_service
