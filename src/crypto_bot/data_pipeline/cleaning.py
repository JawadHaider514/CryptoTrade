"""
Data cleaning and validation for OHLCV data.

Functions:
    clean_ohlcv() - Remove duplicates, sort, validate continuity
"""

import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def clean_ohlcv(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """
    Clean OHLCV data.
    
    Operations:
    1. Remove duplicate timestamps (keep first)
    2. Sort by timestamp ascending
    3. Validate dtype (ensure float columns)
    4. Check for missing candles (log warnings but don't fill)
    
    Args:
        df: Raw DataFrame from Binance
        symbol: Trading symbol (for logging)
        interval: Timeframe (for logging)
    
    Returns:
        Cleaned DataFrame
    """
    original_len = len(df)
    
    if df.empty:
        logger.warning(f"Empty DataFrame for {symbol} {interval}")
        return df
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    removed_duplicates = original_len - len(df)
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} duplicate timestamps from {symbol} {interval}")
    
    # Sort by timestamp ascending
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure float dtypes for OHLCV columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for missing candles (based on interval)
    if len(df) > 1:
        _check_candle_continuity(df, symbol, interval)
    
    logger.info(f"Cleaned {symbol} {interval}: {len(df)} candles")
    
    return df


def _check_candle_continuity(df: pd.DataFrame, symbol: str, interval: str) -> None:
    """
    Check for missing candles and log warnings.
    
    Args:
        df: Sorted DataFrame with timestamps
        symbol: Trading symbol
        interval: Timeframe
    """
    # Map interval to minutes
    interval_map = {
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
    }
    
    expected_minutes = interval_map.get(interval)
    if not expected_minutes:
        return
    
    # Check time differences between consecutive candles
    df_copy = df.copy()
    df_copy['time_diff_minutes'] = (
        df_copy['timestamp'].diff().dt.total_seconds() / 60
    )
    
    # Skip first row (NaT diff)
    missing_candles = df_copy[df_copy['time_diff_minutes'] != expected_minutes]
    
    if len(missing_candles) > 0:
        logger.warning(
            f"{symbol} {interval}: Detected {len(missing_candles)} gaps in candle continuity"
        )
    
    return
