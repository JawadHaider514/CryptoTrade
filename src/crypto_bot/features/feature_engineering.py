"""
Feature engineering for OHLCV data.

Functions:
    build_features() - Calculate technical indicators and features
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical indicator features from OHLCV data.
    
    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
    
    Returns:
        DataFrame with original OHLCV + feature columns
    
    Features added:
        - returns: Simple price returns (close/close.shift(1) - 1)
        - log_returns: Log returns (log(close/close.shift(1)))
        - rsi_14: RSI(14)
        - macd: MACD line (12,26)
        - macd_signal: MACD signal (9)
        - macd_diff: MACD - signal
        - atr_14: Average True Range(14)
        - ema_20: EMA(20)
        - ema_50: EMA(50)
        - bb_upper: Bollinger Bands upper (20, 2 std)
        - bb_middle: Bollinger Bands middle (20)
        - bb_lower: Bollinger Bands lower (20, 2 std)
        - bb_width: (upper - lower) / middle
        - volatility: Rolling std of returns (20)
        - volume_change: % change in volume
    """
    df = df.copy()
    
    # Simple returns
    df['returns'] = df['close'].pct_change()
    
    # Log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # RSI(14)
    df['rsi_14'] = _calculate_rsi(df['close'], period=14)
    
    # MACD(12, 26, 9)
    macd, signal, diff = _calculate_macd(df['close'], fast=12, slow=26, signal_period=9)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_diff'] = diff
    
    # ATR(14)
    df['atr_14'] = _calculate_atr(df, period=14)
    
    # EMA(20, 50)
    df['ema_20'] = _calculate_ema(df['close'], period=20)
    df['ema_50'] = _calculate_ema(df['close'], period=50)
    
    # Bollinger Bands(20, 2)
    bb_upper, bb_middle, bb_lower = _calculate_bollinger_bands(df['close'], period=20, num_std=2)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # Volatility (rolling std of returns, 20)
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Volume change %
    df['volume_change'] = df['volume'].pct_change()
    
    logger.info(f"Built features for {len(df)} rows")
    
    return df


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD."""
    ema_fast = _calculate_ema(prices, fast)
    ema_slow = _calculate_ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = _calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def _calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr


def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2) -> tuple:
    """Calculate Bollinger Bands."""
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper, middle, lower
