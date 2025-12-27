"""
Binance OHLCV data fetcher.

Functions:
    fetch_klines() - Fetch klines from Binance API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"
MAX_KLINES_PER_REQUEST = 1000
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds
REQUEST_TIMEOUT = 10  # seconds


def fetch_klines(
    symbol: str,
    interval: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch klines from Binance API.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Candle interval ('15m', '1h', etc.)
        start: Start datetime (UTC). If None, uses last year.
        end: End datetime (UTC). If None, uses now.
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        All timestamps are UTC datetime objects, sorted ascending.
    
    Raises:
        Exception: If all retry attempts fail
    """
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(days=365)
    if end is None:
        end = datetime.now(timezone.utc)
    
    # Convert to milliseconds for Binance API
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    all_klines = []
    current_start_ms = start_ms
    
    logger.info(f"Fetching {symbol} {interval} from {start} to {end}")
    
    while current_start_ms < end_ms:
        klines_chunk = _fetch_klines_chunk(
            symbol=symbol,
            interval=interval,
            start_time=current_start_ms,
            end_time=end_ms,
        )
        
        if not klines_chunk:
            break
        
        all_klines.extend(klines_chunk)
        
        # Update start time for next chunk (avoid duplicates)
        last_close_time = klines_chunk[-1][6]  # Close time is index 6
        current_start_ms = last_close_time + 1
        
        # Small delay to respect rate limits
        time.sleep(0.1)
    
    if not all_klines:
        logger.warning(f"No klines fetched for {symbol} {interval}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_klines,
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
    )
    
    # Keep only required columns
    df = df[['close_time', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert timestamp to UTC datetime
    df['timestamp'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    df = df.drop('close_time', axis=1)
    
    # Reorder columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert price/volume columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Fetched {len(df)} candles for {symbol} {interval}")
    
    return df


def _fetch_klines_chunk(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
) -> list:
    """
    Fetch a chunk of klines with retry logic.
    
    Args:
        symbol: Trading symbol
        interval: Candle interval
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
    
    Returns:
        List of kline data or empty list if all retries fail
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": MAX_KLINES_PER_REQUEST,
    }
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed for {symbol} {interval}: {e}"
            )
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"Failed to fetch {symbol} {interval} after {RETRY_ATTEMPTS} attempts")
                return []
    
    return []
