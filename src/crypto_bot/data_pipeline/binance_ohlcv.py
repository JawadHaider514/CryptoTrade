"""
Binance OHLCV data fetcher.

Functions:
    fetch_klines() - Fetch klines from Binance API with pagination
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"


def _to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Binance with pagination (forward iteration).
    
    Works forward from start_date, fetching 1000 candles per request,
    until reaching end_date or no more data available.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Candle interval ('15m', '1h', etc.)
        start: Start datetime (UTC). If None, uses last year.
        end: End datetime (UTC). If None, uses now.
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        timestamp is UTC open_time, sorted ascending.
    
    Raises:
        requests.RequestException: If HTTP request fails
    """
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(days=365)
    if end is None:
        end = datetime.now(timezone.utc)
    
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    
    all_rows = []
    cur = start_ms  # Start from earliest, iterate forward
    safety = 0
    request_count = 0
    
    logger.info(f"Fetching {symbol} {interval} from {start} to {end}")
    logger.info(f"  Start: {start_ms} ms")
    logger.info(f"  End:   {end_ms} ms")
    
    while cur < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000  # Max per Binance API
        }
        
        request_count += 1
        try:
            response = requests.get(
                f"{BINANCE_BASE_URL}/api/v3/klines",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            klines = response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        
        if not klines:
            logger.info(f"  Request {request_count}: No klines returned, stopping")
            break
        
        logger.info(f"  Request {request_count}: Got {len(klines)} candles")
        all_rows.extend(klines)
        
        # Use open_time (index 0) as anchor for next request
        last_open_time = klines[-1][0]  # open_time in ms
        next_cur = last_open_time + 1   # +1ms to avoid duplicates
        
        # Safety check: if next_cur didn't advance, stop
        if next_cur <= cur:
            logger.info(f"  Cursor didn't advance, stopping")
            break
        
        cur = next_cur
        
        # Stop if we got fewer than 1000 candles (reached the end)
        if len(klines) < 1000:
            logger.info(f"  Got {len(klines)} < 1000 candles, reached end of data")
            break
        
        # Rate limit friendly delay
        time.sleep(0.25)
        
        # Hard safety limit
        safety += 1
        if safety > 20000:
            logger.warning(f"Safety limit reached after {safety} iterations")
            break
    
    logger.info(f"Total requests: {request_count}, Total klines: {len(all_rows)}")
    
    if not all_rows:
        logger.warning(f"No klines fetched for {symbol} {interval}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
    )
    
    # Convert open_time to UTC datetime
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    
    # Keep only required columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    
    # Convert to numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Clean: drop NaN timestamps, sort, remove duplicates
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Fetched {len(df)} unique candles for {symbol} {interval}")
    logger.info(f"  Min timestamp: {df['timestamp'].min()}")
    logger.info(f"  Max timestamp: {df['timestamp'].max()}")
    
    return df
