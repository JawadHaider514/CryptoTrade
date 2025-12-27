"""
Parquet storage for OHLCV data.

Functions:
    save_parquet() - Save DataFrame to parquet
    load_parquet() - Load DataFrame from parquet
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to parquet file.
    
    Creates parent directories if they don't exist.
    
    Args:
        df: DataFrame with OHLCV data
        path: Full path to parquet file
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(path, index=False, compression='snappy')
    logger.info(f"Saved {len(df)} rows to {path}")


def load_parquet(path: str) -> pd.DataFrame:
    """
    Load DataFrame from parquet file.
    
    Args:
        path: Full path to parquet file
    
    Returns:
        DataFrame or empty DataFrame if file doesn't exist
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        logger.warning(f"Parquet file not found: {path}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    
    return df
