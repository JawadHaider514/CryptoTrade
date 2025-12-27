"""
Label generation for supervised learning.

Functions:
    create_labels() - Generate LONG/SHORT/NO_TRADE labels based on future returns
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class LabelConfig:
    """Configuration for label generation."""
    
    def __init__(self, horizon: int = 3, threshold: float = 0.0025):
        """
        Args:
            horizon: Number of candles into future to look (default 3)
            threshold: Return threshold for LONG/SHORT classification (default 0.25% = 0.0025)
        """
        self.horizon = horizon
        self.threshold = threshold


def create_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    threshold: float = 0.0025,
) -> Tuple[pd.DataFrame, dict]:
    """
    Create trading labels based on future returns.
    
    Args:
        df: DataFrame with 'close' column
        horizon: Number of future candles to look ahead
        threshold: Return threshold (0.0025 = 0.25%)
    
    Returns:
        Tuple of:
        - df: Original dataframe with added 'label' column (0=NO_TRADE, 1=LONG, -1=SHORT)
        - label_config: Dict with configuration used
    
    Logic:
        future_return = (close[t+horizon] / close[t]) - 1
        
        if future_return > +threshold  → label = 1 (LONG)
        if future_return < -threshold  → label = -1 (SHORT)
        else                            → label = 0 (NO_TRADE)
    """
    df = df.copy()
    
    # Calculate future returns (look ahead by horizon)
    future_price = df['close'].shift(-horizon)
    future_return = (future_price / df['close']) - 1
    
    # Create labels based on threshold
    df['label'] = 0  # Default NO_TRADE
    df.loc[future_return > threshold, 'label'] = 1  # LONG
    df.loc[future_return < -threshold, 'label'] = -1  # SHORT
    
    # Count labels
    label_counts = df['label'].value_counts().to_dict()
    
    label_config = {
        'horizon': horizon,
        'threshold': threshold,
        'threshold_pct': f"{threshold * 100:.2f}%",
        'label_distribution': {
            'LONG': label_counts.get(1, 0),
            'SHORT': label_counts.get(-1, 0),
            'NO_TRADE': label_counts.get(0, 0),
        }
    }
    
    logger.info(
        f"Created labels: LONG={label_counts.get(1, 0)} | "
        f"SHORT={label_counts.get(-1, 0)} | "
        f"NO_TRADE={label_counts.get(0, 0)}"
    )
    
    return df, label_config
