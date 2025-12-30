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
        - df: Original dataframe with added 'label' column (0=SHORT, 1=NO_TRADE, 2=LONG)
        - label_config: Dict with configuration used
    
    Logic:
        future_return = (close[t+horizon] / close[t]) - 1
        
        if future_return > +threshold  → label = 2 (LONG)
        if future_return < -threshold  → label = 0 (SHORT)
        else                            → label = 1 (NO_TRADE)
    """
    try:
        df = df.copy()
        
        # Validate input
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        
        # Calculate future returns (look ahead by horizon)
        future_price = df['close'].shift(-horizon)
        future_return = (future_price / df['close']) - 1
        
        # Create labels based on threshold
        df['label'] = 1  # Default NO_TRADE
        df.loc[future_return > threshold, 'label'] = 2  # LONG
        df.loc[future_return < -threshold, 'label'] = 0  # SHORT
        
        # Count labels
        label_counts = df['label'].value_counts().to_dict()
        
        label_config = {
            'horizon': horizon,
            'threshold': threshold,
            'threshold_pct': f"{threshold * 100:.2f}%",
            'label_mapping': {
                'SHORT': 0,
                'NO_TRADE': 1,
                'LONG': 2,
            },
            'label_distribution': {
                'SHORT': int(label_counts.get(0, 0)),
                'NO_TRADE': int(label_counts.get(1, 0)),
                'LONG': int(label_counts.get(2, 0)),
            }
        }
        
        logger.info(
            f"Created labels: SHORT={label_counts.get(0, 0)} | "
            f"NO_TRADE={label_counts.get(1, 0)} | "
            f"LONG={label_counts.get(2, 0)}"
        )
        
        return df, label_config
    
    except Exception as e:
        logger.error(f"Error creating labels: {e}")
        raise


__all__ = ['create_labels', 'LabelConfig']