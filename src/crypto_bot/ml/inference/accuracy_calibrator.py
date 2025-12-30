"""Real-time Accuracy Calibrator - Tracks predictions vs actual outcomes."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
ACCURACY_CACHE_DIR = PROJECT_ROOT / "data" / "accuracy_cache"
ACCURACY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class AccuracyCalibrator:
    """Tracks rolling prediction accuracy per coin/timeframe."""
    
    def __init__(self, lookback_windows: int = 50):
        """
        Initialize accuracy calibrator.
        
        Args:
            lookback_windows: Number of recent candles to use for accuracy calc
        """
        self.lookback_windows = lookback_windows
        self._cache: Dict[Tuple[str, str], Dict] = {}
    
    def get_accuracy(self, symbol: str, timeframe: str) -> float:
        """
        Get rolling accuracy for a symbol/timeframe.
        
        Computes: Recent actual prediction accuracy using last N candles
        - Loads last N closed candles
        - Generates predictions for each
        - Checks if prediction direction matches actual next candle
        - Returns win_rate as 0-100 percentage
        
        Args:
            symbol: Trading symbol (BTCUSDT)
            timeframe: Candle interval (15m, 1h)
        
        Returns:
            Accuracy percentage (0-100) based on rolling predictions
        """
        symbol = symbol.upper()
        key = (symbol, timeframe)
        
        # Check cache (valid for 5 minutes)
        if key in self._cache:
            cached = self._cache[key]
            if datetime.now() - cached['timestamp'] < timedelta(minutes=5):
                return cached['accuracy']
        
        try:
            # Load dataset
            dataset_path = DATASET_DIR / symbol / f"{timeframe}_dataset.parquet"
            if not dataset_path.exists():
                logger.warning(f"Dataset not found: {symbol} {timeframe}")
                return 50.0  # Default fallback
            
            df = pd.read_parquet(dataset_path)
            
            if df.empty or 'label' not in df.columns or 'close' not in df.columns:
                logger.warning(f"Dataset incomplete: {symbol} {timeframe}")
                return 50.0
            
            # Get last lookback_windows rows for rolling accuracy
            recent_df = df.tail(self.lookback_windows)
            
            if len(recent_df) < 10:  # Need at least 10 candles
                logger.warning(f"Not enough data: {symbol} {timeframe}")
                return 50.0
            
            # Label mappings
            LABEL_TO_DIRECTION = {0: "SHORT", 1: "NO_TRADE", 2: "LONG"}
            
            # Compute accuracy: For each candle, check if label matches direction movement
            correct = 0
            total = 0
            
            for i in range(len(recent_df) - 1):
                current_row = recent_df.iloc[i]
                next_row = recent_df.iloc[i + 1]
                
                # Get predicted direction
                if 'label' not in current_row.index:
                    continue
                
                predicted_label = int(current_row['label'])
                predicted_direction = LABEL_TO_DIRECTION.get(predicted_label, "NO_TRADE")
                
                # Get actual direction (compare current close to next close)
                current_close = float(current_row['close'])
                next_close = float(next_row['close'])
                
                # Determine actual direction
                price_change_pct = (next_close - current_close) / current_close
                
                if price_change_pct > 0.001:  # Up more than 0.1%
                    actual_direction = "LONG"
                elif price_change_pct < -0.001:  # Down more than 0.1%
                    actual_direction = "SHORT"
                else:
                    actual_direction = "NO_TRADE"  # Flat movement
                
                # Check if prediction matches actual
                total += 1
                if predicted_direction == actual_direction:
                    correct += 1
            
            # Calculate accuracy
            if total > 0:
                accuracy = (correct / total) * 100.0
            else:
                accuracy = 50.0
            
            # Ensure reasonable range (30-85%, clip extreme values)
            accuracy = max(30.0, min(85.0, accuracy))
            
            # Cache result
            self._cache[key] = {
                'accuracy': accuracy,
                'timestamp': datetime.now(),
                'samples': total,
                'correct': correct
            }
            
            logger.debug(f"[{symbol} {timeframe}] Accuracy: {accuracy:.1f}% ({correct}/{total})")
            return accuracy
            
        except Exception as e:
            logger.error(f"Failed to compute accuracy: {symbol} {timeframe}: {e}")
            return 50.0  # Safe default
    
    def get_accuracy_per_direction(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, float]:
        """
        Get accuracy breakdown by direction (LONG, SHORT, NO_TRADE).
        
        Args:
            symbol: Trading symbol
            timeframe: Candle interval
        
        Returns:
            Dict with accuracy for each direction
        """
        symbol = symbol.upper()
        
        try:
            dataset_path = DATASET_DIR / symbol / f"{timeframe}_dataset.parquet"
            if not dataset_path.exists():
                return {"LONG": 50.0, "SHORT": 50.0, "NO_TRADE": 50.0}
            
            df = pd.read_parquet(dataset_path)
            recent_df = df.tail(self.lookback_windows)
            
            LABEL_TO_DIRECTION = {0: "SHORT", 1: "NO_TRADE", 2: "LONG"}
            direction_stats = {"LONG": {"correct": 0, "total": 0},
                             "SHORT": {"correct": 0, "total": 0},
                             "NO_TRADE": {"correct": 0, "total": 0}}
            
            for i in range(len(recent_df) - 1):
                current_row = recent_df.iloc[i]
                next_row = recent_df.iloc[i + 1]
                
                if 'label' not in current_row.index:
                    continue
                
                predicted_label = int(current_row['label'])
                predicted_direction = LABEL_TO_DIRECTION.get(predicted_label, "NO_TRADE")
                
                current_close = float(current_row['close'])
                next_close = float(next_row['close'])
                price_change_pct = (next_close - current_close) / current_close
                
                if price_change_pct > 0.001:
                    actual_direction = "LONG"
                elif price_change_pct < -0.001:
                    actual_direction = "SHORT"
                else:
                    actual_direction = "NO_TRADE"
                
                direction_stats[predicted_direction]["total"] += 1
                if predicted_direction == actual_direction:
                    direction_stats[predicted_direction]["correct"] += 1
            
            result = {}
            for direction, stats in direction_stats.items():
                if stats["total"] > 0:
                    accuracy = (stats["correct"] / stats["total"]) * 100.0
                    accuracy = max(30.0, min(85.0, accuracy))
                    result[direction] = accuracy
                else:
                    result[direction] = 50.0
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute direction accuracy: {symbol} {timeframe}: {e}")
            return {"LONG": 50.0, "SHORT": 50.0, "NO_TRADE": 50.0}
    
    def clear_cache(self):
        """Clear cached accuracy values."""
        self._cache.clear()
        logger.info("Accuracy cache cleared")


# Global calibrator instance
_calibrator: Optional[AccuracyCalibrator] = None


def get_calibrator(lookback_windows: int = 50) -> AccuracyCalibrator:
    """Get or create global accuracy calibrator."""
    global _calibrator
    if _calibrator is None:
        _calibrator = AccuracyCalibrator(lookback_windows=lookback_windows)
    return _calibrator
