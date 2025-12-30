"""Inference Service - Make predictions with trained models."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from crypto_bot.ml.inference.model_registry import get_registry
from crypto_bot.ml.inference.accuracy_calibrator import get_calibrator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"

# Label mappings
LABEL_TO_DIRECTION = {0: "SHORT", 1: "NO_TRADE", 2: "LONG"}
DIRECTION_TO_LABEL = {v: k for k, v in LABEL_TO_DIRECTION.items()}


@dataclass
class PredictionResult:
    """Prediction result with metadata - API Contract Frozen."""
    symbol: str
    timeframe: str
    direction: str  # LONG, SHORT, NO_TRADE
    confidence: float  # 0-100 confidence percentage
    accuracy_estimate: float  # 0-100 model accuracy percentage
    current_price: float  # Current market price
    entry_price: float  # Suggested entry price
    stop_loss: float  # Stop loss price
    take_profits: list  # [{level: 1, price: X, eta: "ISO"}]
    timestamp: str  # ISO timestamp
    valid_until: str  # When this prediction expires
    
    # Internal fields (not exposed in API)
    probabilities: Optional[Dict[str, float]] = None  # {LONG: 0.6, SHORT: 0.3, NO_TRADE: 0.1}
    model_accuracy: Optional[float] = None  # Raw model accuracy 0-1
    
    def to_dict(self) -> dict:
        """Convert to API contract dictionary."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction,
            'confidence': float(self.confidence),
            'accuracy_estimate': float(self.accuracy_estimate),
            'current_price': float(self.current_price),
            'entry_price': float(self.entry_price),
            'stop_loss': float(self.stop_loss),
            'take_profits': self.take_profits,
            'timestamp': self.timestamp,
            'valid_until': self.valid_until,
        }


class InferenceService:
    """Service for making predictions with trained models."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize inference service.
        
        Args:
            device: Device to run inference on (cpu/cuda)
        """
        self.device = device
        self.registry = get_registry()
    
    def predict(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 60
    ) -> Optional[PredictionResult]:
        """
        Make prediction for a symbol/timeframe.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Candle interval (e.g., 15m)
            lookback: Number of candles to use for prediction
        
        Returns:
            PredictionResult or None if prediction fails
        """
        symbol = symbol.upper()
        
        try:
            # Load model and scaler
            model = self.registry.load_model(symbol, timeframe, device=self.device)
            scaler = self.registry.load_scaler(symbol, timeframe)
            
            if model is None or scaler is None:
                logger.warning(f"Model or scaler not available: {symbol} {timeframe}")
                return None
            
            # Load latest features and price data
            features = self._load_latest_features(symbol, timeframe, lookback)
            if features is None:
                logger.warning(f"Could not load features: {symbol} {timeframe}")
                return None
            
            # Get current price
            current_price = self._get_current_price(symbol, timeframe)
            if current_price is None:
                logger.warning(f"Could not get current price: {symbol}")
                return None
            
            # Make prediction
            with torch.no_grad():
                X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                logits = model(X)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                predicted_class = int(np.argmax(probabilities))
            
            direction = LABEL_TO_DIRECTION[predicted_class]
            confidence_raw = float(probabilities[predicted_class])
            confidence_pct = confidence_raw * 100  # Convert to 0-100
            
            # Get model accuracy from calibrator (rolling actual accuracy)
            calibrator = get_calibrator()
            accuracy_estimate = calibrator.get_accuracy(symbol, timeframe)
            
            # Calculate price levels based on direction
            entry_price, stop_loss, take_profits = self._calculate_price_levels(
                current_price, direction, timeframe
            )
            
            # Create prediction result
            now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
            valid_until = self._calculate_valid_until(timeframe)
            
            return PredictionResult(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                confidence=confidence_pct,
                accuracy_estimate=accuracy_estimate,
                current_price=current_price,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profits=take_profits,
                timestamp=now,
                valid_until=valid_until,
                probabilities={
                    'LONG': float(probabilities[2]),
                    'NO_TRADE': float(probabilities[1]),
                    'SHORT': float(probabilities[0]),
                },
                model_accuracy=None,
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol} {timeframe}: {e}", exc_info=True)
            return None
    
    def predict_batch(
        self,
        symbols: list,
        timeframe: str,
        lookback: int = 60
    ) -> Dict[str, Optional[PredictionResult]]:
        """
        Make predictions for multiple symbols.
        
        Args:
            symbols: List of symbols
            timeframe: Candle interval
            lookback: Number of candles
        
        Returns:
            Dict mapping symbol to PredictionResult
        """
        results = {}
        for symbol in symbols:
            results[symbol.upper()] = self.predict(symbol, timeframe, lookback)
        return results
    
    def _load_latest_features(
        self,
        symbol: str,
        timeframe: str,
        lookback: int
    ) -> Optional[np.ndarray]:
        """
        Load latest features from dataset.
        
        Args:
            symbol: Trading symbol
            timeframe: Candle interval
            lookback: Number of candles
        
        Returns:
            Feature array of shape (lookback, num_features) or None
        """
        dataset_path = DATASET_DIR / symbol / f"{timeframe}_dataset.parquet"
        
        if not dataset_path.exists():
            return None
        
        try:
            df = pd.read_parquet(dataset_path)
            
            # Get feature columns (exclude timestamp, OHLCV, and label)
            feature_cols = [c for c in df.columns if c not in [
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'label'
            ]]
            
            # Get last lookback rows
            features = df[feature_cols].tail(lookback).values.astype(np.float32)
            
            # Ensure we have exactly lookback rows
            if len(features) < lookback:
                logger.warning(f"Not enough data: got {len(features)}, need {lookback}")
                return None
            
            # Load scaler and normalize
            scaler = self.registry.load_scaler(symbol, timeframe)
            if scaler:
                features = scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return None
    
    def _calculate_valid_until(self, timeframe: str) -> str:
        """
        Calculate when prediction expires based on timeframe.
        
        Args:
            timeframe: Candle interval
        
        Returns:
            ISO 8601 timestamp string with timezone (e.g., 2025-12-28T09:15:51Z)
        """
        from datetime import timedelta
        
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
        }
        
        minutes = timeframe_minutes.get(timeframe, 15)
        # Prediction valid for 2x the candle period
        valid_duration = timedelta(minutes=minutes * 2)
        
        valid_until = datetime.now(timezone.utc) + valid_duration
        # Return ISO format with Z suffix for UTC
        return valid_until.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    
    def _get_current_price(self, symbol: str, timeframe: str) -> Optional[float]:
        """
        Get current price from latest dataset.
        
        Args:
            symbol: Trading symbol
            timeframe: Candle interval
        
        Returns:
            Current price or None
        """
        dataset_path = DATASET_DIR / symbol / f"{timeframe}_dataset.parquet"
        
        if not dataset_path.exists():
            return None
        
        try:
            df = pd.read_parquet(dataset_path)
            if 'close' in df.columns:
                return float(df['close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    def _calculate_price_levels(
        self,
        current_price: float,
        direction: str,
        timeframe: str
    ) -> Tuple[float, float, list]:
        """
        Calculate entry, stop loss, and take profit levels.
        
        Args:
            current_price: Current market price
            direction: Prediction direction (LONG/SHORT/NO_TRADE)
            timeframe: Candle interval
        
        Returns:
            Tuple of (entry_price, stop_loss, take_profits_list)
        """
        # Risk/reward percentages based on timeframe
        timeframe_config = {
            '1m': {'risk': 0.003, 'tp': [0.007, 0.015, 0.025]},
            '5m': {'risk': 0.005, 'tp': [0.010, 0.020, 0.035]},
            '15m': {'risk': 0.008, 'tp': [0.015, 0.030, 0.050]},
            '1h': {'risk': 0.010, 'tp': [0.020, 0.040, 0.065]},
            '4h': {'risk': 0.015, 'tp': [0.030, 0.060, 0.100]},
            '1d': {'risk': 0.020, 'tp': [0.050, 0.100, 0.150]},
        }
        
        config = timeframe_config.get(timeframe, timeframe_config['15m'])
        
        if direction == "NO_TRADE":
            # No trade signal - return current price for all levels
            return current_price, current_price, []
        
        elif direction == "LONG":
            # Buy signal
            entry_price = current_price * 1.002  # Slightly above current
            stop_loss = entry_price * (1 - config['risk'])
            
            take_profits = []
            for level, tp_pct in enumerate(config['tp'], start=1):
                tp_price = entry_price * (1 + tp_pct)
                # TP expires at 3x the timeframe
                timeframe_minutes = {
                    '1m': 1, '5m': 5, '15m': 15, '1h': 60,
                    '4h': 240, '1d': 1440
                }
                minutes = timeframe_minutes.get(timeframe, 15)
                from datetime import timedelta
                eta_dt = datetime.now(timezone.utc) + timedelta(minutes=minutes * 3)
                eta = eta_dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
                
                take_profits.append({
                    'level': level,
                    'price': float(round(tp_price, 8)),
                    'eta': eta
                })
        
        else:  # SHORT
            # Sell signal
            entry_price = current_price * 0.998  # Slightly below current
            stop_loss = entry_price * (1 + config['risk'])
            
            take_profits = []
            for level, tp_pct in enumerate(config['tp'], start=1):
                tp_price = entry_price * (1 - tp_pct)
                timeframe_minutes = {
                    '1m': 1, '5m': 5, '15m': 15, '1h': 60,
                    '4h': 240, '1d': 1440
                }
                minutes = timeframe_minutes.get(timeframe, 15)
                from datetime import timedelta
                eta_dt = datetime.now(timezone.utc) + timedelta(minutes=minutes * 3)
                eta = eta_dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
                
                take_profits.append({
                    'level': level,
                    'price': float(round(tp_price, 8)),
                    'eta': eta
                })
        
        return (
            float(round(entry_price, 8)),
            float(round(stop_loss, 8)),
            take_profits
        )


def make_prediction(
    symbol: str,
    timeframe: str = "15m",
    device: str = "cpu"
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to make a prediction.
    
    Args:
        symbol: Trading symbol
        timeframe: Candle interval
        device: Device (cpu/cuda)
    
    Returns:
        Prediction dict or None
    """
    service = InferenceService(device=device)
    result = service.predict(symbol, timeframe)
    return result.to_dict() if result else None
