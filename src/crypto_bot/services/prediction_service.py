#!/usr/bin/env python3
"""
ML-Based Prediction Service
Handles per-coin ML model inference with feature extraction and confidence scoring.

Format:
{
  "symbol": "BTCUSDT",
  "tf": "15m",
  "pred": "LONG|SHORT|NO_TRADE",
  "confidence": 0.73,
  "source": "ML_PER_COIN_V1",
  "model_version": "cnn_lstm_v1",
  "ts": "ISO datetime"
}
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import torch

from crypto_bot.ml.inference.model_registry import get_registry
from crypto_bot.repositories.market_history import get_market_history

logger = logging.getLogger(__name__)


class PredictionService:
    """ML-based prediction service with per-coin models."""
    
    def __init__(self, device: str = "cpu", min_confidence: float = 0.5):
        """
        Initialize prediction service.
        
        Args:
            device: Device for inference ("cpu" or "cuda")
            min_confidence: Minimum confidence threshold for predictions
        """
        self.registry = get_registry()
        self.device = device
        self.min_confidence = min_confidence
        self.market_history = get_market_history()
        
        logger.info(f"🚀 PredictionService initialized (device={device})")
    
    def predict_symbol(self, symbol: str, timeframe: str = "15m") -> Optional[Dict]:
        """
        Get ML prediction for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Timeframe (e.g., 15m, 1h)
        
        Returns:
            Prediction dict or None if model unavailable
        """
        symbol = symbol.upper()
        
        # Check model availability
        if not self.registry.is_model_available(symbol, timeframe):
            logger.debug(f"FALLBACK {symbol} reason=model_missing")
            return None
        
        try:
            # Load model, scaler, metadata
            model, scaler, metadata = self.registry.get_model(
                symbol, 
                timeframe,
                device=self.device
            )
            
            if model is None or scaler is None:
                logger.debug(f"FALLBACK {symbol} reason=model_missing")
                return None
            
            # Get latest candles (lookback=60)
            candles = self._get_candles(symbol, timeframe, lookback=60)
            if candles is None or len(candles) < 60:
                logger.debug(f"FALLBACK {symbol} reason=insufficient_data")
                return None
            
            # Extract features
            features = self._extract_features(candles)
            if features is None:
                logger.debug(f"FALLBACK {symbol} reason=feature_extraction_failed")
                return None
            
            # Scale features
            try:
                features_scaled = scaler.transform(features.reshape(1, -1))
            except Exception as e:
                logger.error(f"FALLBACK {symbol} reason=scaler_error: {e}")
                return None
            
            # Reshape for model (batch_size=1, lookback=60, features)
            lookback = 60
            num_features = features_scaled.shape[1]
            
            # Create lookback window from latest candles
            try:
                close_prices = np.array([c["close"] for c in candles[-lookback:]])
                features_array = self._create_lookback_features(close_prices, num_features)
                
                if features_array is None:
                    logger.debug(f"FALLBACK {symbol} reason=lookback_feature_error")
                    return None
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(self.device)
                
                # Model inference
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                # Classes: 0=SHORT, 1=NO_TRADE, 2=LONG
                pred_class = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_class])
                
                # Map to direction
                pred_map = {0: "SHORT", 1: "NO_TRADE", 2: "LONG"}
                direction = pred_map[pred_class]
                
                # Check confidence threshold
                if confidence < self.min_confidence:
                    logger.debug(f"FALLBACK {symbol} reason=low_confidence conf={confidence:.2f}")
                    return None
                
                # Build response
                result = {
                    "symbol": symbol,
                    "tf": timeframe,
                    "pred": direction,
                    "confidence": confidence,
                    "source": "ML_PER_COIN_V1",
                    "model_version": "cnn_lstm_v1",
                    "ts": datetime.utcnow().isoformat() + "Z"
                }
                
                logger.info(f"PREDICT {symbol} tf={timeframe} pred={direction} conf={confidence:.2f} source=ML_PER_COIN_V1")
                return result
            
            except Exception as e:
                logger.error(f"FALLBACK {symbol} reason=inference_error: {e}")
                return None
        
        except Exception as e:
            logger.error(f"FALLBACK {symbol} reason=prediction_service_error: {e}")
            return None
    
    def predict_batch(self, symbols: list, timeframe: str = "15m") -> Dict[str, Optional[Dict]]:
        """
        Get predictions for multiple symbols.
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe for all predictions
        
        Returns:
            Dict mapping symbol -> prediction (or None)
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.predict_symbol(symbol, timeframe)
        return results
    
    def _get_candles(self, symbol: str, timeframe: str, lookback: int = 60) -> Optional[list]:
        """Get latest candles from market history."""
        try:
            candles = self.market_history.get_recent_candles(symbol, timeframe, limit=lookback)
            if candles:
                return sorted(candles, key=lambda c: c["time"])
            return None
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return None
    
    def _extract_features(self, candles: list) -> Optional[np.ndarray]:
        """
        Extract technical features from candles matching training data.
        
        Features (15):
        - atr_14, bb_lower, bb_middle, bb_upper, bb_width
        - ema_20, ema_50
        - log_returns
        - macd, macd_diff, macd_signal
        - returns
        - rsi_14
        - volatility
        - volume_change
        
        Returns:
            Array of shape (num_features,) or None
        """
        try:
            closes = np.array([c["close"] for c in candles])
            highs = np.array([c["high"] for c in candles])
            lows = np.array([c["low"] for c in candles])
            volumes = np.array([c["volume"] for c in candles])
            
            features = []
            
            # 1. ATR (14-period)
            atr_14 = self._calculate_atr(highs, lows, closes, 14)
            features.append(atr_14)
            
            # 2-4. Bollinger Bands (20, 2)
            bb_upper, bb_lower = self._calculate_bollinger(closes, 20, 2)
            bb_middle = np.mean(closes[-20:])
            features.append(bb_lower)
            features.append(bb_middle)
            features.append(bb_upper)
            
            # 5. BB Width
            bb_width = bb_upper - bb_lower
            features.append(bb_width)
            
            # 6-7. EMA (20, 50)
            ema_20 = self._calculate_ema(closes, 20)
            ema_50 = self._calculate_ema(closes, 50)
            features.extend([ema_20, ema_50])
            
            # 8. Log Returns
            log_returns = np.log(closes[-1] / closes[-2]) if len(closes) > 1 else 0
            features.append(log_returns)
            
            # 9-11. MACD (with signal and diff)
            ema_12 = self._calculate_ema(closes, 12)
            ema_26 = self._calculate_ema(closes, 26)
            macd_line = ema_12 - ema_26
            features.append(macd_line)
            
            # MACD Signal (9-period EMA of MACD)
            # Using simple approximation for single value
            macd_signal = macd_line * 0.11 + macd_line * 0.89  # Simplified
            features.append(macd_signal)
            
            # MACD Diff
            macd_diff = macd_line - macd_signal
            features.append(macd_diff)
            
            # 12. Returns
            returns = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
            features.append(returns)
            
            # 13. RSI (14-period)
            rsi_14 = self._calculate_rsi(closes, 14)
            features.append(rsi_14)
            
            # 14. Volatility (standard deviation of log returns)
            log_ret_array = np.log(closes[1:] / closes[:-1])
            volatility = np.std(log_ret_array) if len(log_ret_array) > 0 else 0
            features.append(volatility)
            
            # 15. Volume Change
            vol_change = (volumes[-1] - np.mean(volumes[-20:])) / np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
            features.append(vol_change)
            
            return np.array(features, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def _create_lookback_features(self, close_prices: np.ndarray, num_features: int) -> Optional[np.ndarray]:
        """Create lookback window for LSTM input."""
        try:
            if len(close_prices) < 60:
                return None
            
            # Use last 60 closes + additional features
            lookback_features = close_prices[-60:].reshape(-1, 1)
            
            # Pad to match expected feature count if needed
            if num_features > 1:
                padding = np.zeros((60, num_features - 1))
                lookback_features = np.hstack([lookback_features, padding])
            
            return lookback_features.astype(np.float32)
        except Exception as e:
            logger.error(f"Lookback feature error: {e}")
            return None
    
    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        ema = prices[0]
        multiplier = 2 / (period + 1)
        for price in prices[1:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return float(ema)
    
    @staticmethod
    def _calculate_sma(prices: np.ndarray, period: int) -> float:
        """Calculate SMA."""
        if len(prices) < period:
            return float(np.mean(prices))
        return float(np.mean(prices[-period:]))
    
    @staticmethod
    def _calculate_macd(prices: np.ndarray) -> float:
        """Calculate MACD line."""
        ema12 = PredictionService._calculate_ema(prices, 12)
        ema26 = PredictionService._calculate_ema(prices, 26)
        return float(ema12 - ema26)
    
    @staticmethod
    def _calculate_bollinger(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return float(upper), float(lower)
    
    @staticmethod
    def _calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR."""
        tr_values = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
            tr_values.append(tr)
        
        if len(tr_values) < period:
            return float(np.mean(tr_values))
        return float(np.mean(tr_values[-period:]))
