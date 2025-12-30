"""Model Registry - Load and cache trained models with standard artifact structure."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Project root detection
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

MODELS_DIR = PROJECT_ROOT / "models"


class ModelRegistry:
    """Registry for managing trained CNN-LSTM models with per-coin artifact structure."""
    
    def __init__(self):
        """Initialize model registry."""
        self._models: Dict[Tuple[str, str], nn.Module] = {}
        self._scalers: Dict[Tuple[str, str], StandardScaler] = {}
        self._metadata: Dict[Tuple[str, str], dict] = {}
    
    def get_model_path(self, symbol: str, timeframe: str) -> Path:
        """Get path to model file following standard structure: models/per_coin/<symbol>/<timeframe>/"""
        symbol = symbol.upper()
        return MODELS_DIR / "per_coin" / symbol / timeframe / "cnn_lstm_v1.pth"
    
    def get_scaler_path(self, symbol: str, timeframe: str) -> Path:
        """Get path to scaler file."""
        symbol = symbol.upper()
        return MODELS_DIR / "per_coin" / symbol / timeframe / "scaler.pkl"
    
    def get_metadata_path(self, symbol: str, timeframe: str) -> Path:
        """Get path to metadata file."""
        symbol = symbol.upper()
        return MODELS_DIR / "per_coin" / symbol / timeframe / "meta.json"
    
    def load_model(
        self,
        symbol: str,
        timeframe: str,
        device: str = "cpu"
    ) -> Optional[nn.Module]:
        """
        Load a trained model from disk.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Timeframe (e.g., 15m)
            device: Device to load model on (cpu/cuda)
        
        Returns:
            Loaded model or None if not found
        """
        symbol = symbol.upper()
        key = (symbol, timeframe)
        
        # Return cached model if available
        if key in self._models:
            return self._models[key]
        
        model_path = self.get_model_path(symbol, timeframe)
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            # Import model class
            from crypto_bot.ml.train.train_cnn_lstm import CNNLSTM
            
            # Load model metadata to get actual feature count
            metadata = self.load_metadata(symbol, timeframe)
            num_features = 14  # default
            if metadata:
                num_features = metadata.get('dataset_info', {}).get('num_features', 14)
            
            # Create model instance with correct feature count
            model = CNNLSTM(
                num_features=num_features,
                lookback=60,
                num_classes=3
            ).to(device)
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Cache
            self._models[key] = model
            
            logger.info(f"✅ Loaded model: {symbol} {timeframe} (features={num_features})")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {symbol} {timeframe}: {e}")
            return None
    
    def load_scaler(self, symbol: str, timeframe: str) -> Optional[StandardScaler]:
        """
        Load feature scaler.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        
        Returns:
            StandardScaler or None if not found
        """
        symbol = symbol.upper()
        key = (symbol, timeframe)
        
        # Return cached scaler if available
        if key in self._scalers:
            return self._scalers[key]
        
        scaler_path = self.get_scaler_path(symbol, timeframe)
        
        if not scaler_path.exists():
            logger.warning(f"Scaler not found: {scaler_path}")
            return None
        
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            self._scalers[key] = scaler
            logger.info(f"✅ Loaded scaler: {symbol} {timeframe}")
            return scaler
            
        except Exception as e:
            logger.error(f"❌ Failed to load scaler {symbol} {timeframe}: {e}")
            return None
    
    def load_metadata(self, symbol: str, timeframe: str) -> Optional[dict]:
        """
        Load model metadata and metrics.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        
        Returns:
            Metadata dict or None if not found
        """
        symbol = symbol.upper()
        key = (symbol, timeframe)
        
        # Return cached metadata if available
        if key in self._metadata:
            return self._metadata[key]
        
        meta_path = self.get_metadata_path(symbol, timeframe)
        
        if not meta_path.exists():
            logger.warning(f"Metadata not found: {meta_path}")
            return None
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            self._metadata[key] = metadata
            logger.info(f"✅ Loaded metadata: {symbol} {timeframe}")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load metadata {symbol} {timeframe}: {e}")
            return None
    
    def get_model_accuracy(self, symbol: str, timeframe: str) -> Optional[float]:
        """
        Get model test accuracy.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        
        Returns:
            Test accuracy or None
        """
        metadata = self.load_metadata(symbol, timeframe)
        if metadata and 'test_metrics' in metadata:
            return metadata['test_metrics'].get('accuracy')
        return None
    
    def is_model_available(self, symbol: str, timeframe: str) -> bool:
        """Check if model is available."""
        symbol = symbol.upper()
        model_path = self.get_model_path(symbol, timeframe)
        return model_path.exists()
    
    def get_model(
        self,
        symbol: str,
        timeframe: str,
        device: str = "cpu"
    ) -> Tuple[Optional[nn.Module], Optional[StandardScaler], Optional[dict]]:
        """
        Get complete model bundle (model, scaler, metadata).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            device: Device for inference
        
        Returns:
            Tuple of (model, scaler, metadata) - any can be None if not found
        """
        return (
            self.load_model(symbol, timeframe, device),
            self.load_scaler(symbol, timeframe),
            self.load_metadata(symbol, timeframe)
        )
    
    def clear_cache(self):
        """Clear cached models and scalers."""
        self._models.clear()
        self._scalers.clear()
        self._metadata.clear()
        logger.info("🧹 Model registry cache cleared")


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
