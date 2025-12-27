#!/usr/bin/env python3
"""
Prediction Mapper / Adapter
==========================
Normalizes different analyzer outputs into one stable contract.

Supported inputs:
- SignalModel (from signal_engine_service)
- ProfessionalAnalyzer output
- Dashboard analyzer output
- Raw dictionary analysis

Output: Unified prediction schema with ISO datetime format
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class PredictionMapper:
    """Normalizes different prediction formats into stable schema"""
    
    @staticmethod
    def from_signal_model(signal_model) -> Dict[str, Any]:
        """
        Convert SignalModel to unified prediction schema
        
        Args:
            signal_model: SignalModel instance from signal_engine_service
        
        Returns:
            Unified prediction dict with ISO datetimes
        """
        try:
            # Extract take profits
            take_profits = []
            if hasattr(signal_model, 'take_profits') and signal_model.take_profits:
                for tp in signal_model.take_profits:
                    take_profits.append({
                        "level": tp.level,
                        "price": float(tp.price),
                        "eta": PredictionMapper._to_iso_string(tp.eta)
                    })
            
            # Build unified schema
            prediction = {
                "symbol": signal_model.symbol,
                "timeframe": signal_model.timeframe,
                "source": getattr(signal_model, 'source', 'UNKNOWN'),
                "direction": signal_model.direction,
                "entry_price": float(signal_model.entry_price),
                "stop_loss": float(signal_model.stop_loss),
                "take_profits": take_profits,
                "confidence_score": int(signal_model.confidence_score),
                "accuracy_percent": float(signal_model.accuracy_percent),
                "leverage": int(signal_model.leverage),
                "current_price": float(signal_model.current_price),
                "timestamp": PredictionMapper._to_iso_string(signal_model.timestamp),
                "valid_until": PredictionMapper._to_iso_string(signal_model.valid_until),
                "reasons": signal_model.reasons or [],
                "patterns": signal_model.patterns or [],
                "market_context": signal_model.market_context or ""
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error mapping SignalModel: {e}")
            raise
    
    @staticmethod
    def from_pro_analyzer(setup: Dict[str, Any], symbol: str, timeframe: str = "15m") -> Dict[str, Any]:
        """
        Convert ProfessionalAnalyzer output to unified schema
        
        Args:
            setup: Output from ProfessionalAnalyzer.analyze_complete_setup()
            symbol: Trading symbol
            timeframe: Timeframe (default: 15m)
        
        Returns:
            Unified prediction dict with ISO datetimes
        """
        try:
            if not setup:
                raise ValueError("Setup is None or empty")
            
            # Extract take profits
            take_profits = []
            tp_data = setup.get('take_profits', [])
            if tp_data:
                for i, tp_price in enumerate(tp_data, 1):
                    eta = setup.get(f'tp{i}_eta', datetime.utcnow())
                    take_profits.append({
                        "level": i,
                        "price": float(tp_price),
                        "eta": PredictionMapper._to_iso_string(eta)
                    })
            
            # Build unified schema
            prediction = {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "PRO",
                "direction": setup.get('direction', 'LONG'),
                "entry_price": float(setup.get('entry_price', 0)),
                "stop_loss": float(setup.get('stop_loss', 0)),
                "take_profits": take_profits,
                "confidence_score": int(setup.get('confidence', 50)),
                "accuracy_percent": float(setup.get('accuracy_percent', 0.0)),
                "leverage": int(setup.get('leverage', 1)),
                "current_price": float(setup.get('current_price', setup.get('entry_price', 0))),
                "timestamp": PredictionMapper._to_iso_string(setup.get('timestamp', datetime.utcnow())),
                "valid_until": PredictionMapper._to_iso_string(setup.get('valid_until', datetime.utcnow())),
                "reasons": setup.get('reasons', []),
                "patterns": setup.get('patterns', []),
                "market_context": setup.get('market_context', '')
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error mapping ProfessionalAnalyzer output: {e}")
            raise
    
    @staticmethod
    def from_dashboard(analysis: Dict[str, Any], symbol: str, current_price: float, timeframe: str = "15m") -> Dict[str, Any]:
        """
        Convert Dashboard analyzer output to unified schema
        
        Args:
            analysis: Output from dashboard signal_analyzer or streaming_processor
            symbol: Trading symbol
            current_price: Current market price
            timeframe: Timeframe (default: 15m)
        
        Returns:
            Unified prediction dict with ISO datetimes
        """
        try:
            if not analysis:
                raise ValueError("Analysis is None or empty")
            
            # Extract take profits
            take_profits = []
            tp_data = analysis.get('take_profits', [])
            if tp_data:
                for i, tp_price in enumerate(tp_data, 1):
                    eta = analysis.get(f'tp{i}_eta', datetime.utcnow())
                    take_profits.append({
                        "level": i,
                        "price": float(tp_price),
                        "eta": PredictionMapper._to_iso_string(eta)
                    })
            
            # Build unified schema
            prediction = {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "DASHBOARD",
                "direction": analysis.get('direction', 'LONG'),
                "entry_price": float(analysis.get('entry', current_price)),
                "stop_loss": float(analysis.get('sl', current_price * 0.95)),
                "take_profits": take_profits,
                "confidence_score": int(analysis.get('confidence_score', 50)),
                "accuracy_percent": float(analysis.get('accuracy_percent', 0.0)),
                "leverage": int(analysis.get('leverage', 1)),
                "current_price": float(current_price),
                "timestamp": PredictionMapper._to_iso_string(analysis.get('timestamp', datetime.utcnow())),
                "valid_until": PredictionMapper._to_iso_string(analysis.get('valid_until', datetime.utcnow())),
                "reasons": analysis.get('reasons', []),
                "patterns": analysis.get('patterns', []),
                "market_context": analysis.get('market_context', '')
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error mapping Dashboard output: {e}")
            raise
    
    @staticmethod
    def from_fallback(symbol: str, current_price: float, timeframe: str = "15m") -> Dict[str, Any]:
        """
        Create minimal fallback prediction
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            timeframe: Timeframe (default: 15m)
        
        Returns:
            Unified prediction dict with minimal values
        """
        try:
            now = datetime.utcnow()
            
            # Calculate minimal TPs
            distance = current_price * 0.01
            tp_prices = [
                current_price + (distance * 0.5),
                current_price + (distance * 1.0),
                current_price + (distance * 1.5),
            ]
            
            take_profits = [
                {
                    "level": 1,
                    "price": float(tp_prices[0]),
                    "eta": PredictionMapper._to_iso_string(datetime.utcnow())
                },
                {
                    "level": 2,
                    "price": float(tp_prices[1]),
                    "eta": PredictionMapper._to_iso_string(datetime.utcnow())
                },
                {
                    "level": 3,
                    "price": float(tp_prices[2]),
                    "eta": PredictionMapper._to_iso_string(datetime.utcnow())
                }
            ]
            
            prediction = {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "FALLBACK",
                "direction": "LONG",
                "entry_price": float(current_price),
                "stop_loss": float(current_price * 0.98),
                "take_profits": take_profits,
                "confidence_score": 15,
                "accuracy_percent": 0.0,
                "leverage": 1,
                "current_price": float(current_price),
                "timestamp": PredictionMapper._to_iso_string(now),
                "valid_until": PredictionMapper._to_iso_string(now),
                "reasons": ["Fallback minimal signal"],
                "patterns": [],
                "market_context": "Fallback mode - no analyzer available"
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error creating fallback prediction: {e}")
            raise
    
    @staticmethod
    def _to_iso_string(dt: Any) -> str:
        """
        Convert datetime to ISO format: YYYY-MM-DDTHH:MM:SSZ
        
        Args:
            dt: datetime object or string
        
        Returns:
            ISO formatted string (no space before Z)
        """
        if isinstance(dt, str):
            # Already a string, try to parse and reformat
            try:
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except:
                return dt
        
        if isinstance(dt, datetime):
            # Format as ISO with Z suffix
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        return str(dt)
    
    @staticmethod
    def validate_prediction(prediction: Dict[str, Any]) -> bool:
        """
        Validate prediction schema
        
        Args:
            prediction: Prediction dict to validate
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'symbol', 'timeframe', 'source', 'direction',
            'entry_price', 'stop_loss', 'take_profits',
            'confidence_score', 'accuracy_percent', 'leverage',
            'current_price', 'timestamp', 'valid_until'
        ]
        
        for field in required_fields:
            if field not in prediction:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate take_profits structure
        if not isinstance(prediction['take_profits'], list) or len(prediction['take_profits']) != 3:
            logger.warning("Invalid take_profits: must be list of 3 items")
            return False
        
        for i, tp in enumerate(prediction['take_profits']):
            if not all(k in tp for k in ['level', 'price', 'eta']):
                logger.warning(f"Invalid take_profit at index {i}: missing required fields")
                return False
        
        # Validate ISO datetime format
        for dt_field in ['timestamp', 'valid_until']:
            dt_str = prediction[dt_field]
            if not dt_str.endswith('Z') or 'T' not in dt_str:
                logger.warning(f"Invalid datetime format in {dt_field}: {dt_str}")
                return False
        
        return True


def map_to_prediction(source_data: Any, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to map any source to unified prediction
    
    Args:
        source_data: SignalModel, dict, or other format
        **kwargs: Additional context (symbol, current_price, timeframe, etc.)
    
    Returns:
        Unified prediction dict
    """
    # Detect type and map accordingly
    if hasattr(source_data, 'symbol'):
        # SignalModel
        return PredictionMapper.from_signal_model(source_data)
    
    source = kwargs.get('source', 'UNKNOWN')
    
    if source == 'PRO':
        symbol = kwargs.get('symbol', 'UNKNOWN')
        timeframe = kwargs.get('timeframe', '15m')
        return PredictionMapper.from_pro_analyzer(source_data, symbol, timeframe)
    
    elif source == 'DASHBOARD':
        symbol = kwargs.get('symbol', 'UNKNOWN')
        current_price = kwargs.get('current_price', 0)
        timeframe = kwargs.get('timeframe', '15m')
        return PredictionMapper.from_dashboard(source_data, symbol, current_price, timeframe)
    
    elif source == 'FALLBACK':
        symbol = kwargs.get('symbol', 'UNKNOWN')
        current_price = kwargs.get('current_price', 0)
        timeframe = kwargs.get('timeframe', '15m')
        return PredictionMapper.from_fallback(symbol, current_price, timeframe)
    
    else:
        raise ValueError(f"Unknown source: {source}")
