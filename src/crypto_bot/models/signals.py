"""
📊 SIGNAL MODELS
================
Data models for trading signals and predictions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any


class SignalQuality(Enum):
    """Signal quality classification"""
    PREMIUM = "PREMIUM"   # 85+ confluence score
    HIGH = "HIGH"         # 75-84 confluence score
    MEDIUM = "MEDIUM"     # 65-74 confluence score
    LOW = "LOW"           # <65 confluence score


class SignalDirection(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"


class SignalStatus(Enum):
    """Signal status"""
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    TP3_HIT = "TP3_HIT"
    SL_HIT = "SL_HIT"
    CANCELLED = "CANCELLED"


@dataclass
class PredictionMetrics:
    """
    Advanced prediction metrics with ML features.
    Contains all ML-based predictions for a signal.
    """
    price_target_30m: float
    price_target_1h: float
    price_target_3h: float
    breakout_probability: float
    volume_surge_probability: float
    trend_reversal_probability: float
    market_correlation_score: float
    volatility_prediction: float
    confidence_interval: Tuple[float, float]
    risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'price_target_30m': self.price_target_30m,
            'price_target_1h': self.price_target_1h,
            'price_target_3h': self.price_target_3h,
            'breakout_probability': self.breakout_probability,
            'volume_surge_probability': self.volume_surge_probability,
            'trend_reversal_probability': self.trend_reversal_probability,
            'market_correlation_score': self.market_correlation_score,
            'volatility_prediction': self.volatility_prediction,
            'confidence_interval': list(self.confidence_interval),
            'risk_score': self.risk_score,
        }


@dataclass
class EnhancedSignal:
    """
    Enhanced signal with ML predictions and streaming metadata.
    This is the primary signal object used throughout the system.
    """
    symbol: str
    direction: str
    confidence: float
    quality: SignalQuality
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    current_price: float
    volume_24h: float
    change_24h: float
    predictions: PredictionMetrics
    patterns: List[str]
    timestamp: datetime
    processing_time: float
    data_sources: List[str]
    ml_features: Dict
    leverage: int = 20
    status: SignalStatus = SignalStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for API response"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'quality': self.quality.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'current_price': self.current_price,
            'volume_24h': self.volume_24h,
            'change_24h': self.change_24h,
            'predictions': self.predictions.to_dict(),
            'patterns': self.patterns,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'data_sources': self.data_sources,
            'ml_features': self.ml_features,
            'leverage': self.leverage,
            'status': self.status.value,
        }
    
    @property
    def take_profits(self) -> List[Tuple[float, int]]:
        """Get take profits as list of (price, percentage) tuples"""
        return [
            (self.take_profit_1, 40),
            (self.take_profit_2, 35),
            (self.take_profit_3, 25),
        ]
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit_1 - self.entry_price)
        return reward / risk if risk > 0 else 0
    
    @property
    def is_valid(self) -> bool:
        """Check if signal is still valid"""
        from datetime import timedelta
        validity_window = timedelta(minutes=5)
        return (
            self.status == SignalStatus.ACTIVE and
            datetime.now() - self.timestamp < validity_window
        )


@dataclass
class BasicSignal:
    """
    Basic signal format (simpler version).
    Used for compatibility with older components.
    """
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profits: List[Tuple[float, int]]  # [(price, percentage), ...]
    confluence_score: float
    accuracy_estimate: float
    timestamp: datetime
    detected_patterns: List[str] = field(default_factory=list)
    leverage: int = 20
    risk_percentage: float = 2.0
    entry_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profits': self.take_profits,
            'confluence_score': self.confluence_score,
            'accuracy_estimate': self.accuracy_estimate,
            'timestamp': self.timestamp.isoformat(),
            'detected_patterns': self.detected_patterns,
            'leverage': self.leverage,
            'risk_percentage': self.risk_percentage,
            'entry_range': list(self.entry_range),
        }


@dataclass
class SignalTimeline:
    """
    Timeline information for a signal.
    Contains expected times for each target.
    """
    entry_time: str
    entry_label: str
    tp1_time: str
    tp1_label: str
    tp2_time: str
    tp2_label: str
    tp3_time: str
    tp3_label: str
    exit_time: str
    exit_label: str
    total_window: str
    strategy: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            'entry_time': self.entry_time,
            'entry_label': self.entry_label,
            'tp1_time': self.tp1_time,
            'tp1_label': self.tp1_label,
            'tp2_time': self.tp2_time,
            'tp2_label': self.tp2_label,
            'tp3_time': self.tp3_time,
            'tp3_label': self.tp3_label,
            'exit_time': self.exit_time,
            'exit_label': self.exit_label,
            'total_window': self.total_window,
            'strategy': self.strategy,
        }


def create_signal_timeline(
    signal_timestamp: datetime,
    tp1_minutes: float = 2,
    tp2_minutes: float = 3.5,
    tp3_minutes: float = 5
) -> SignalTimeline:
    """
    Create a signal timeline with expected times.
    
    Args:
        signal_timestamp: When the signal was generated
        tp1_minutes: Minutes until TP1 (default: 2)
        tp2_minutes: Minutes until TP2 (default: 3.5)
        tp3_minutes: Minutes until TP3 (default: 5)
    
    Returns:
        SignalTimeline object
    """
    from datetime import timedelta
    
    return SignalTimeline(
        entry_time=signal_timestamp.strftime("%H:%M:%S"),
        entry_label='NOW',
        tp1_time=(signal_timestamp + timedelta(minutes=tp1_minutes)).strftime("%H:%M:%S"),
        tp1_label=f'Must Hit ({tp1_minutes} min)',
        tp2_time=(signal_timestamp + timedelta(minutes=tp2_minutes)).strftime("%H:%M:%S"),
        tp2_label=f'Scalp ({tp2_minutes} min)',
        tp3_time=(signal_timestamp + timedelta(minutes=tp3_minutes)).strftime("%H:%M:%S"),
        tp3_label=f'Final Exit ({tp3_minutes} min)',
        exit_time=(signal_timestamp + timedelta(minutes=tp3_minutes)).strftime("%H:%M:%S"),
        exit_label=f'Hard Stop ({tp3_minutes} min)',
        total_window=f'{tp3_minutes} minutes',
        strategy='INTELLIGENT_SCALP'
    )
