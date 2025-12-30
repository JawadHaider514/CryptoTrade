"""Signal quality enforcement - uses per-coin thresholds to filter weak signals"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class SignalQualityGate:
    """Enforces per-coin quality thresholds"""
    
    def __init__(self, thresholds_file: Optional[Path] = None):
        self.thresholds = {}
        self.thresholds_file = thresholds_file or (CONFIG_DIR / "per_coin_thresholds.json")
        self.load_thresholds()
    
    def load_thresholds(self):
        """Load per-coin thresholds from config"""
        if self.thresholds_file.exists():
            try:
                with open(self.thresholds_file) as f:
                    data = json.load(f)
                    self.thresholds = data.get("coins", {})
                    logger.info(f"Loaded thresholds for {len(self.thresholds)} coins")
            except Exception as e:
                logger.warning(f"Failed to load thresholds: {e}")
                self.thresholds = {}
        else:
            logger.warning(f"Thresholds file not found: {self.thresholds_file}")
    
    def should_trade(self, symbol: str, confidence: float = 0.5) -> Dict:
        """
        Check if signal should be traded based on quality gates
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            confidence: ML model confidence (0-1)
        
        Returns:
            Dict with decision, action, reason
        """
        
        if symbol not in self.thresholds:
            return {
                "should_trade": False,
                "action": "NO_TRADE",
                "reason": "Symbol not in thresholds - insufficient quality data",
                "min_confidence": 0.75  # Default high threshold
            }
        
        coin_config = self.thresholds[symbol]
        action = coin_config.get("action", "NO_TRADE")
        min_confidence = coin_config.get("min_confidence", 0.75)
        score = coin_config.get("score_0_100", 0)
        
        # Quality gates
        if action == "NO_TRADE":
            return {
                "should_trade": False,
                "action": "NO_TRADE",
                "reason": coin_config.get("reason", "Insufficient quality"),
                "score": score,
                "quality_tier": coin_config.get("quality_tier", "REJECTED")
            }
        
        # Check confidence threshold
        if confidence < min_confidence:
            return {
                "should_trade": False,
                "action": "FILTERED",
                "reason": f"Low confidence ({confidence:.1%}) < min ({min_confidence:.1%})",
                "score": score,
                "quality_tier": coin_config.get("quality_tier")
            }
        
        # Check action level
        if action == "ACTIVE":
            return {
                "should_trade": True,
                "action": "ACTIVE",
                "reason": f"Quality tier {coin_config.get('quality_tier')} - confidence {confidence:.1%}",
                "score": score,
                "quality_tier": coin_config.get("quality_tier")
            }
        
        elif action == "WARMUP":
            return {
                "should_trade": False,
                "action": "WARMUP",
                "reason": coin_config.get("reason", "In warmup period - monitor before trading"),
                "score": score,
                "quality_tier": coin_config.get("quality_tier")
            }
        
        elif action == "TRADE_WITH_CAUTION":
            return {
                "should_trade": True,
                "action": "TRADE_WITH_CAUTION",
                "reason": coin_config.get("reason", "Marginal quality - reduce position size"),
                "score": score,
                "quality_tier": coin_config.get("quality_tier"),
                "position_size_multiplier": 0.5  # Suggest half position size
            }
        
        else:
            return {
                "should_trade": False,
                "action": "NO_TRADE",
                "reason": f"Unknown action: {action}",
                "score": score
            }
    
    def filter_signals(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Apply quality gates to all signals
        
        Args:
            signals: Dict of {symbol: signal_dict}
        
        Returns:
            Filtered signals with quality metadata
        """
        filtered = {}
        
        for symbol, signal in signals.items():
            if not signal:
                continue
            
            confidence = signal.get("confidence", 0.5)
            gate_decision = self.should_trade(symbol, confidence)
            
            # Add quality gate metadata
            signal["quality_gate"] = {
                "should_trade": gate_decision["should_trade"],
                "action": gate_decision["action"],
                "reason": gate_decision["reason"],
                "score": gate_decision.get("score"),
                "quality_tier": gate_decision.get("quality_tier")
            }
            
            # Mark signal if filtered
            if not gate_decision["should_trade"]:
                signal["filtered"] = True
                signal["filter_reason"] = gate_decision["reason"]
            
            filtered[symbol] = signal
        
        return filtered


def get_quality_gate() -> SignalQualityGate:
    """Get singleton SignalQualityGate instance"""
    if not hasattr(get_quality_gate, '_instance'):
        get_quality_gate._instance = SignalQualityGate()
    return get_quality_gate._instance
