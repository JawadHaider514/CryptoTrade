#!/usr/bin/env python3
"""
ML FEATURE EXTRACTOR
Extract 20+ features from signals for machine learning
Converts signals into numerical features that ML models can understand
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class MLFeatureExtractor:
    """Extract ML-ready features from trading signals"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_names = [
            'rsi_value',
            'rsi_delta',
            'macd_line',
            'macd_signal',
            'macd_histogram',
            'ema_fast',
            'ema_slow',
            'ema_separation',
            'atr_value',
            'volume_ratio',
            'close_above_ema',
            'pattern_count',
            'bullish_pattern_count',
            'bearish_pattern_count',
            'strongest_pattern_score',
            'hour_of_day',
            'volatility_score',
            'confluence_score',
            'direction_is_long',
            'indicator_alignment_score'
        ]
        self.feature_count = len(self.feature_names)
    
    def extract_features(self, signal: Dict) -> Dict[str, float]:
        """
        Extract all features from a signal
        Returns dictionary with feature_name: value pairs
        """
        features = {}
        
        try:
            # 1. RSI Features (2 features)
            rsi = signal.get('rsi', 50)
            features['rsi_value'] = float(rsi) / 100.0  # Normalize to 0-1
            features['rsi_delta'] = (rsi - 50) / 50.0  # -1 to +1 from midpoint
            
            # 2. MACD Features (3 features)
            macd = signal.get('macd', 0)
            signal_line = signal.get('macd_signal', 0)
            features['macd_line'] = float(macd)
            features['macd_signal'] = float(signal_line)
            features['macd_histogram'] = float(macd - signal_line)
            
            # 3. EMA Features (3 features)
            ema_fast = signal.get('ema_fast', 0)
            ema_slow = signal.get('ema_slow', 0)
            close_price = signal.get('close_price', ema_fast or 1)
            
            features['ema_fast'] = (float(ema_fast) / float(close_price)) - 1 if close_price > 0 else 0
            features['ema_slow'] = (float(ema_slow) / float(close_price)) - 1 if close_price > 0 else 0
            features['ema_separation'] = (float(ema_fast - ema_slow) / float(close_price)) if close_price > 0 else 0
            
            # 4. ATR Feature (1 feature)
            atr = signal.get('atr', 0)
            atr_percent = (float(atr) / float(close_price) * 100) if close_price > 0 else 0
            features['atr_value'] = min(atr_percent / 5, 1.0)  # Cap at 5% volatility
            
            # 5. Volume Feature (1 feature)
            volume_ratio = signal.get('volume_ratio', 1.0)
            features['volume_ratio'] = min(float(volume_ratio), 3.0) / 3.0  # Normalize (cap at 3x)
            
            # 6. Price Position Features (1 feature)
            close_above_ema = 1.0 if (ema_fast and close_price > ema_fast) else 0.0
            features['close_above_ema'] = float(close_above_ema)
            
            # 7. Pattern Features (4 features)
            patterns = signal.get('detected_patterns', [])
            pattern_score = signal.get('pattern_score', 0)
            
            features['pattern_count'] = min(float(len(patterns)), 3.0) / 3.0  # Max 3 patterns
            
            # Count bullish vs bearish patterns
            bullish_patterns = ['hammer', 'bullish_engulfing', 'three_white_soldiers']
            bearish_patterns = ['shooting_star', 'bearish_engulfing', 'three_black_crows']
            
            bullish_count = sum(1 for p in patterns if p.lower() in bullish_patterns)
            bearish_count = sum(1 for p in patterns if p.lower() in bearish_patterns)
            
            features['bullish_pattern_count'] = float(bullish_count)
            features['bearish_pattern_count'] = float(bearish_count)
            features['strongest_pattern_score'] = float(pattern_score) / 25.0  # Normalize to 0-1
            
            # 8. Time Features (1 feature)
            timestamp = signal.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            hour = timestamp.hour
            features['hour_of_day'] = float(hour) / 24.0  # Normalize to 0-1
            
            # 9. Volatility Feature (1 feature)
            volatility = signal.get('volatility_score', 50)
            features['volatility_score'] = float(volatility) / 100.0  # Normalize
            
            # 10. Confluence Score (1 feature)
            confluence = signal.get('confluence_score', 50)
            features['confluence_score'] = float(confluence) / 100.0  # Normalize
            
            # 11. Direction Feature (1 feature)
            direction = signal.get('direction', 'LONG').upper()
            features['direction_is_long'] = 1.0 if direction == 'LONG' else 0.0
            
            # 12. Indicator Alignment (1 feature)
            # Check if RSI, MACD, and EMA all agree on direction
            rsi_bullish = rsi > 50
            macd_bullish = macd > signal_line
            ema_bullish = ema_fast > ema_slow
            
            agreement_score = sum([rsi_bullish, macd_bullish, ema_bullish]) / 3.0
            features['indicator_alignment_score'] = agreement_score
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return zeros if extraction fails
            features = {name: 0.0 for name in self.feature_names}
        
        return features
    
    def extract_features_batch(self, signals: List[Dict]) -> pd.DataFrame:
        """
        Extract features from multiple signals
        Returns DataFrame with signals as rows and features as columns
        """
        features_list = []
        
        for signal in signals:
            features = self.extract_features(signal)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def create_feature_labels(self, signals: List[Dict], outcomes: List[Dict]) -> pd.DataFrame:
        """
        Create training data with features and labels
        
        Args:
            signals: List of signal dictionaries
            outcomes: List of outcome dictionaries with 'signal_id' and 'result' (WIN/LOSS/TIMEOUT)
        
        Returns:
            DataFrame with features and 'label' column (1 for WIN, 0 for LOSS)
        """
        df_features = self.extract_features_batch(signals)
        
        # Create label column
        labels = []
        for signal in signals:
            signal_id = signal.get('id', signal.get('signal_id'))
            
            # Find matching outcome
            outcome = None
            for o in outcomes:
                if o.get('signal_id') == signal_id:
                    outcome = o
                    break
            
            # Label: 1 for WIN, 0 for LOSS (ignore TIMEOUT)
            if outcome:
                is_win = outcome.get('result') == 'WIN'
                labels.append(1 if is_win else 0)
            else:
                labels.append(0)  # Default to loss if no outcome
        
        df_features['label'] = labels
        return df_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names
    
    def validate_features(self, features: Dict) -> bool:
        """Validate that all required features are present"""
        for feature_name in self.feature_names:
            if feature_name not in features:
                logger.warning(f"Missing feature: {feature_name}")
                return False
        return True
    
    def normalize_features(self, features: Dict) -> Dict:
        """
        Normalize features to 0-1 range
        (features are already mostly normalized, this is additional safety)
        """
        normalized = {}
        for name, value in features.items():
            # Clamp values to 0-1 range
            normalized[name] = max(0.0, min(1.0, float(value)))
        return normalized
    
    @staticmethod
    def features_to_array(features: Dict, feature_names: List[str]) -> np.ndarray:
        """Convert feature dictionary to numpy array for ML model input"""
        return np.array([features.get(name, 0.0) for name in feature_names], dtype=np.float32)
    
    def explain_features(self, features: Dict, top_k: int = 5) -> List[tuple]:
        """
        Explain which features have the highest values
        Returns list of (feature_name, value) tuples sorted by value
        """
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:top_k]


class SignalFeatureAnalyzer:
    """Analyze feature importance and signal quality"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.extractor = MLFeatureExtractor()
    
    def analyze_signal_quality(self, signal: Dict) -> Dict:
        """
        Analyze overall quality of a signal based on features
        Returns quality metrics
        """
        features = self.extractor.extract_features(signal)
        
        analysis = {
            'confluence_score': signal.get('confluence_score', 50),
            'accuracy_estimate': signal.get('accuracy_estimate', 50),
            'indicator_alignment': features.get('indicator_alignment_score', 0),
            'pattern_strength': features.get('strongest_pattern_score', 0),
            'volatility': features.get('volatility_score', 0),
            'volume_confirmation': features.get('volume_ratio', 0)
        }
        
        # Calculate overall quality score
        weights = {
            'confluence_score': 0.3,
            'accuracy_estimate': 0.25,
            'indicator_alignment': 0.2,
            'pattern_strength': 0.15,
            'volume_confirmation': 0.1
        }
        
        total_quality = sum(
            analysis[key] * weight / 100.0
            for key, weight in weights.items()
        )
        
        analysis['overall_quality'] = total_quality
        return analysis
    
    def compare_signals(self, signals: List[Dict]) -> pd.DataFrame:
        """Compare multiple signals side-by-side"""
        analysis_list = []
        
        for signal in signals:
            analysis = self.analyze_signal_quality(signal)
            analysis['symbol'] = signal.get('symbol', 'UNKNOWN')
            analysis['direction'] = signal.get('direction', 'UNKNOWN')
            analysis_list.append(analysis)
        
        return pd.DataFrame(analysis_list)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example: Extract features from a signal"""
    
    # Sample signal
    sample_signal = {
        'id': 1,
        'symbol': 'BTCUSDT',
        'direction': 'LONG',
        'timestamp': datetime.now(),
        'rsi': 65,
        'macd': 0.0012,
        'macd_signal': 0.0008,
        'ema_fast': 43500,
        'ema_slow': 43200,
        'close_price': 43450,
        'atr': 150,
        'volume_ratio': 1.5,
        'detected_patterns': ['hammer', 'bullish_engulfing'],
        'pattern_score': 18,
        'volatility_score': 55,
        'confluence_score': 78,
        'accuracy_estimate': 68
    }
    
    # Extract features
    extractor = MLFeatureExtractor()
    features = extractor.extract_features(sample_signal)
    
    print("Extracted Features:")
    print("=" * 50)
    for name, value in features.items():
        print(f"{name:30s}: {value:.4f}")
    
    # Explain top features
    print("\n\nTop 5 Strongest Features:")
    print("=" * 50)
    top_features = extractor.explain_features(features, top_k=5)
    for name, value in top_features:
        print(f"{name:30s}: {value:.4f}")
    
    # Analyze quality
    analyzer = SignalFeatureAnalyzer()
    quality = analyzer.analyze_signal_quality(sample_signal)
    
    print("\n\nSignal Quality Analysis:")
    print("=" * 50)
    for key, value in quality.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.2f}")
        else:
            print(f"{key:30s}: {value}")
