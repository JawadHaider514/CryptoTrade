"""
Core Trading Module
===================
Contains the main trading dashboard and analysis components.
"""

# Import main classes for easier access
try:
    from .enhanced_crypto_dashboard import (
        EnhancedScalpingDashboard,
        ScalpingConfig,
        SignalFormatter,
        BinanceStreamingAPI,
        StreamingSignalProcessor,
        AdvancedMLPredictor,
        DemoTradingBot,
        # Re-export signal classes
        SignalQuality,
        PredictionMetrics,
        EnhancedSignal,
    )
    
    __all__ = [
        'EnhancedScalpingDashboard',
        'ScalpingConfig',
        'SignalFormatter',
        'BinanceStreamingAPI',
        'StreamingSignalProcessor',
        'AdvancedMLPredictor',
        'DemoTradingBot',
        'SignalQuality',
        'PredictionMetrics',
        'EnhancedSignal',
    ]
except ImportError as e:
    print(f"Warning: Could not import core components: {e}")
    __all__ = []
