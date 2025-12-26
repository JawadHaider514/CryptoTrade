"""
API Module
==========
Contains API integrations and trading system connectors.
"""

try:
    from .trading_integration import TradingSystemIntegration
    
    __all__ = [
        'TradingSystemIntegration',
    ]
except ImportError as e:
    print(f"Warning: Could not import API components: {e}")
    __all__ = []
