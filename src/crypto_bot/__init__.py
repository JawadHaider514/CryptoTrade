"""
Crypto Trading System
=====================

A complete cryptocurrency trading system with backtesting, live trading,
and ML-based signal generation.

Modules:
    - core: Core trading logic (signals, backtest, traders, etc.)
    - api: Dashboard and trading API endpoints
    - server: Flask/SocketIO web server
    - config: Configuration management
    - models: Data models and ML models
"""

__version__ = "1.0.0"
__author__ = "Trading System Team"

# Avoid circular imports - users should import directly from submodules
__all__ = [
    "core",
    "api",
    "server",
    "config",
    "models",
]
