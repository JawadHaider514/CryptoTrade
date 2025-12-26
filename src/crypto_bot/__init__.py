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

# Main exports
from crypto_bot.core.signal_generator import SignalGenerator
from crypto_bot.core.exchange_adapter import ExchangeAdapter, get_adapter
from crypto_bot.core.paper_trader import PaperTrader
from crypto_bot.core.risk_manager import RiskManager
from crypto_bot.server.web_server import create_app

__all__ = [
    "SignalGenerator",
    "ExchangeAdapter",
    "get_adapter",
    "PaperTrader",
    "RiskManager",
    "create_app",
]
