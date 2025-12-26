"""
Configuration Package
"""

from .settings import (
    APP_CONFIG,
    TRADING_CONFIG,
    SCALPING_CONFIG,
    BINANCE_CONFIG,
    ML_CONFIG,
    DISCORD_CONFIG,
    LOGGING_CONFIG,
    CSV_CONFIG,
    TRADING_PAIRS,
    COIN_NAMES,
    get_coin_name,
    PROJECT_ROOT,
    DATA_DIR,
    TRADES_DIR,
    LOGS_DIR,
    TEMPLATES_DIR,
    STATIC_DIR,
)

__all__ = [
    'APP_CONFIG',
    'TRADING_CONFIG',
    'SCALPING_CONFIG',
    'BINANCE_CONFIG',
    'ML_CONFIG',
    'DISCORD_CONFIG',
    'LOGGING_CONFIG',
    'CSV_CONFIG',
    'TRADING_PAIRS',
    'COIN_NAMES',
    'get_coin_name',
    'PROJECT_ROOT',
    'DATA_DIR',
    'TRADES_DIR',
    'LOGS_DIR',
    'TEMPLATES_DIR',
    'STATIC_DIR',
]
