"""
⚙️ CRYPTO TRADING SYSTEM - CONFIGURATION
=========================================
Central configuration for all system components.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TRADES_DIR = DATA_DIR / "trades"
LOGS_DIR = DATA_DIR / "logs"

# Template directories
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

# Ensure directories exist
for directory in [DATA_DIR, TRADES_DIR, LOGS_DIR, TEMPLATES_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

APP_CONFIG = {
    # Server settings
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'DEBUG': True,
    'USE_RELOADER': False,
    
    # CORS settings
    'CORS_ENABLED': True,
    'CORS_ORIGINS': '*',
    
    # WebSocket settings
    'WEBSOCKET_PING_TIMEOUT': 60,
    'WEBSOCKET_PING_INTERVAL': 25,
    
    # Cache settings
    'SIGNAL_CACHE_DURATION': 180,  # 3 minutes in seconds
    
    # Auto-refresh interval (milliseconds)
    'AUTO_REFRESH_INTERVAL': 30000,  # 30 seconds
    
    # Discord webhook for notifications
    'DISCORD_WEBHOOK': 'https://discord.com/api/webhooks/1447651247749337089/tajiT4cIfvOrAUxVxHyR2lQT3S6wxMb_iPJ2PCkshPoeH7g6UoxW-FPVIEQMfC70BblV',
    
    # Binance API Keys (Testnet)
    'BINANCE_API_KEY': 'Gkc3zKLbCxgoT5aM1lKMKynLSOfdkwDfUPGfwS1bYiBCVvXyeSergvumsIFSz1nA',
    'BINANCE_SECRET_KEY': 's8LPPl1DhZ4QVwODh4ZgyiorxcxNC59Kv44cCGohLzElxj0ci4xibwr0u8YikkrI',
    'BINANCE_TESTNET': True,  # Use testnet instead of live trading
    'BINANCE_TESTNET_URL': 'https://testnet.binance.vision',
}


# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

TRADING_CONFIG = {
    # Signal thresholds
    'MIN_CONFLUENCE_SCORE': 65,
    'MIN_ACCURACY_ESTIMATE': 75,
    
    # Leverage settings
    'DEFAULT_LEVERAGE': 20,
    'MAX_LEVERAGE': 50,
    'MIN_LEVERAGE': 1,
    
    # Risk management
    'RISK_PERCENTAGE': 2.0,
    'MAX_CONCURRENT_TRADES': 5,
    'POSITION_SIZE_USDT': 100.0,
    
    # Take profit configuration
    'TP1_PERCENTAGE': 40,  # First TP takes 40% of position
    'TP2_PERCENTAGE': 35,  # Second TP takes 35%
    'TP3_PERCENTAGE': 25,  # Final TP takes remaining 25%
    
    # Timing windows (in minutes)
    'SIGNAL_VALIDITY': 5,  # Signal valid for 5 minutes
    'TP1_WINDOW': 2,       # TP1 should hit within 2 minutes
    'TP2_WINDOW': 3.5,     # TP2 within 3.5 minutes
    'TP3_WINDOW': 5,       # TP3 within 5 minutes (hard stop)
}


# =============================================================================
# SCALPING CONFIGURATION
# =============================================================================

SCALPING_CONFIG = {
    # Time intervals
    'INTERVALS': ['1m', '3m', '5m', '15m'],
    'DEFAULT_INTERVAL': '5m',
    
    # Volatility categories
    'VOLATILITY': {
        'LOW': {
            'tp1_range': (5, 15),      # TP1: 5-15 minutes
            'tp2_range': (10, 20),     # TP2: +10-20 minutes
            'tp3_range': (15, 45),     # TP3: +15-45 minutes
        },
        'MEDIUM': {
            'tp1_range': (3, 10),      # TP1: 3-10 minutes
            'tp2_range': (5, 15),      # TP2: +5-15 minutes
            'tp3_range': (10, 30),     # TP3: +10-30 minutes
        },
        'HIGH': {
            'tp1_range': (1, 5),       # TP1: 1-5 minutes
            'tp2_range': (3, 10),      # TP2: +3-10 minutes
            'tp3_range': (5, 20),      # TP3: +5-20 minutes
        },
    },
    
    # Coins by volatility
    'HIGH_VOLATILITY_COINS': ['PEPEUSDT', 'SHIBUSDT', 'DOGEUSDT', 'FLOKIUSDT'],
    'LOW_VOLATILITY_COINS': ['BTCUSDT', 'ETHUSDT'],
}


# =============================================================================
# BINANCE CONFIGURATION
# =============================================================================

BINANCE_CONFIG = {
    # API endpoints
    'BASE_URL': 'https://api.binance.com/api/v3',
    'FUTURES_URL': 'https://fapi.binance.com',
    'TESTNET_URL': 'https://testnet.binancefuture.com',
    
    # API credentials (set via environment variables for security)
    'API_KEY': os.environ.get('BINANCE_API_KEY', ''),
    'SECRET_KEY': os.environ.get('BINANCE_SECRET_KEY', ''),
    
    # Testnet credentials
    'TESTNET_API_KEY': os.environ.get('BINANCE_TESTNET_API_KEY', ''),
    'TESTNET_SECRET_KEY': os.environ.get('BINANCE_TESTNET_SECRET_KEY', ''),
    
    # Use testnet by default
    'USE_TESTNET': True,
    
    # Request settings
    'TIMEOUT': 5,
    'MAX_RETRIES': 3,
}


# =============================================================================
# TRADING PAIRS
# =============================================================================

TRADING_PAIRS = [
    # Major pairs
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT',
    
    # Alt coins
    'XRPUSDT',
    'ADAUSDT',
    'SOLUSDT',
    'DOTUSDT',
    'AVAXUSDT',
    'MATICUSDT',
    'LINKUSDT',
    
    # Meme coins
    'DOGEUSDT',
    'SHIBUSDT',
    'PEPEUSDT',
    'FLOKIUSDT',
]


# =============================================================================
# ML CONFIGURATION
# =============================================================================

ML_CONFIG = {
    # Feature weights for ML predictions
    'FEATURE_WEIGHTS': {
        'rsi': 0.15,
        'macd': 0.12,
        'bollinger': 0.10,
        'volume_profile': 0.15,
        'price_momentum': 0.18,
        'market_correlation': 0.10,
        'volatility': 0.08,
        'pattern_recognition': 0.12,
    },
    
    # Technical indicator periods
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BB_PERIOD': 20,
    'BB_STD': 2,
    
    # Signal quality thresholds
    'QUALITY_THRESHOLDS': {
        'PREMIUM': 85,
        'HIGH': 75,
        'MEDIUM': 65,
        'LOW': 0,
    },
}


# =============================================================================
# DISCORD CONFIGURATION
# =============================================================================

DISCORD_CONFIG = {
    'WEBHOOK_URL': os.environ.get('DISCORD_WEBHOOK_URL', ''),
    'ENABLED': False,  # Set to True when webhook is configured
    'NOTIFICATION_TYPES': ['signals', 'trades', 'alerts'],
}


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'FILE_ENABLED': True,
    'FILE_PATH': LOGS_DIR / 'trading.log',
    'MAX_BYTES': 10 * 1024 * 1024,  # 10 MB
    'BACKUP_COUNT': 5,
}


# =============================================================================
# CSV EXPORT CONFIGURATION
# =============================================================================

CSV_CONFIG = {
    'TRADES_FILE_PREFIX': 'trades',
    'PORTFOLIO_FILE_PREFIX': 'portfolio',
    'SIGNALS_FILE_PREFIX': 'signals',
    'DATE_FORMAT': '%Y%m%d',
    'TIMESTAMP_FORMAT': '%Y-%m-%d %H:%M:%S',
}


# =============================================================================
# COIN NAME MAPPING
# =============================================================================

COIN_NAMES = {
    'BTCUSDT': 'Bitcoin',
    'ETHUSDT': 'Ethereum',
    'XRPUSDT': 'Ripple',
    'BNBUSDT': 'Binance Coin',
    'ADAUSDT': 'Cardano',
    'SOLUSDT': 'Solana',
    'DOGEUSDT': 'Dogecoin',
    'DOTUSDT': 'Polkadot',
    'AVAXUSDT': 'Avalanche',
    'MATICUSDT': 'Polygon',
    'LINKUSDT': 'Chainlink',
    'SHIBUSDT': 'Shiba Inu',
    'PEPEUSDT': 'Pepe',
    'FLOKIUSDT': 'Floki',
}


def get_coin_name(symbol: str) -> str:
    """Get human-readable coin name from symbol"""
    return COIN_NAMES.get(symbol, symbol.replace('USDT', ''))
