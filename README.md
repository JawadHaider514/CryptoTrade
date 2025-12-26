# Crypto Trading System

A professional-grade cryptocurrency trading system with real-time signal generation, backtesting capabilities, paper trading, and machine learning integration.

## Features

- **Real-time Signal Generation**: Live signal analysis using multiple technical indicators
- **Backtesting Engine**: Test strategies on historical data with detailed performance metrics
- **Paper Trading**: Simulate trades without using real money
- **Live Trading Dashboard**: Web-based dashboard with real-time updates via WebSocket
- **Machine Learning**: Train and predict signal quality with RandomForest models
- **Risk Management**: Advanced position sizing and risk controls
- **Multi-exchange Support**: Binance integration with testnet support

## Project Structure

```
crypto_trading_system/
│
├── 📂 core/                          # Core Trading Logic
│   ├── __init__.py
│   ├── enhanced_crypto_dashboard.py  # Main dashboard with ML signals
│   └── trade_tracker.py              # Trade tracking & statistics
│
├── 📂 api/                           # API Layer
│   ├── __init__.py
│   ├── binance_api.py                # Binance streaming API
│   └── trading_integration.py        # Integration module
│
├── 📂 server/                        # Web Servers
│   ├── __init__.py
│   ├── web_server.py                 # Basic Flask server
│   └── advanced_web_server.py        # Advanced server with WebSocket
│
├── 📂 models/                        # Data Models & Enums
│   ├── __init__.py
│   ├── signals.py                    # Signal dataclasses
│   └── portfolio.py                  # Portfolio models
│
├── 📂 templates/                     # HTML Templates
│   └── index.html                    # Dashboard HTML
│
├── 📂 static/                        # Static Assets
│   ├── css/
│   └── js/
│
├── 📂 data/                          # Data Storage
│   ├── trades/                       # Trade CSV exports
│   └── logs/                         # Application logs
│
├── 📂 config/                        # Configuration
│   ├── __init__.py
│   ├── settings.py                   # App settings
│   └── binance_config.py             # Binance API config
│
├── 📂 tests/                         # Unit Tests
│   ├── __init__.py
│   ├── test_signals.py
│   └── test_trading.py
│
├── requirements.txt                  # Python dependencies
├── run.py                            # Main entry point
└── README.md                         # This file
```

---

## 🔧 Components Description

### 1. Core Module (`core/`)

| File | Purpose |
|------|---------|
| `enhanced_crypto_dashboard.py` | Main trading engine with ML predictions, technical analysis, signal generation |
| `trade_tracker.py` | Tracks all trades, calculates statistics, exports to CSV |

### 2. API Module (`api/`)

| File | Purpose |
|------|---------|
| `binance_api.py` | Binance streaming API wrapper for real-time data |
| `trading_integration.py` | Connects trading signals to trade tracker |

### 3. Server Module (`server/`)

| File | Purpose |
|------|---------|
| `web_server.py` | Basic Flask server with REST API endpoints |
| `advanced_web_server.py` | Advanced server with WebSocket for real-time updates |

### 4. Models Module (`models/`)

| File | Purpose |
|------|---------|
| `signals.py` | Signal dataclasses (EnhancedSignal, PredictionMetrics) |
| `portfolio.py` | Portfolio and trade position models |

---

## ⚡ Key Features

### Signal Generation
- ✅ Real-time price data from Binance
- ✅ Technical indicators (RSI, MACD, Bollinger Bands)
- ✅ ML-based predictions
- ✅ Multiple take-profit levels (TP1, TP2, TP3)
- ✅ Dynamic stop-loss calculation

### Trading Bot
- ✅ Demo trading mode
- ✅ Portfolio management
- ✅ Position tracking
- ✅ PnL calculation
- ✅ Risk management

### Web Dashboard
- ✅ Real-time signal display
- ✅ WebSocket updates
- ✅ Trade history
- ✅ Statistics & analytics
- ✅ CSV export

### Integrations
- ✅ Discord notifications
- ✅ Binance Testnet support
- ✅ CSV trade logging

---

## 🚀 Quick Start

```bash
# 1. Navigate to project
cd crypto_trading_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the system
python run.py

# 4. Open dashboard
# Visit: http://localhost:5000
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/signals` | GET | Get all active signals |
| `/api/signals/<symbol>` | GET | Get signal for specific symbol |
| `/api/bot/start` | POST | Start trading bot |
| `/api/bot/stop` | POST | Stop trading bot |
| `/api/statistics` | GET | Get trading statistics |
| `/api/trades` | GET | Get trade history |
| `/api/portfolio/history` | GET | Get portfolio history |
| `/download/csv` | GET | Download trades CSV |

---

## ⚙️ Configuration

### Binance API (Optional)
```python
# config/binance_config.py
TESTNET_API_KEY = "your_api_key"
TESTNET_SECRET_KEY = "your_secret_key"
```

### Trading Settings
```python
# config/settings.py
SCALPING_CONFIG = {
    'min_confluence_score': 65,
    'min_accuracy_estimate': 75,
    'default_leverage': 20,
    'risk_percentage': 2.0
}
```

---

## 📈 Signal Quality Levels

| Quality | Confluence Score | Description |
|---------|-----------------|-------------|
| PREMIUM | 85+ | Highest confidence signals |
| HIGH | 75-84 | Strong trading signals |
| MEDIUM | 65-74 | Moderate confidence |
| LOW | <65 | Use with caution |

---

## 🔄 Workflow

```
┌─────────────────┐
│  Binance API    │
│  (Price Data)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Technical      │
│  Analysis       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Predictor   │
│  (Signals)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Trade Tracker  │
│  (Logging)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web Dashboard  │
│  (Display)      │
└─────────────────┘
```

---

## 📝 Files Mapping

| Original File | New Location |
|---------------|--------------|
| `enhanced_crypto_dashboard.py` | `core/enhanced_crypto_dashboard.py` |
| `trading_integration.py` | `api/trading_integration.py` |
| `web_server.py` | `server/web_server.py` |
| `advanced_web_server.py` | `server/advanced_web_server.py` |

---

## 🛠️ Dependencies

```
flask>=2.0.0
flask-cors>=3.0.0
flask-socketio>=5.0.0
pandas>=1.3.0
numpy>=1.21.0
requests>=2.26.0
python-socketio>=5.0.0
eventlet>=0.30.0
```

---

## 📞 Support

For issues or questions about this project, refer to the code documentation or contact the developer.

---

**Version:** 1.0.0  
**Last Updated:** December 2025  
**License:** MIT
