# 🏗️ System Architecture

## Overview

This document provides a comprehensive overview of the Crypto Trading System architecture, component relationships, and data flow.

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CRYPTO TRADING SYSTEM                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   External   │     │     Core     │     │    Server    │        │
│  │    APIs      │────▶│   Engine     │────▶│   Layer      │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                    │                    │                 │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Binance    │     │     ML       │     │   Frontend   │        │
│  │   Streaming  │     │  Predictor   │     │  Dashboard   │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
crypto_trading_system/
│
├── 📂 core/                          # Core Trading Logic
│   ├── __init__.py                   # Package exports
│   ├── enhanced_crypto_dashboard.py  # Main dashboard (3400+ lines)
│   └── trade_tracker.py              # Trade logging system
│
├── 📂 api/                           # API Integrations
│   ├── __init__.py
│   └── trading_integration.py        # Trading system connector
│
├── 📂 server/                        # Web Servers
│   ├── __init__.py
│   ├── web_server.py                 # Basic Flask server
│   └── advanced_web_server.py        # WebSocket server
│
├── 📂 models/                        # Data Models
│   ├── __init__.py
│   ├── signals.py                    # Signal dataclasses
│   └── portfolio.py                  # Portfolio models
│
├── 📂 config/                        # Configuration
│   ├── __init__.py
│   └── settings.py                   # All app settings
│
├── 📂 templates/                     # HTML Templates
├── 📂 static/                        # Static Assets
├── 📂 data/                          # Data Storage
│   ├── trades/                       # Trade exports
│   └── logs/                         # Application logs
│
├── 📂 tests/                         # Unit Tests
├── requirements.txt                  # Dependencies
├── run.py                            # Entry point
└── README.md                         # Documentation
```

---

## 🔧 Component Details

### 1. Core Engine (`core/enhanced_crypto_dashboard.py`)

The heart of the system. Contains:

| Class | Purpose | Lines |
|-------|---------|-------|
| `EnhancedScalpingDashboard` | Main orchestrator | ~500 |
| `BinanceStreamingAPI` | Real-time data fetching | ~100 |
| `AdvancedMLPredictor` | ML-based predictions | ~300 |
| `StreamingSignalProcessor` | Signal generation | ~400 |
| `DemoTradingBot` | Paper trading bot | ~600 |
| `ScalpingConfig` | Configuration | ~50 |
| `SignalFormatter` | Output formatting | ~200 |

**Key Features:**
- Real-time Binance data streaming
- Technical indicator calculations (RSI, MACD, Bollinger)
- ML-based price predictions
- Signal quality classification
- Demo trading with portfolio tracking

### 2. API Layer (`api/trading_integration.py`)

Connects signals to trade execution:

```python
class TradingSystemIntegration:
    def on_signal_generated(signal) -> trade_id
    def on_trade_exit(symbol, exit_data) -> None
    def get_statistics() -> Dict
```

### 3. Server Layer

#### Basic Server (`server/web_server.py`)
- Flask-based REST API
- Signal caching (3 minutes)
- HTML template serving
- CSV export endpoints

#### Advanced Server (`server/advanced_web_server.py`)
- WebSocket support via Flask-SocketIO
- Real-time updates
- Bot control endpoints
- Portfolio history streaming

### 4. Data Models (`models/`)

#### Signal Models (`signals.py`)
```python
@dataclass
class EnhancedSignal:
    symbol: str
    direction: str
    confidence: float
    quality: SignalQuality
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    predictions: PredictionMetrics
    ...
```

#### Portfolio Models (`portfolio.py`)
```python
@dataclass
class TradePosition:
    trade_id: str
    symbol: str
    entry_price: float
    pnl: float
    ...

@dataclass
class Portfolio:
    balance: float
    equity: float
    active_trades: List[TradePosition]
    ...
```

---

## 🔄 Data Flow

### Signal Generation Flow

```
1. Binance API
   │
   ├─▶ Get 24hr ticker data
   │
   ▼
2. Technical Analysis
   │
   ├─▶ Calculate RSI
   ├─▶ Calculate MACD
   ├─▶ Calculate Bollinger Bands
   ├─▶ Detect patterns
   │
   ▼
3. ML Predictor
   │
   ├─▶ Feature extraction
   ├─▶ Confidence scoring
   ├─▶ Price predictions
   │
   ▼
4. Signal Generation
   │
   ├─▶ Quality classification
   ├─▶ Entry/Exit levels
   ├─▶ Risk calculation
   │
   ▼
5. Output
   │
   ├─▶ Web Dashboard
   ├─▶ Discord (optional)
   └─▶ CSV Export
```

### Trade Execution Flow

```
1. Signal Received
   │
   ▼
2. TradingSystemIntegration
   │
   ├─▶ Log to TradeTracker
   ├─▶ Generate trade_id
   │
   ▼
3. DemoTradingBot
   │
   ├─▶ Check portfolio limits
   ├─▶ Calculate position size
   ├─▶ Open position
   │
   ▼
4. Position Monitoring
   │
   ├─▶ Update prices
   ├─▶ Check TP/SL levels
   │
   ▼
5. Position Close
   │
   ├─▶ Calculate PnL
   ├─▶ Update statistics
   └─▶ Export to CSV
```

---

## 📡 API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/signals` | GET | All active signals |
| `/api/signals/<symbol>` | GET | Single symbol signal |
| `/api/bot/start` | POST | Start trading bot |
| `/api/bot/stop` | POST | Stop trading bot |
| `/api/bot/status` | GET | Bot status |
| `/api/statistics` | GET | Trading statistics |
| `/api/trades` | GET | Trade history |
| `/api/portfolio/history` | GET | Portfolio equity history |
| `/api/coins` | GET | Coin data with signals |
| `/download/csv` | GET | Export trades CSV |

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client → Server | Client connection |
| `disconnect` | Client → Server | Client disconnection |
| `request_status` | Client → Server | Request bot status |
| `start_bot` | Client → Server | Start bot command |
| `stop_bot` | Client → Server | Stop bot command |
| `bot_update` | Server → Client | Real-time updates |
| `bot_status` | Server → Client | Status response |
| `connection_response` | Server → Client | Connection confirmation |

---

## ⚙️ Configuration System

Configuration is centralized in `config/settings.py`:

```python
APP_CONFIG = {...}        # Server settings
TRADING_CONFIG = {...}    # Trading parameters
SCALPING_CONFIG = {...}   # Scalping specific settings
BINANCE_CONFIG = {...}    # Binance API settings
ML_CONFIG = {...}         # ML model configuration
DISCORD_CONFIG = {...}    # Discord integration
LOGGING_CONFIG = {...}    # Logging settings
```

---

## 🔒 Security Considerations

1. **API Keys**: Store in environment variables
   ```bash
   export BINANCE_API_KEY="your_key"
   export BINANCE_SECRET_KEY="your_secret"
   ```

2. **CORS**: Configured in server files
3. **WebSocket**: Secure with proper origin checks
4. **Data**: Sensitive data not logged

---

## 🚀 Deployment

### Development
```bash
python run.py --mode basic
```

### Production
```bash
python run.py --mode advanced
```

### Testing
```bash
python run.py --test-bot
python run.py --test-timing
```

---

## 📈 Performance

| Component | Expected Latency |
|-----------|-----------------|
| Binance API call | ~100-500ms |
| Technical analysis | ~50-100ms |
| ML prediction | ~10-50ms |
| Signal generation | ~200-500ms total |
| WebSocket update | ~10-50ms |

---

## 🔮 Future Enhancements

1. **Real Trading Integration**
   - Connect to Binance Futures
   - Order execution
   - Position management

2. **Advanced ML**
   - Deep learning models
   - Sentiment analysis
   - News integration

3. **Additional Features**
   - Mobile app
   - Telegram bot
   - Advanced charting

---

*Last Updated: December 2025*
