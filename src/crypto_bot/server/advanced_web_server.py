#!/usr/bin/env python3
"""
ADVANCED CRYPTO TRADING BOT WEB SERVER
Complete API with real-time updates, bot control, and live analytics

Templates:
  <PROJECT_ROOT>/templates/index.html
  <PROJECT_ROOT>/templates/dashboard.html

This file also provides compatibility endpoints:
  /api/prediction  -> /api/predictions
  /api/price       -> /api/prices
  /api/health
"""

from __future__ import annotations

import os
import sys
import time
import threading
import logging
import queue
import csv
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Note: SocketIO uses threading mode (no eventlet)

# =============================================================================
# Logging (must be before any code that uses logger)
# =============================================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Note: Predictions API blueprint is NOT used - advanced_web_server.py provides
# superior /api/predictions endpoint that reads from signal_repo directly
PREDICTIONS_API_AVAILABLE = True


# -----------------------------------------------------------------------------
# Optional dotenv
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Load feature flags from settings
try:
    from crypto_bot.config.settings import SIGNAL_REFRESH_INTERVAL, SIGNAL_VALID_MINUTES, MIN_CONFIDENCE, MIN_ACCURACY
except ImportError:
    # Fallback defaults
    SIGNAL_REFRESH_INTERVAL = 30
    SIGNAL_VALID_MINUTES = 240
    MIN_CONFIDENCE = 15
    MIN_ACCURACY = 0

# -----------------------------------------------------------------------------
# Paths / imports
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]  # .../crypto_trading_system
SRC_DIR = PROJECT_ROOT / "src"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

# Allow running this file directly (optional). main.py already adds ./src.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -----------------------------------------------------------------------------
# Core dashboard imports (always bind names to avoid Pylance "unbound")
# -----------------------------------------------------------------------------
DASHBOARD_AVAILABLE = False
EnhancedScalpingDashboard: Any = None
DemoTradingBot: Any = None
BinanceStreamingAPI: Any = None
StreamingSignalProcessor: Any = None
PredictionMetrics: Any = None
EnhancedSignal: Any = None
SignalQuality: Any = None
ScalpingConfig: Any = None
SignalFormatter: Any = None

try:
    from crypto_bot.core.enhanced_crypto_dashboard import (
        EnhancedScalpingDashboard as _EnhancedScalpingDashboard,
        DemoTradingBot as _DemoTradingBot,
        BinanceStreamingAPI as _BinanceStreamingAPI,
        StreamingSignalProcessor as _StreamingSignalProcessor,
        PredictionMetrics as _PredictionMetrics,
        EnhancedSignal as _EnhancedSignal,
        SignalQuality as _SignalQuality,
        ScalpingConfig as _ScalpingConfig,
        SignalFormatter as _SignalFormatter,
    )

    EnhancedScalpingDashboard = _EnhancedScalpingDashboard
    DemoTradingBot = _DemoTradingBot
    BinanceStreamingAPI = _BinanceStreamingAPI
    StreamingSignalProcessor = _StreamingSignalProcessor
    PredictionMetrics = _PredictionMetrics
    EnhancedSignal = _EnhancedSignal
    SignalQuality = _SignalQuality
    ScalpingConfig = _ScalpingConfig
    SignalFormatter = _SignalFormatter

    DASHBOARD_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Dashboard import error: {e}")
    DASHBOARD_AVAILABLE = False

# -----------------------------------------------------------------------------
# Service imports (always bind names to avoid Pylance "unbound")
# -----------------------------------------------------------------------------
SERVICES_AVAILABLE = False
MarketDataService: Any = None
SignalEngineService: Any = None
SignalRepository: Any = None
SignalOrchestrator: Any = None
ProfessionalAnalyzer: Any = None

try:
    from crypto_bot.services.market_data_service import MarketDataService as _MarketDataService
    from crypto_bot.services.signal_engine_service import SignalEngineService as _SignalEngineService
    from crypto_bot.repositories.signal_repository import SignalRepository as _SignalRepository
    from crypto_bot.services.signal_orchestrator import SignalOrchestrator as _SignalOrchestrator
    from crypto_bot.analyzers.professional_analyzer import ProfessionalAnalyzer as _ProfessionalAnalyzer

    MarketDataService = _MarketDataService
    SignalEngineService = _SignalEngineService
    SignalRepository = _SignalRepository
    SignalOrchestrator = _SignalOrchestrator
    ProfessionalAnalyzer = _ProfessionalAnalyzer

    SERVICES_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Services import error: {e}")
    SERVICES_AVAILABLE = False

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float even if it is list/tuple/None/str."""
    try:
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            value = value[0]
        return float(value)
    except (TypeError, ValueError):
        return default


def serialize_signal(obj: Any) -> Any:
    """Make signal JSON-safe. Converts datetime to ISO 8601 with Z suffix."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        # Return ISO format with Z suffix (UTC indicator)
        iso_str = obj.isoformat()
        if not iso_str.endswith('Z'):
            iso_str = iso_str + 'Z'
        return iso_str
    if isinstance(obj, dict):
        # Make sure any datetime is isoformat with Z
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, datetime):
                iso_str = v.isoformat()
                if not iso_str.endswith('Z'):
                    iso_str = iso_str + 'Z'
                out[k] = iso_str
            elif isinstance(v, (list, tuple)):
                out[k] = [serialize_signal(x) for x in v]
            else:
                out[k] = serialize_signal(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [serialize_signal(x) for x in obj]
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return serialize_signal(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return serialize_signal(dict(obj.__dict__))
        except Exception:
            pass
    return str(obj)

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR) if STATIC_DIR.exists() else None,
    static_url_path="/static",
)

CORS(app, supports_credentials=True)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
if not DISCORD_WEBHOOK_URL:
    logger.warning("⚠️ DISCORD_WEBHOOK_URL not set. Discord notifications disabled.")

# CORS preflight
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, Access-Control-Request-Private-Network"
        )
        response.headers["Access-Control-Max-Age"] = "86400"
        return response

@app.after_request
def add_private_network_headers(response):
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, Access-Control-Request-Private-Network"
    )
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# -----------------------------------------------------------------------------
# SocketIO
# Use threading mode for Windows stability (no eventlet)
socketio = None
try:
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="eventlet",
        ping_timeout=60,
        ping_interval=25,
        engineio_logger=False,
        logger=False,
    )

    logger.info(f"✅ SocketIO initialized with async_mode='threading'")
except Exception as e:
    logger.error(f"❌ Failed to initialize SocketIO: {e}")
    socketio = None

# Predictions API blueprint disabled - advanced_web_server.py /api/predictions
# provides superior implementation reading directly from signal_repo

# =============================================================================
# Trading Symbols (loaded from config)
# =============================================================================
# Load symbols from config
try:
    import json as _json
    _coins_config = _json.load(open(PROJECT_ROOT / "config" / "coins.json"))
    SYMBOLS = _coins_config.get("symbols", [])
except Exception:
    # Fallback if config not found
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "UNIUSDT",
        "LINKUSDT", "XLMUSDT", "ATOMUSDT", "MANAUSDT", "SANDUSDT",
        "DASHUSDT", "VETUSDT", "ICPUSDT", "GMTUSDT", "PEOPLEUSDT",
        "LUNCUSDT", "CHZUSDT", "NEARUSDT", "FLOWUSDT", "FILUSDT",
        "QTUMUSDT", "SNXUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT",
        "FLOKIUSDT", "OPUSDT",
    ]

# =============================================================================
# Service instances
# =============================================================================
market_data_service: Any = None
signal_engine_service: Any = None
signal_repo: Any = None
signal_orchestrator: Any = None
dashboard: Any = None  # Global dashboard instance for services

_services_lock = threading.Lock()
_services_started = False
_services_started_at: Optional[datetime] = None
_last_services_error: Optional[str] = None


def init_services() -> bool:
    """Initialize all background services for live trading signals."""
    global market_data_service, signal_engine_service, signal_repo, signal_orchestrator, dashboard
    global _services_started, _services_started_at, _last_services_error

    with _services_lock:
        if _services_started:
            return True

        if not SERVICES_AVAILABLE:
            _last_services_error = "SERVICES_AVAILABLE=False (imports failed)"
            logger.warning("⚠️ Services module import failed - skipping service initialization")
            return False

        logger.info("🔧 Initializing background services...")
        initialized_count = 0

        # 1) Market data service
        try:
            if market_data_service is None:
                use_ws = os.getenv("USE_BINANCE_WS", "true").lower() in {"1", "true", "yes", "y", "on"}
                market_data_service = MarketDataService(SYMBOLS, use_websocket=use_ws)
                # Support both names:
                if hasattr(market_data_service, "start_websocket"):
                    market_data_service.start_websocket()
                elif hasattr(market_data_service, "start"):
                    market_data_service.start()
                logger.info("✅ Market data service initialized")
                initialized_count += 1
        except Exception as e:
            _last_services_error = f"Market data init failed: {e}"
            logger.error(f"❌ Market data service init failed: {e}", exc_info=True)

        # 2) Signal repository
        try:
            if signal_repo is None:
                db_path = os.getenv("SIGNALS_DB_PATH", "data/signals.db")
                use_sqlite = os.getenv("USE_SQLITE", "true").lower() in {"1", "true", "yes", "y", "on"}
                signal_repo = SignalRepository(use_sqlite=use_sqlite, db_path=db_path)
                logger.info("✅ Signal repository initialized")
                initialized_count += 1
        except Exception as e:
            _last_services_error = f"Signal repository init failed: {e}"
            logger.error(f"❌ Signal repository init failed: {e}", exc_info=True)

        # 3) Enhanced dashboard (core ML)
        try:
            if dashboard is None and DASHBOARD_AVAILABLE:
                dashboard = EnhancedScalpingDashboard(
                    use_streaming_ml=True,
                    enable_demo_bot=True,
                    use_binance_testnet=False,
                )
                logger.info("✅ EnhancedScalpingDashboard initialized")
                initialized_count += 1
        except Exception as e:
            _last_services_error = f"Dashboard init failed: {e}"
            logger.error(f"❌ Dashboard init failed: {e}", exc_info=True)

        # 4) Signal engine
        try:
            if signal_engine_service is None and dashboard is not None and market_data_service is not None:
                # Initialize ProfessionalAnalyzer with config
                pro_analyzer = None
                try:
                    if ProfessionalAnalyzer:
                        # Create config dict from settings
                        analyzer_config = {
                            'MIN_CONFLUENCE_SCORE': 60,
                            'LEVERAGE': 10,
                            'MAX_RISK_PER_TRADE': MIN_ACCURACY / 100.0,  # Convert percentage to decimal
                            'RSI_OVERBOUGHT': 70,
                            'RSI_OVERSOLD': 30,
                            'VOLUME_SURGE_REQUIREMENT': 1.5,
                            'CONFLUENCE_WEIGHTS': {
                                'trend': 0.25,
                                'price_action': 0.20,
                                'indicators': 0.15,
                                'volume': 0.15,
                                'order_book': 0.15,
                                'multi_timeframe': 0.10
                            },
                            'BANKROLL': 10000,
                            'MIN_CATEGORY_SCORES': {
                                'trend': 0.6,
                                'price_action': 0.6,
                                'indicators': 0.6
                            },
                            'MIN_ATR_PERCENTAGE': 0.0015,
                            'SL_ATR_MULTIPLIER': 3.5,
                            'MIN_SL_PERCENTAGE': 0.0025,
                            'POSITION_SIZING': 'fixed'
                        }
                        pro_analyzer = ProfessionalAnalyzer(config=analyzer_config)
                        logger.info("✅ ProfessionalAnalyzer initialized with config")
                except Exception as pa_err:
                    logger.warning(f"⚠️ ProfessionalAnalyzer init failed: {pa_err}")
                
                signal_engine_service = SignalEngineService(
                    market_data=market_data_service,
                    enhanced_dashboard=dashboard,
                    professional_analyzer=pro_analyzer,
                )
                logger.info("✅ Signal engine service initialized")
                initialized_count += 1
        except Exception as e:
            _last_services_error = f"Signal engine init failed: {e}"
            logger.error(f"❌ Signal engine init failed: {e}", exc_info=True)

        # 5) Orchestrator
        try:
            if signal_orchestrator is None and signal_engine_service is not None and signal_repo is not None:
                signal_orchestrator = SignalOrchestrator(
                    signal_engine=signal_engine_service,
                    signal_repo=signal_repo,
                    socketio=socketio,
                    refresh_interval_secs=SIGNAL_REFRESH_INTERVAL,
                )
                # Support both signatures: start(symbols) or start()
                if hasattr(signal_orchestrator, "start"):
                    try:
                        signal_orchestrator.start(SYMBOLS)
                    except TypeError:
                        signal_orchestrator.start()
                logger.info(f"✅ Signal orchestrator started ({SIGNAL_REFRESH_INTERVAL}s refresh)")
                initialized_count += 1
        except Exception as e:
            _last_services_error = f"Signal orchestrator init failed: {e}"
            logger.error(f"❌ Signal orchestrator init failed: {e}", exc_info=True)

        _services_started = initialized_count > 0
        _services_started_at = datetime.now() if _services_started else None
        logger.info(f"✅ Service initialization complete ({initialized_count}/5 services initialized)")
        return _services_started


@app.before_request
def _startup_services_once():
    # When running via main.py (importing app/socketio), this ensures services still start.
    try:
        init_services()
    except Exception as e:
        logger.error(f"❌ init_services() failed on startup hook: {e}", exc_info=True)


# =============================================================================
# Legacy bot_state (kept for backwards-compat routes)
# =============================================================================
bot_state: Dict[str, Any] = {
    "running": False,
    "dashboard": None,
    "bot": None,
    "thread": None,
    "signals": [],
    "portfolio_history": [],
    "trade_history": [],
    "start_time": None,
    "stats": {
        "total_signals": 0,
        "total_trades": 0,
        "active_trades": 0,
        "pnl": 0.0,
        "pnl_percent": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
    },
}

update_queue: queue.Queue[Any] = queue.Queue()

def init_bot() -> bool:
    """Initialize legacy dashboard and bot"""
    if bot_state["dashboard"] is None:
        try:
            if DASHBOARD_AVAILABLE and EnhancedScalpingDashboard is not None:
                bot_state["dashboard"] = EnhancedScalpingDashboard(
                    use_streaming_ml=True,
                    enable_demo_bot=True,
                    use_binance_testnet=False,
                )
                logger.info("✅ Legacy Dashboard initialized")
                return True
            logger.error("❌ Dashboard modules not available")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize legacy dashboard: {e}", exc_info=True)
            return False
    return True

def run_bot_loop():
    """Legacy bot loop"""
    bot_state["running"] = True
    bot_state["start_time"] = datetime.now()
    iteration = 0

    while bot_state["running"]:
        iteration += 1
        try:
            dash = bot_state["dashboard"]
            if dash and getattr(dash, "demo_bot", None):
                signals = []
                if hasattr(dash, "streaming_processor") and dash.streaming_processor:
                    try:
                        logger.info(f"[STEP 2] Calling streaming_processor.process_symbols_batch()...")
                        signals = dash.streaming_processor.process_symbols_batch()
                        logger.info(f"[STEP 2] ✅ Got {len(signals)} signals from streaming processor")
                    except TypeError as te:
                        logger.error(f"[STEP 2] ❌ TypeError (signature mismatch?): {str(te)[:150]}", exc_info=True)
                        signals = []
                    except Exception as e:
                        logger.error(f"[STEP 2] ❌ Error processing symbols: {type(e).__name__}: {str(e)[:150]}", exc_info=True)
                        signals = []

                bot_state["signals"] = [
                    {
                        "symbol": s.symbol,
                        "direction": s.direction,
                        "confidence": s.confidence,
                        "entry_price": s.entry_price,
                        "stop_loss": s.stop_loss,
                        "tp1": s.take_profit_1,
                        "tp2": s.take_profit_2,
                        "tp3": s.take_profit_3,
                        "timestamp": s.timestamp.isoformat() if hasattr(s, "timestamp") else datetime.now().isoformat(),
                        "quality": s.quality.value if hasattr(s, "quality") else "MEDIUM",
                    }
                    for s in signals
                ]

                if socketio is not None:
                    socketio.emit(
                        "bot_update",
                        {
                            "signals": (bot_state["signals"][:5] if bot_state["signals"] else []),
                            "stats": bot_state["stats"],
                            "portfolio_history": bot_state["portfolio_history"][-100:],
                            "running": bot_state["running"],
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                logger.info(f"🔄 Legacy Iteration {iteration}: {len(signals)} signals")

        except Exception as e:
            logger.error(f"❌ Legacy bot loop error: {e}", exc_info=True)

        time.sleep(30)

# =============================================================================
# WEB ROUTES
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

# =============================================================================
# Binance chart + stats (existing)
# =============================================================================
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")

_INTERVAL_MAP = {
    "1M": "1m", "3M": "3m", "5M": "5m", "15M": "15m", "30M": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h", "6H": "6h", "8H": "8h", "12H": "12h",
    "1D": "1d", "3D": "3d", "1W": "1w", "1MO": "1M",
}

def normalize_interval(iv: str) -> str:
    if not iv:
        return "15m"
    iv = iv.strip()
    if iv in _INTERVAL_MAP:
        return _INTERVAL_MAP[iv]
    return iv.lower()

@app.route("/api/chart/<symbol>", methods=["GET"])
def api_chart(symbol: str):
    try:
        interval = normalize_interval(request.args.get("interval", "15m"))
        limit_raw = request.args.get("limit", "500")
        try:
            limit = int(limit_raw)
        except ValueError:
            limit = 500
        limit = max(1, min(limit, 1000))

        url = f"{BINANCE_BASE_URL}/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        klines = r.json()

        candles = []
        for k in klines:
            ts = int(int(k[0]) // 1000)
            candles.append(
                {
                    "time": ts,                # lightweight-charts compatible
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                }
            )

        return jsonify({"success": True, "symbol": symbol.upper(), "interval": interval, "limit": limit, "candles": candles})

    except requests.HTTPError as e:
        return jsonify({"success": False, "error": f"Binance HTTP error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/stats/<symbol>", methods=["GET"])
def api_symbol_stats(symbol: str):
    try:
        sym = symbol.upper()
        url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
        r = requests.get(url, params={"symbol": sym}, timeout=10)
        r.raise_for_status()
        t = r.json()

        last_price = to_float(t.get("lastPrice"), 0.0)
        price_change = to_float(t.get("priceChange"), 0.0)
        price_change_percent = to_float(t.get("priceChangePercent"), 0.0)
        high_price = to_float(t.get("highPrice"), 0.0)
        low_price = to_float(t.get("lowPrice"), 0.0)
        volume = to_float(t.get("volume"), 0.0)

        # legacy memory signals
        sym_signals = [s for s in bot_state.get("signals", []) if s.get("symbol") == sym]
        avg_conf = 0.0
        if sym_signals:
            avg_conf = sum(to_float(s.get("confidence"), 0.0) for s in sym_signals) / max(len(sym_signals), 1)

        return jsonify(
            {
                "success": True,
                "symbol": sym,
                "market": {
                    "last_price": last_price,
                    "change_24h": price_change,
                    "change_24h_percent": price_change_percent,
                    "high_24h": high_price,
                    "low_24h": low_price,
                    "volume_24h": volume,
                },
                "signals": {
                    "count": len(sym_signals),
                    "avg_confidence": avg_conf,
                    "latest": sym_signals[-1] if sym_signals else None,
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    except requests.HTTPError as e:
        return jsonify({"success": False, "error": f"Binance HTTP error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =============================================================================
# NEW: Predictions + Prices + Health (services pipeline)
# =============================================================================
@app.route("/api/health", methods=["GET"])
def api_health():
    init_services()  # best-effort
    cache_count = 0
    try:
        if signal_repo is not None and hasattr(signal_repo, "get_latest_all"):
            cached = signal_repo.get_latest_all()
            cache_count = len(cached or {})
    except Exception:
        cache_count = 0

    return jsonify(
        {
            "success": True,
            "status": "ok",
            "services_available": SERVICES_AVAILABLE,
            "dashboard_available": DASHBOARD_AVAILABLE,
            "services_started": _services_started,
            "services_started_at": _services_started_at.isoformat() if _services_started_at else None,
            "last_services_error": _last_services_error,
            "symbols_count": len(SYMBOLS),
            "cached_predictions": cache_count,
            "timestamp": datetime.now().isoformat(),
        }
    ), 200

# Prediction mapper (if not available in crypto_bot.mappers)
class PredictionMapper:
    @staticmethod
    def _to_iso_string(dt: datetime) -> str:
        iso_str = dt.isoformat()
        if not iso_str.endswith('Z'):
            iso_str += 'Z'
        return iso_str
    
    @staticmethod
    def _calculate_take_profits(entry: float, stop_loss: float, direction: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate TP1, TP2, TP3 using R-multiple system.
        
        Formula:
            risk = abs(entry - stop_loss)
            For LONG: TP = entry + k*risk (k=1,2,3)
            For SHORT: TP = entry - k*risk (k=1,2,3)
        
        Returns:
            {
                'tp1': float, 'tp2': float, 'tp3': float,
                'tp1_eta': str, 'tp2_eta': str, 'tp3_eta': str,
                'take_profits': [array]
            }
        """
        try:
            entry = float(entry) if entry else 0
            stop_loss = float(stop_loss) if stop_loss else entry
            
            if entry == 0 or stop_loss == 0 or entry == stop_loss:
                logger.warning(f"Invalid TP inputs: entry={entry}, stop_loss={stop_loss}")
                return {
                    'tp1': 0, 'tp2': 0, 'tp3': 0,
                    'tp1_eta': None, 'tp2_eta': None, 'tp3_eta': None,
                    'take_profits': []
                }
            
            # Calculate risk distance (1R)
            risk = abs(entry - stop_loss)
            
            # Calculate TPs based on direction
            if direction.upper() == "LONG":
                tp1 = entry + risk
                tp2 = entry + (2 * risk)
                tp3 = entry + (3 * risk)
            elif direction.upper() == "SHORT":
                tp1 = entry - risk
                tp2 = entry - (2 * risk)
                tp3 = entry - (3 * risk)
            else:
                # NO_TRADE or unknown
                tp1 = tp2 = tp3 = entry
            
            # Round to 8 decimals (crypto precision)
            tp1 = round(tp1, 8)
            tp2 = round(tp2, 8)
            tp3 = round(tp3, 8)
            
            # Estimate ETAs (simple: 30 min per TP level from now)
            if timestamp is None:
                timestamp = datetime.utcnow()
            elif isinstance(timestamp, str):
                try:
                    # Remove 'Z' suffix and replace with UTC offset for fromisoformat compatibility
                    iso_str = timestamp.rstrip('Z')
                    if iso_str.endswith('Z'):
                        iso_str = iso_str[:-1] + '+00:00'
                    elif not iso_str[-6] in ('+', '-'):
                        iso_str = iso_str + '+00:00'
                    timestamp = datetime.fromisoformat(iso_str)
                except:
                    timestamp = datetime.utcnow()
            
            tp1_eta = PredictionMapper._to_iso_string(timestamp + timedelta(minutes=30))
            tp2_eta = PredictionMapper._to_iso_string(timestamp + timedelta(minutes=60))
            tp3_eta = PredictionMapper._to_iso_string(timestamp + timedelta(minutes=120))
            
            # Create take_profits array for backward compatibility
            take_profits = [
                {"level": 1, "price": tp1, "eta": tp1_eta},
                {"level": 2, "price": tp2, "eta": tp2_eta},
                {"level": 3, "price": tp3, "eta": tp3_eta},
            ]
            
            return {
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'tp1_eta': tp1_eta,
                'tp2_eta': tp2_eta,
                'tp3_eta': tp3_eta,
                'take_profits': take_profits
            }
            
        except Exception as e:
            logger.error(f"Error calculating TPs: {e}")
            return {
                'tp1': 0, 'tp2': 0, 'tp3': 0,
                'tp1_eta': None, 'tp2_eta': None, 'tp3_eta': None,
                'take_profits': []
            }
    
    @staticmethod
    def from_signal_model(signal):
        """Convert signal model to unified prediction schema with TPs"""
        if signal is None:
            return None
            
        try:
            # Get confidence and accuracy
            confidence = getattr(signal, 'confidence_score', 0)
            accuracy = getattr(signal, 'accuracy_percent', 0)
            
            # Determine status based on thresholds
            if MIN_CONFIDENCE > 0 and confidence < MIN_CONFIDENCE:
                status = "filtered"
            elif MIN_ACCURACY > 0 and accuracy < MIN_ACCURACY:
                status = "filtered"
            else:
                status = "active"
            
            # Get core trading data
            entry_price = getattr(signal, 'entry_price', 0)
            stop_loss = getattr(signal, 'stop_loss', 0)
            direction = getattr(signal, 'direction', 'NEUTRAL')
            timestamp = getattr(signal, 'timestamp', datetime.utcnow())
            
            # Calculate TPs
            tp_data = PredictionMapper._calculate_take_profits(entry_price, stop_loss, direction, timestamp)
            
            # Create unified prediction object
            pred = {
                "symbol": getattr(signal, 'symbol', 'UNKNOWN'),
                "direction": direction,
                "confidence": confidence,
                "accuracy": accuracy,
                "status": status,
                "timestamp": PredictionMapper._to_iso_string(timestamp),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                # Take profit fields (R-multiple system)
                "tp1": tp_data['tp1'],
                "tp2": tp_data['tp2'],
                "tp3": tp_data['tp3'],
                "tp1_eta": tp_data['tp1_eta'],
                "tp2_eta": tp_data['tp2_eta'],
                "tp3_eta": tp_data['tp3_eta'],
                "take_profits": tp_data['take_profits'],
                # Legacy field
                "take_profit": tp_data['tp1'],
                "source": getattr(signal, 'source', 'unknown'),
                "reason": getattr(signal, 'reason', '')
            }
            
            # Add any additional fields
            for field in ['quality', 'risk_level', 'timeframe', 'patterns']:
                if hasattr(signal, field):
                    pred[field] = getattr(signal, field)
                    
            return pred
            
        except Exception as e:
            logger.error(f"Error mapping signal: {e}")
            return None
    
    @staticmethod
    def validate_prediction(pred):
        """Validate prediction schema"""
        if not pred or not isinstance(pred, dict):
            return False
        required = ['symbol', 'direction', 'confidence', 'timestamp', 'tp1', 'tp2', 'tp3']
        return all(key in pred for key in required)

@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    init_services()  # best-effort
    try:
        cached = {}
        if signal_repo is not None and hasattr(signal_repo, "get_latest_all"):
            cached = signal_repo.get_latest_all() or {}
            logger.info(f"📡 GET PREDICTIONS: Retrieved {len(cached)} signals from repo")
            
            if len(cached) == 0:
                logger.warning("⚠️  Cache empty - attempting to reload from DB...")
                # Try to reload from database
                if hasattr(signal_repo, "_load_from_db"):
                    try:
                        signal_repo._load_from_db()
                        cached = signal_repo.get_latest_all() or {}
                        logger.info(f"📡 Reloaded from DB: {len(cached)} signals")
                    except Exception as reload_err:
                        logger.warning(f"⚠️ DB reload failed: {reload_err}")
            else:
                logger.info(f"📊 Signals in repo: {list(cached.keys())}")

        # Map signal objects to unified prediction schema
        predictions = {}
        filtered_predictions = {}
        
        for symbol, signal in cached.items():
            if signal is None:
                logger.warning(f"Signal for {symbol} is None")
                continue
            
            try:
                pred_obj = PredictionMapper.from_signal_model(signal)
                
                if pred_obj is not None and PredictionMapper.validate_prediction(pred_obj):
                    confidence = signal.confidence_score
                    accuracy = signal.accuracy_percent
                    filtered_reason = None
                    
                    # Check filtering logic - but track WHY filtered
                    if MIN_CONFIDENCE > 0 and confidence < MIN_CONFIDENCE:
                        filtered_reason = f"confidence({confidence}%) < MIN_CONFIDENCE({MIN_CONFIDENCE}%)"
                    elif MIN_ACCURACY > 0 and accuracy < MIN_ACCURACY:
                        filtered_reason = f"accuracy({accuracy:.1f}%) < MIN_ACCURACY({MIN_ACCURACY}%)"
                    
                    pred_obj["raw_confidence"] = confidence
                    pred_obj["raw_accuracy"] = accuracy
                    pred_obj["source"] = signal.source if hasattr(signal, 'source') else "UNKNOWN"
                    
                    if filtered_reason:
                        # IMPORTANT: Still return it, but mark as filtered
                        pred_obj["filtered_out_reason"] = filtered_reason
                        pred_obj["status"] = "filtered"
                        # Add to BOTH lists so UI can show it
                        filtered_predictions[symbol] = pred_obj
                        predictions[symbol] = pred_obj  # Keep in main list too for UI
                        logger.info(f"⊘ Filtered {symbol}: {filtered_reason}")
                    else:
                        predictions[symbol] = pred_obj
                        logger.debug(f"✅ Mapped {symbol} to prediction")
                else:
                    logger.warning(f"Invalid prediction schema for {symbol}")
            except Exception as map_err:
                logger.error(f"Error mapping signal for {symbol}: {map_err}")
                continue

        # FALLBACK: If still empty, try InferenceService
        if len(predictions) == 0 and SERVICES_AVAILABLE:
            logger.warning("⚠️  No predictions in repo - attempting InferenceService fallback...")
            try:
                from crypto_bot.ml.inference.inference_service import InferenceService
                inference_service = InferenceService(device="cpu")
                
                for symbol in SYMBOLS[:10]:  # Top 10 symbols as fallback
                    try:
                        pred = inference_service.predict(symbol, "15m")
                        if pred:
                            predictions[symbol] = pred.to_dict() if hasattr(pred, 'to_dict') else pred
                            logger.info(f"✅ Fallback prediction for {symbol}")
                    except Exception as inf_err:
                        logger.debug(f"Inference fallback for {symbol} failed: {inf_err}")
                        
                if predictions:
                    logger.info(f"✅ InferenceService fallback generated {len(predictions)} predictions")
            except Exception as fallback_err:
                logger.warning(f"⚠️ InferenceService fallback failed: {fallback_err}")

        logger.info(f"✅ API RETURNING: {len(predictions)} predictions (filtered: {len(filtered_predictions)})")
        
        # Always return 200 with whatever we have
        return jsonify(
            {
                "success": True,
                "predictions": predictions,
                "filtered_predictions": filtered_predictions,
                "count": len(predictions),
                "filtered_count": len(filtered_predictions),
                "warming_up": (len(predictions) == 0),
                "dev_thresholds": {
                    "MIN_CONFIDENCE": MIN_CONFIDENCE,
                    "MIN_ACCURACY": MIN_ACCURACY
                },
                "timestamp": PredictionMapper._to_iso_string(datetime.utcnow()),
            }
        ), 200
    except Exception as e:
        logger.error(f"/api/predictions error: {e}", exc_info=True)
        return jsonify(
            {
                "success": True,
                "predictions": {},
                "filtered_predictions": {},
                "count": 0,
                "filtered_count": 0,
                "warming_up": True,
                "error": str(e),
                "timestamp": PredictionMapper._to_iso_string(datetime.utcnow()),
            }
        ), 200

@app.route("/api/predictions/<symbol>", methods=["GET"])
def api_predictions_symbol(symbol: str):
    init_services()
    sym = symbol.upper()
    try:
        item = None
        if signal_repo is not None and hasattr(signal_repo, "get_latest"):
            item = signal_repo.get_latest(sym)
        elif signal_repo is not None and hasattr(signal_repo, "get_latest_all"):
            all_ = signal_repo.get_latest_all() or {}
            item = all_.get(sym)

        prediction = None
        filtered_reason = None
        
        if item:
            try:
                # Use PredictionMapper to normalize to unified schema
                prediction = PredictionMapper.from_signal_model(item)
                
                if prediction is not None:
                    # Add raw values for transparency
                    confidence = item.confidence_score
                    accuracy = item.accuracy_percent
                    
                    prediction["raw_confidence"] = confidence
                    prediction["raw_accuracy"] = accuracy
                    prediction["source"] = item.source if hasattr(item, 'source') else "UNKNOWN"
                    
                    # Check if would be filtered
                    if MIN_CONFIDENCE > 0 and confidence < MIN_CONFIDENCE:
                        filtered_reason = f"confidence({confidence}%) < MIN_CONFIDENCE({MIN_CONFIDENCE}%)"
                    elif MIN_ACCURACY > 0 and accuracy < MIN_ACCURACY:
                        filtered_reason = f"accuracy({accuracy:.1f}%) < MIN_ACCURACY({MIN_ACCURACY}%)"
                    else:
                        filtered_reason = None
                    
                    if filtered_reason:
                        prediction["filtered_out_reason"] = filtered_reason
                        prediction["status"] = "filtered"
                    
            except Exception as map_err:
                logger.error(f"Error mapping signal for {sym}: {map_err}")

        return jsonify(
            {
                "success": True,
                "symbol": sym,
                "prediction": prediction,
                "dev_thresholds": {
                    "MIN_CONFIDENCE": MIN_CONFIDENCE,
                    "MIN_ACCURACY": MIN_ACCURACY
                },
                "timestamp": PredictionMapper._to_iso_string(datetime.utcnow())
            }
        ), 200
    except Exception as e:
        logger.error(f"/api/predictions/{sym} error: {e}", exc_info=True)
        # Never return 503
        return jsonify(
            {
                "success": True,
                "symbol": sym,
                "prediction": None,
                "timestamp": PredictionMapper._to_iso_string(datetime.utcnow())
            }
        ), 200

# Compatibility: singular endpoints used by your template JS
@app.route("/api/prediction", methods=["GET"])
def api_prediction_alias():
    return api_predictions()

@app.route("/api/prediction/<symbol>", methods=["GET"])
def api_prediction_symbol_alias(symbol: str):
    return api_predictions_symbol(symbol)

# Backwards compatibility: /api/signals returns predictions as array
@app.route("/api/signals", methods=["GET"])
def api_signals_compat():
    """Backwards compatibility: /api/signals returns Object.values(predictions)"""
    try:
        cached = {}
        if signal_repo is not None and hasattr(signal_repo, "get_latest_all"):
            cached = signal_repo.get_latest_all() or {}
        
        signals_array = []
        for symbol, signal in cached.items():
            if signal is None:
                continue
            try:
                # Use PredictionMapper to normalize to unified schema
                pred_obj = PredictionMapper.from_signal_model(signal)
                signals_array.append(pred_obj)
            except Exception as map_err:
                logger.error(f"Error mapping signal for {symbol}: {map_err}")
                continue
        
        return jsonify({
            "success": True,
            "signals": signals_array,
            "count": len(signals_array),
            "warming_up": (len(signals_array) == 0),
            "timestamp": PredictionMapper._to_iso_string(datetime.utcnow()),
        }), 200
    except Exception as e:
        logger.error(f"/api/signals error: {e}", exc_info=True)
        # Never return 503
        return jsonify({
            "success": True,
            "signals": [],
            "count": 0,
            "warming_up": True,
            "timestamp": PredictionMapper._to_iso_string(datetime.utcnow()),
        }), 200

@app.route("/api/prices", methods=["GET"])
def api_prices():
    init_services()
    symbol = (request.args.get("symbol") or "").strip().upper()
    try:
        if market_data_service is not None:
            # Support a few possible method names
            if symbol:
                if hasattr(market_data_service, "get_price"):
                    p = market_data_service.get_price(symbol)
                else:
                    p = None
                return jsonify({"success": True, "symbol": symbol, "price": serialize_signal(p), "timestamp": datetime.now().isoformat()}), 200

            if hasattr(market_data_service, "get_all_prices"):
                prices = market_data_service.get_all_prices()
            elif hasattr(market_data_service, "get_prices"):
                prices = market_data_service.get_prices()
            else:
                prices = {}

            return jsonify({"success": True, "prices": serialize_signal(prices), "timestamp": datetime.now().isoformat()}), 200

        # Fallback: Binance REST for a single symbol
        if symbol:
            r = requests.get(f"{BINANCE_BASE_URL}/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            j = r.json()
            return jsonify({"success": True, "symbol": symbol, "price": to_float(j.get("price"), 0.0), "timestamp": datetime.now().isoformat()}), 200

        return jsonify({"success": True, "prices": {}, "timestamp": datetime.now().isoformat(), "note": "market_data_service not available"}), 200

    except Exception as e:
        logger.error(f"/api/prices error: {e}", exc_info=True)
        return jsonify({"success": False, "prices": {}, "error": str(e), "timestamp": datetime.now().isoformat()}), 200

@app.route("/api/prices/<symbol>", methods=["GET"])
def api_prices_symbol(symbol: str):
    return api_prices()

@app.route("/api/price", methods=["GET"])
def api_price_alias():
    return api_prices()

@app.route("/api/price/<symbol>", methods=["GET"])
def api_price_symbol_alias(symbol: str):
    return api_prices()

# =============================================================================
# Legacy bot endpoints (kept)
# =============================================================================
@app.route("/api/bot/status")
def get_bot_status():
    start_time = bot_state["start_time"]
    return jsonify(
        {
            "running": bot_state["running"],
            "start_time": start_time.isoformat() if start_time else None,
            "uptime_seconds": (datetime.now() - start_time).total_seconds() if start_time else 0,
            "stats": bot_state["stats"],
            "signals_count": len(bot_state["signals"]),
            "portfolio_history_points": len(bot_state["portfolio_history"]),
        }
    )

@app.route("/api/bot/start", methods=["POST"])
def start_bot():
    if bot_state["running"]:
        return jsonify({"error": "Bot is already running"}), 400

    try:
        if not init_bot():
            return jsonify({"error": "Failed to initialize bot"}), 500

        bot_state["running"] = True
        bot_state["start_time"] = datetime.now()

        bot_state["thread"] = threading.Thread(target=run_bot_loop, daemon=True)
        bot_state["thread"].start()

        if socketio is not None:
            socketio.emit("bot_started", {"timestamp": datetime.now().isoformat()})
        logger.info("✅ Bot started")
        return jsonify({"success": True, "message": "Bot started successfully"})

    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}", exc_info=True)
        bot_state["running"] = False
        return jsonify({"error": str(e)}), 500

@app.route("/api/bot/stop", methods=["POST"])
def stop_bot():
    if not bot_state["running"]:
        return jsonify({"error": "Bot is not running"}), 400

    try:
        bot_state["running"] = False
        if bot_state["thread"]:
            bot_state["thread"].join(timeout=5)
        logger.info("✅ Bot stopped")
        return jsonify({"success": True, "message": "Bot stopped successfully"})

    except Exception as e:
        logger.error(f"❌ Failed to stop bot: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/trades")
def get_trades():
    trades = []
    if bot_state["dashboard"] and getattr(bot_state["dashboard"], "demo_bot", None):
        bot = bot_state["dashboard"].demo_bot
        trades = [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.current_price,
                "pnl": t.pnl,
                "pnl_percentage": t.pnl_percentage,
                "status": t.status.value if hasattr(t, "status") else "COMPLETED",
                "entry_time": t.entry_time.isoformat() if hasattr(t, "entry_time") else "",
                "exit_time": t.exit_time.isoformat() if hasattr(t, "exit_time") and t.exit_time else "",
            }
            for t in getattr(bot, "completed_trades", [])
        ]
    return jsonify({"trades": trades, "count": len(trades)})

@app.route("/api/statistics")
def get_statistics():
    trades = []
    total_pnl = 0
    winning_trades = 0

    if bot_state["dashboard"] and getattr(bot_state["dashboard"], "demo_bot", None):
        bot = bot_state["dashboard"].demo_bot
        trades = getattr(bot, "completed_trades", [])
        winning_trades = sum(1 for t in trades if hasattr(t, "pnl") and t.pnl > 0)
        total_pnl = sum(getattr(t, "pnl", 0) for t in trades)

    win_rate = (winning_trades / len(trades) * 100) if len(trades) > 0 else 0
    start_time = bot_state["start_time"]
    uptime = (datetime.now() - start_time).total_seconds() if start_time else 0

    return jsonify(
        {
            "success": True,
            "statistics": {
                "win_rate": win_rate,
                "total_trades": len(trades),
                "winning_trades": winning_trades,
                "losing_trades": len(trades) - winning_trades,
                "total_pnl": total_pnl,
                "bot_running": bot_state["running"],
                "uptime_seconds": uptime,
            },
            "timestamp": datetime.now().isoformat(),
        }
    )

@app.route("/api/discord-notify", methods=["POST"])
def discord_notify():
    if not DISCORD_WEBHOOK_URL:
        return jsonify({"success": False, "error": "Discord webhook not configured"}), 400

    try:
        data = request.get_json() or {}
        symbol = data.get("symbol", "UNKNOWN")
        direction = data.get("direction", "N/A")
        entry_price = to_float(data.get("entry_price"), 0.0)
        stop_loss = to_float(data.get("stop_loss"), 0.0)
        confluence_score = to_float(data.get("confluence_score"), 0.0)

        take_profits = data.get("take_profits", [])
        tp_text = ""
        if isinstance(take_profits, list):
            for i, tp in enumerate(take_profits[:3], 1):
                tp_val = to_float(tp[0] if isinstance(tp, (list, tuple)) and tp else tp, 0.0)
                tp_text += f"TP{i}: ${tp_val:.2f}\n"

        embed = {
            "title": f"🎯 {symbol} - {direction}",
            "description": f"**Confluence Score:** {confluence_score:.0f}/100",
            "color": 65280 if direction == "LONG" else 16711680,
            "fields": [
                {"name": "📍 Entry Price", "value": f"${entry_price:.2f}", "inline": True},
                {"name": "🛑 Stop Loss", "value": f"${stop_loss:.2f}", "inline": True},
                {"name": "🎯 Take Profits", "value": tp_text or "N/A", "inline": False},
                {"name": "⏰ Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True},
            ],
            "footer": {"text": "CryptoTrader Pro Dashboard"},
        }

        payload = {"embeds": [embed]}
        headers = {"Content-Type": "application/json"}
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, headers=headers, timeout=5)

        if response.status_code == 204:
            logger.info(f"✅ Discord notification sent for {symbol}")
            return jsonify({"success": True, "message": f"Signal {symbol} sent to Discord"})
        return jsonify({"success": False, "error": f"Discord API error: {response.status_code}"}), 500

    except Exception as e:
        logger.error(f"❌ Discord notification error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/download/csv", methods=["GET"])
def download_csv():
    try:
        trades = []
        if bot_state["dashboard"] and getattr(bot_state["dashboard"], "demo_bot", None):
            bot = bot_state["dashboard"].demo_bot
            trades = getattr(bot, "completed_trades", [])

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Symbol", "Direction", "Entry Price", "Exit Price", "PnL", "PnL %", "Status", "Entry Time", "Exit Time", "Duration"])

        for trade in trades:
            entry_time = getattr(trade, "entry_time", datetime.now())
            exit_time = getattr(trade, "exit_time", datetime.now())
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time)
                except Exception:
                    entry_time = datetime.now()
            if isinstance(exit_time, str):
                try:
                    exit_time = datetime.fromisoformat(exit_time)
                except Exception:
                    exit_time = datetime.now()

            duration = exit_time - entry_time if exit_time and entry_time else timedelta(0)
            duration_str = str(duration).split(".")[0]

            writer.writerow(
                [
                    getattr(trade, "symbol", "N/A"),
                    getattr(trade, "direction", "N/A"),
                    f"${getattr(trade, 'entry_price', 0):.2f}",
                    f"${getattr(trade, 'current_price', 0):.2f}",
                    f"${getattr(trade, 'pnl', 0):.2f}",
                    f"{getattr(trade, 'pnl_percentage', 0):.2f}%",
                    getattr(trade, "status", "UNKNOWN"),
                    entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    exit_time.strftime("%Y-%m-%d %H:%M:%S") if exit_time else "",
                    duration_str,
                ]
            )

        output.seek(0)
        csv_bytes = BytesIO(output.getvalue().encode("utf-8"))
        return send_file(csv_bytes, mimetype="text/csv", as_attachment=True, download_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    except Exception as e:
        logger.error(f"❌ CSV export error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# =============================================================================
# WEBSOCKET EVENTS
# =============================================================================
if socketio is not None:
    @socketio.on("connect")
    def handle_connect():
        sid = getattr(request, "sid", "unknown")
        logger.info(f"🔌 Client connected: {sid}")
        emit("connection_response", {"data": "Connected to trading bot server", "timestamp": datetime.now().isoformat()})

    @socketio.on("disconnect")
    def handle_disconnect():
        sid = getattr(request, "sid", "unknown")
        logger.info(f"🔌 Client disconnected: {sid}")

    @socketio.on("request_status")
    def handle_status_request():
        emit("bot_status", {"running": bot_state["running"], "stats": bot_state["stats"], "timestamp": datetime.now().isoformat()})

    @socketio.on("start_bot")
    def handle_start_bot():
        if not bot_state["running"]:
            if init_bot():
                bot_state["thread"] = threading.Thread(target=run_bot_loop, daemon=True)
                bot_state["thread"].start()
                emit("bot_started", {"timestamp": datetime.now().isoformat()}, broadcast=True)
                logger.info("✅ Bot started via WebSocket")
        else:
            emit("error", {"message": "Bot already running"})

    @socketio.on("stop_bot")
    def handle_stop_bot():
        if bot_state["running"]:
            bot_state["running"] = False
            if bot_state["thread"]:
                bot_state["thread"].join(timeout=5)
            emit("bot_stopped", {"timestamp": datetime.now().isoformat()}, broadcast=True)
            logger.info("✅ Bot stopped via WebSocket")
        else:
            emit("error", {"message": "Bot is not running"})
else:
    logger.warning("⚠️ WebSocket event handlers not registered - socketio initialization failed")

# =============================================================================
# 404 HANDLER
# =============================================================================
@app.errorhandler(404)
def not_found(e):
    p = request.path or ""
    if p.startswith("/api") or p.startswith("/socket.io") or p.startswith("/static"):
        return jsonify({"error": "Not Found", "path": p}), 404
    return render_template("index.html"), 200

# =============================================================================
# MAIN (optional direct run)
# =============================================================================
def main():
    print("\n" + "=" * 80)
    print("🚀 ADVANCED CRYPTO TRADING BOT - WEB SERVER (TEMPLATES MODE)".center(80))
    print("=" * 80)
    print("\n📄 UI: http://localhost:5000")
    print("📄 Health: http://localhost:5000/api/health")
    print("📄 Predictions: http://localhost:5000/api/predictions")
    print("📄 Prices: http://localhost:5000/api/prices")
    print("🔌 WebSocket: ws://localhost:5000/socket.io")
    print("\n✅ Server starting...")
    
    # Log signal thresholds at startup
    logger.info(f"📊 Signal Thresholds Loaded: MIN_CONFIDENCE={MIN_CONFIDENCE}, MIN_ACCURACY={MIN_ACCURACY}%")
    print(f"📊 Signal Thresholds: MIN_CONFIDENCE={MIN_CONFIDENCE}, MIN_ACCURACY={MIN_ACCURACY}%")
    print("=" * 80 + "\n")

    init_services()  # ensure services are started even when running directly
    if not init_bot():
        print("❌ Failed to initialize legacy bot/dashboard (services pipeline may still work).")

    if socketio is None:
        logger.error("❌ SocketIO initialization failed. Cannot start server.")
        print("❌ SocketIO initialization failed. Cannot start server.")
        sys.exit(1)

    try:
        # Threading mode only (no eventlet dependency)
        socketio.run(
            app,
            host="0.0.0.0",
            port=int(os.getenv("SERVER_PORT", "5000")),
            debug=os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "y", "on"},
            use_reloader=False,
            log_output=True,
            ping_timeout=120,
            ping_interval=25,
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        bot_state["running"] = False
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()