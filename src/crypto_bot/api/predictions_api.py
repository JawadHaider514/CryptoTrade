"""API Routes for ML predictions with entry/SL/TP levels."""

import logging
from pathlib import Path
from flask import Blueprint, jsonify, request
from datetime import datetime, timezone, timedelta
from typing import Optional
import json

logger = logging.getLogger(__name__)

# Create blueprint
predictions_bp = Blueprint('predictions', __name__, url_prefix='/api')

PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

CONFIG_DIR = PROJECT_ROOT / "config"

# Lazy import to avoid torch initialization issues on app startup
_inference_service = None

def get_inference_service():
    """Lazy load InferenceService on first use"""
    global _inference_service
    if _inference_service is None:
        from crypto_bot.ml.inference.inference_service import InferenceService
        _inference_service = InferenceService(device="cpu")
    return _inference_service


def load_coins() -> list:
    """Load coin list from config."""
    try:
        coins_path = CONFIG_DIR / "coins.json"
        with open(coins_path) as f:
            config = json.load(f)
            return config.get('coins', [])
    except:
        return []


def calculate_entry_sl_tp(
    direction: str,
    current_price: float,
    confidence: float,
    atr: Optional[float] = None
) -> dict:
    """
    Calculate entry, stop loss, and take profit levels.
    
    Args:
        direction: LONG, SHORT, or NO_TRADE
        current_price: Current price
        confidence: Model confidence (0-1)
        atr: Average True Range (optional)
    
    Returns:
        Dict with entry, sl, tp1/2/3
    """
    if direction == "NO_TRADE":
        return {
            "entry": None,
            "stop_loss": None,
            "take_profits": []
        }
    
    # Use 1% of price as default ATR if not provided
    risk_unit = atr if atr else (current_price * 0.01)
    
    if direction == "LONG":
        entry = current_price
        stop_loss = entry - (risk_unit * 1.5)
        tp1 = entry + (risk_unit * 1.0)
        tp2 = entry + (risk_unit * 2.0)
        tp3 = entry + (risk_unit * 3.0)
    else:  # SHORT
        entry = current_price
        stop_loss = entry + (risk_unit * 1.5)
        tp1 = entry - (risk_unit * 1.0)
        tp2 = entry - (risk_unit * 2.0)
        tp3 = entry - (risk_unit * 3.0)
    
    return {
        "entry": round(entry, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profits": [
            {"level": 1, "price": round(tp1, 8)},
            {"level": 2, "price": round(tp2, 8)},
            {"level": 3, "price": round(tp3, 8)},
        ]
    }


def get_tp_eta(timeframe: str) -> int:
    """
    Estimate minutes until first TP is hit.
    
    Args:
        timeframe: Candle interval
    
    Returns:
        Minutes until TP
    """
    timeframe_minutes = {
        '1m': 5,
        '5m': 15,
        '15m': 45,
        '1h': 180,
        '4h': 360,
        '1d': 1440,
    }
    return timeframe_minutes.get(timeframe, 45)


@predictions_bp.route('/predictions', methods=['GET'])
def get_predictions():
    """
    Get predictions for all coins.
    
    Query params:
        - timeframe: 15m, 1h, 4h, 1d (default: 15m)
        - symbols: comma-separated list (default: all)
    
    Response: {
        "success": true,
        "predictions": {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "direction": "LONG",
                "confidence": 78.5,
                "accuracy_estimate": 87.3,
                "current_price": 45250.50,
                "entry_price": 45341.01,
                "stop_loss": 45135.50,
                "take_profits": [
                    {"level": 1, "price": 45796.01, "eta": "2025-12-28T..."},
                    {"level": 2, "price": 46468.01, "eta": "2025-12-28T..."},
                    {"level": 3, "price": 47340.50, "eta": "2025-12-28T..."}
                ],
                "timestamp": "2025-12-28T...",
                "valid_until": "2025-12-28T..."
            },
            ...
        },
        "count": 28,
        "warming_up": false,
        "timestamp": "2025-12-28T..."
    }
    """
    try:
        timeframe = request.args.get('timeframe', '15m').strip()
        symbols_param = request.args.get('symbols', '').strip()
        
        # Get symbols
        if symbols_param:
            symbols = [s.strip().upper() for s in symbols_param.split(',')]
        else:
            symbols = load_coins()
        
        if not symbols:
            return jsonify({
                "success": False,
                "message": "No symbols provided",
                "predictions": {},
                "count": 0,
                "warming_up": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200
        
        # Get predictions
        service = get_inference_service()
        predictions = {}
        
        for symbol in symbols:
            prediction = service.predict(symbol, timeframe)
            
            if prediction:
                # Use the frozen API contract from PredictionResult.to_dict()
                predictions[symbol] = prediction.to_dict()
            else:
                logger.warning(f"No prediction for {symbol} {timeframe}")
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "warming_up": (len(predictions) == 0),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": str(e),
            "predictions": {},
            "count": 0,
            "warming_up": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200  # Never return 500, always return 200 with success:false


@predictions_bp.route('/predictions/<symbol>', methods=['GET'])
def get_prediction_single(symbol: str):
    """
    Get prediction for a single symbol.
    
    Query params:
        - timeframe: 15m, 1h, 4h, 1d (default: 15m)
    
    Response: {
        "success": true,
        "symbol": "BTCUSDT",
        "prediction": {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "direction": "LONG",
            "confidence": 78.5,
            "accuracy_estimate": 87.3,
            "current_price": 45250.50,
            "entry_price": 45341.01,
            "stop_loss": 45135.50,
            "take_profits": [...],
            "timestamp": "2025-12-28T...",
            "valid_until": "2025-12-28T..."
        },
        "timestamp": "2025-12-28T..."
    }
    """
    try:
        timeframe = request.args.get('timeframe', '15m').strip()
        symbol = symbol.upper()
        
        service = get_inference_service()
        prediction = service.predict(symbol, timeframe)
        
        if not prediction:
            return jsonify({
                "success": False,
                "symbol": symbol,
                "message": f"No prediction available for {symbol} {timeframe}",
                "prediction": None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200
        
        # Use frozen API contract
        pred_dict = prediction.to_dict()
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "prediction": pred_dict,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Single prediction error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "symbol": symbol,
            "message": str(e),
            "prediction": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200  # Never return 500


@predictions_bp.route('/predictions/<symbol>/summary', methods=['GET'])
def get_prediction_summary(symbol: str):
    """
    Get summary prediction for a symbol across all timeframes.
    
    Response: {
        "status": "success",
        "symbol": "BTCUSDT",
        "summary": {
            "15m": {"direction": "LONG", "confidence": 0.78},
            "1h": {"direction": "LONG", "confidence": 0.65},
            "4h": {"direction": "SHORT", "confidence": 0.72},
            "1d": {"direction": "NO_TRADE", "confidence": 0.55}
        }
    }
    """
    try:
        symbol = symbol.upper()
        timeframes = ['15m', '1h', '4h', '1d']
        
        service = get_inference_service()
        summary = {}
        
        for tf in timeframes:
            prediction = service.predict(symbol, tf)
            if prediction:
                summary[tf] = {
                    "direction": prediction.direction,
                    "confidence": round(prediction.confidence, 4),
                }
        
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "summary": summary
        }), 200
    
    except Exception as e:
        logger.error(f"Summary error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
