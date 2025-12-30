"""
WEEK 3 INFERENCE INTEGRATION - IMPLEMENTATION GUIDE

This file documents the complete inference integration for the trading dashboard.

STATUS: Ready to integrate into Flask server

KEY MODULES:
1. ModelRegistry (src/crypto_bot/ml/inference/model_registry.py)
   - Loads trained CNN-LSTM models from disk
   - Caches models and scalers for performance
   - Provides metadata and accuracy info

2. InferenceService (src/crypto_bot/ml/inference/inference_service.py)
   - Makes predictions with loaded models
   - Handles feature loading and normalization
   - Returns structured predictions with confidence scores
   - Supports batch predictions

3. PredictionsAPI (src/crypto_bot/api/predictions_api.py)
   - Flask blueprint with 3 endpoints:
     * GET /api/predictions - All coins, single timeframe
     * GET /api/predictions/<symbol> - Single coin
     * GET /api/predictions/<symbol>/summary - Multi-timeframe summary
   - Returns predictions with entry/SL/TP levels
   - Includes model accuracy and confidence scores

========================================
STEP 1: Integration into Flask App
========================================

In src/crypto_bot/server/advanced_web_server.py (or equivalent):

    from crypto_bot.api.predictions_api import predictions_bp
    
    # Register blueprint
    app.register_blueprint(predictions_bp)

That's it! Endpoints are now live.

========================================
STEP 2: API Response Format
========================================

GET /api/predictions?timeframe=15m
Response:
{
    "status": "success",
    "timestamp": "2025-12-28T13:25:00+00:00",
    "timeframe": "15m",
    "count": 34,
    "predictions": [
        {
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "confidence": 0.78,
            "probabilities": {
                "LONG": 0.78,
                "SHORT": 0.15,
                "NO_TRADE": 0.07
            },
            "model_accuracy": 0.685,
            "prediction_time": "2025-12-28T13:25:00+00:00",
            "valid_until": "2025-12-28T13:27:00+00:00",
            "entry": 42500.50,
            "stop_loss": 41900.25,
            "take_profits": [
                {"level": 1, "price": 43100.75},
                {"level": 2, "price": 43701.00},
                {"level": 3, "price": 44301.25}
            ],
            "tp_eta_minutes": 45
        },
        ...
    ]
}

========================================
STEP 3: Query Parameters
========================================

GET /api/predictions
  ?timeframe=15m (or 1h, 4h, 1d)
  &symbols=BTCUSDT,ETHUSDT,SOLUSDT

GET /api/predictions/BTCUSDT
  ?timeframe=15m

GET /api/predictions/BTCUSDT/summary
  (Returns predictions for 15m, 1h, 4h, 1d)

========================================
STEP 4: Label Mapping (Already Done)
========================================

Model output classes:
  0 = SHORT
  1 = NO_TRADE
  2 = LONG

Feature engineering:
  14 indicators (RSI, MACD, ATR, EMA, Bollinger Bands, etc.)
  Normalized with StandardScaler (scaler saved during training)

========================================
STEP 5: Key Functions
========================================

from crypto_bot.ml.inference.inference_service import make_prediction

# Quick prediction
result = make_prediction("BTCUSDT", timeframe="15m")
# Returns: {
#   'symbol': 'BTCUSDT',
#   'direction': 'LONG',
#   'confidence': 0.78,
#   ...
# }

========================================
STEP 6: Error Handling
========================================

Prediction fails if:
  - Model not found in models/<SYMBOL>/<TF>/cnn_lstm_best.pt
  - Scaler not found in models/<SYMBOL>/<TF>/scaler.pkl
  - Dataset doesn't have latest features
  - Feature columns don't exist

All errors are logged to stdout/stderr.

========================================
STEP 7: Performance Notes
========================================

- Models are cached in memory (first prediction loads, subsequent are instant)
- Inference is CPU-optimized (set device="cuda" for GPU)
- Feature loading from parquet: ~5-10ms per coin
- Model inference: ~10-20ms per coin
- Total per-coin latency: ~20-30ms
- /api/predictions for 34 coins: ~1 second

========================================
STEP 8: Dashboard Integration
========================================

Dashboard can now:

1. Display real-time signals for all coins
2. Show per-timeframe predictions (15m, 1h, 4h, 1d)
3. Highlight high-confidence trades (confidence > 0.7)
4. Display entry/SL/TP levels for manual trading
5. Show model accuracy and reliability metrics
6. Warn when predictions expire (valid_until)

========================================
STEP 9: Testing
========================================

CLI test:
  python -c "
  from crypto_bot.ml.inference.inference_service import make_prediction
  result = make_prediction('BTCUSDT', '15m')
  print(result)
  "

API test:
  curl "http://localhost:5000/api/predictions?timeframe=15m&symbols=BTCUSDT,ETHUSDT"

========================================
STEP 10: Next Steps
========================================

Remaining Week 3 work:

1. Dashboard UI updates to display predictions
2. WebSocket real-time updates for new signals
3. Signal logging to database
4. Performance tracking (prediction accuracy vs actual moves)
5. Model retraining pipeline (weekly/monthly)

All infrastructure is ready. Just wire the endpoints into the dashboard!
"""
