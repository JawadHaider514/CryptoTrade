#!/usr/bin/env python3
"""
Test ML Integration - Verify complete flow
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import os
os.environ["USE_ML_PER_COIN"] = "1"
os.environ["ML_DEFAULT_TF"] = "15m"
os.environ["ML_DEVICE"] = "cpu"

print("="*70)
print("🧪 ML INTEGRATION TEST")
print("="*70)

# Test 1: ModelRegistry
print("\n1️⃣  Testing ModelRegistry...")
try:
    from crypto_bot.ml.inference.model_registry import get_registry
    registry = get_registry()
    print(f"   ✅ ModelRegistry initialized")
    
    # Check if any models available
    btc_available = registry.is_model_available("BTCUSDT", "15m")
    print(f"   Model BTCUSDT/15m available: {btc_available}")
    
    if btc_available:
        model, scaler, meta = registry.get_model("BTCUSDT", "15m", device="cpu")
        print(f"   ✅ Model loaded: {model is not None}")
        print(f"   ✅ Scaler loaded: {scaler is not None}")
        print(f"   ✅ Metadata loaded: {meta is not None}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: PredictionService
print("\n2️⃣  Testing PredictionService...")
try:
    from crypto_bot.services.prediction_service import PredictionService
    service = PredictionService(device="cpu")
    print(f"   ✅ PredictionService initialized")
    
    # Test predict_symbol (will use fallback if model missing)
    pred = service.predict_symbol("BTCUSDT", "15m")
    if pred:
        print(f"   ✅ Prediction received:")
        print(f"      Symbol: {pred.get('symbol')}")
        print(f"      Direction: {pred.get('pred')}")
        print(f"      Confidence: {pred.get('confidence'):.2f}")
        print(f"      Source: {pred.get('source')}")
    else:
        print(f"   ℹ️  No prediction (model not available - expected if no trained model)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: SignalEngineService with ML
print("\n3️⃣  Testing SignalEngineService with ML...")
try:
    from crypto_bot.services.signal_engine_service import get_prediction_service, USE_ML_PER_COIN
    pred_service = get_prediction_service()
    if pred_service:
        print(f"   ✅ ML PredictionService active")
    else:
        if USE_ML_PER_COIN:
            print(f"   ⚠️  ML enabled but PredictionService not initialized (expected if import fails)")
        else:
            print(f"   ℹ️  ML disabled via environment")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Check Flask app with /api/predictions
print("\n4️⃣  Testing /api/predictions endpoint...")
try:
    from crypto_bot.server.advanced_web_server import app
    with app.test_client() as client:
        response = client.get('/api/predictions')
        data = response.get_json()
        
        if response.status_code == 200 and data.get('success'):
            count = data.get('count', 0)
            print(f"   ✅ /api/predictions working")
            print(f"   ✅ Returned {count} predictions")
            
            # Check first prediction source
            predictions = data.get('predictions', {})
            if predictions:
                first_symbol = list(predictions.keys())[0]
                first_pred = predictions[first_symbol]
                source = first_pred.get('source', 'UNKNOWN')
                print(f"   ✅ First symbol ({first_symbol}) source: {source}")
                
                if source == "ML_PER_COIN_V1":
                    print(f"   ✅ ML SOURCE ACTIVE!")
                elif "FALLBACK" in source:
                    print(f"   ℹ️  Using fallback (models not available)")
        else:
            print(f"   ❌ /api/predictions failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("✅ ML INTEGRATION TEST COMPLETE")
print("="*70)
