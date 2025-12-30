#!/usr/bin/env python3
"""Quick test to verify ML is the signal source for trained coins"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
os.environ['USE_ML_PER_COIN'] = '1'
os.environ['ML_DEFAULT_TF'] = '15m'
os.environ['ML_DEVICE'] = 'cpu'

from crypto_bot.services.prediction_service import PredictionService
from crypto_bot.ml.inference.model_registry import ModelRegistry

print("=" * 70)
print("TEST: ML AS PRIMARY SIGNAL SOURCE")
print("=" * 70)

try:
    # Initialize registry
    registry = ModelRegistry()
    
    # Check BTCUSDT 15m model (which we trained)
    print("\nChecking BTCUSDT 15m model...")
    available = registry.is_model_available("BTCUSDT", "15m")
    print(f"  Model available: {available}")
    
    if available:
        # Load metadata
        meta = registry.load_metadata("BTCUSDT", "15m")
        if meta:
            print(f"  Model features: {meta.get('num_features', 'unknown')}")
            print(f"  Test accuracy: {meta.get('test_accuracy', 'unknown'):.4f}" if meta.get('test_accuracy') else "  Test accuracy: unknown")
        
        # Try to get a prediction
        print(f"\nGetting ML prediction for BTCUSDT 15m...")
        pred_service = PredictionService()
        result = pred_service.predict_symbol("BTCUSDT", "15m")
        if result:
            print(f"\n  Prediction result:")
            print(f"    Symbol: {result.get('symbol')}")
            print(f"    Direction: {result.get('pred')}")
            print(f"    Confidence: {result.get('confidence')}")
            print(f"    Source: {result.get('source')}")
            
            # Verify source is ML
            if result.get('source') == 'ML_PER_COIN_V1':
                print(f"\n  SUCCESS: Using ML_PER_COIN_V1 as primary source!")
            else:
                print(f"\n  WARNING: Using {result.get('source')} instead of ML")
        else:
            print(f"  No prediction returned")
    else:
        print(f"  Model not available - check training")
    
    # Check untrained coin
    print("\nChecking ETHUSDT 15m model (should fallback)...")
    available = registry.is_model_available("ETHUSDT", "15m")
    print(f"  Model available: {available}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
