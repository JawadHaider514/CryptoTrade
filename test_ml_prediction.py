#!/usr/bin/env python3
"""Test ML prediction with trained model"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
os.environ['USE_ML_PER_COIN'] = '1'

from crypto_bot.services.prediction_service import PredictionService

print("=" * 70)
print("🧪 TESTING ML PREDICTION WITH TRAINED MODEL")
print("=" * 70)

try:
    pred_service = PredictionService(device='cpu', min_confidence=0.5)
    print(f"\n✅ PredictionService initialized")
    
    print(f"\nTesting prediction for BTCUSDT 15m...")
    result = pred_service.predict_symbol('BTCUSDT', '15m')
    
    if result:
        print(f"\n✅ PREDICTION RECEIVED FROM ML MODEL")
        print(f"   Symbol: {result['symbol']}")
        print(f"   Direction: {result['pred']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Timeframe: {result['tf']}")
        print(f"   Source: {result['source']}")
        print(f"   Model Version: {result['model_version']}")
        print(f"   Timestamp: {result['ts']}")
    else:
        print(f"\n❌ No prediction (model may not be available)")
    
    print(f"\n" + "=" * 70)
    print("✅ ML PREDICTION SERVICE TEST COMPLETE")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
