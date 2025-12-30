#!/usr/bin/env python3
"""Test trained model loading with registry"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crypto_bot.ml.inference.model_registry import get_registry

print("=" * 70)
print("🧪 TESTING TRAINED MODEL LOADING")
print("=" * 70)

try:
    registry = get_registry()
    print(f"\n1. Testing ModelRegistry...")
    print(f"   ✅ Registry initialized")
    
    # Test availability
    available = registry.is_model_available("BTCUSDT", "15m")
    print(f"\n2. Testing model availability...")
    print(f"   BTCUSDT 15m available: {available}")
    
    if not available:
        print(f"   ❌ Model not found at expected path")
        sys.exit(1)
    
    # Load model
    print(f"\n3. Loading trained model...")
    model, scaler, meta = registry.get_model("BTCUSDT", "15m", device="cpu")
    
    if model is None:
        print(f"   ❌ Failed to load model")
        sys.exit(1)
    
    print(f"   ✅ Model loaded: {type(model).__name__}")
    
    if scaler is None:
        print(f"   ❌ Failed to load scaler")
        sys.exit(1)
    
    print(f"   ✅ Scaler loaded: {type(scaler).__name__}")
    
    if meta is None:
        print(f"   ❌ Failed to load metadata")
        sys.exit(1)
    
    print(f"   ✅ Metadata loaded: {meta.get('model_version')}")
    
    # Check metrics
    print(f"\n4. Model metrics:")
    test_metrics = meta.get("test_metrics", {})
    print(f"   Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
    print(f"   Precision: {test_metrics.get('precision', 0):.4f}")
    print(f"   Recall:    {test_metrics.get('recall', 0):.4f}")
    print(f"   F1:        {test_metrics.get('f1', 0):.4f}")
    
    print(f"\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - MODEL REGISTRY WORKING")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
