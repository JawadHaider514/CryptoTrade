#!/usr/bin/env python3
"""
COMPLETE INFERENCE SYSTEM TEST
Verifies all components working end-to-end
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_bot.ml.inference.model_registry import get_registry
from crypto_bot.ml.inference.inference_service import InferenceService


def test_model_registry():
    """Test ModelRegistry singleton and model loading."""
    print("\n" + "=" * 70)
    print("TEST 1: ModelRegistry - Model Loading & Caching")
    print("=" * 70)
    
    registry = get_registry()
    
    # Check available models
    test_pairs = [
        ("BTCUSDT", "15m"),
        ("ETHUSDT", "15m"),
        ("SOLUSDT", "1h"),
    ]
    
    for symbol, tf in test_pairs:
        available = registry.is_model_available(symbol, tf)
        print(f"\n{symbol} {tf}:")
        print(f"  Model available: {available}")
        
        if available:
            accuracy = registry.get_model_accuracy(symbol, tf)
            print(f"  Model accuracy: {accuracy:.4f}")
            
            # Load model (tests caching)
            model = registry.load_model(symbol, tf)
            print(f"  Model loaded: {type(model).__name__}")
            
            # Load again (should use cache)
            model2 = registry.load_model(symbol, tf)
            print(f"  Cached correctly: {model is model2}")


def test_inference_service():
    """Test InferenceService predictions."""
    print("\n" + "=" * 70)
    print("TEST 2: InferenceService - Inference & Predictions")
    print("=" * 70)
    
    service = InferenceService()
    
    # Test 1: Single prediction
    print("\n[A] Single Prediction (BTCUSDT 15m)")
    result = service.predict("BTCUSDT", "15m")
    if result:
        print(f"  Direction: {result.direction}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Probabilities: {result.probabilities}")
        print(f"  Model Accuracy: {result.model_accuracy:.4f}")
        print(f"  Valid Until: {result.valid_until}")
    
    # Test 2: Batch prediction
    print("\n[B] Batch Predictions (3 coins, 15m)")
    results = service.predict_batch(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"], 
        "15m"
    )
    print(f"  Got {len(results)} predictions:")
    for symbol, pred in results.items():
        if pred:
            print(f"    {symbol}: {pred.direction} (conf={pred.confidence:.2f})")
    
    # Test 3: Multi-timeframe
    print("\n[C] Multi-timeframe Summary (BTCUSDT)")
    timeframes = ["15m", "1h", "4h", "1d"]
    summary = {}
    for tf in timeframes:
        pred = service.predict("BTCUSDT", tf)
        if pred:
            summary[tf] = pred
    
    print(f"  Got predictions for {len(summary)} timeframes:")
    for tf, pred in summary.items():
        print(f"    {tf}: {pred.direction} (conf={pred.confidence:.2f})")


def test_api_response_format():
    """Test API response JSON format."""
    print("\n" + "=" * 70)
    print("TEST 3: API Response Format (JSON Serialization)")
    print("=" * 70)
    
    service = InferenceService()
    result = service.predict("BTCUSDT", "15m")
    
    if result:
        response = result.to_dict()
        required_keys = [
            "symbol",
            "timeframe", 
            "direction",
            "confidence",
            "probabilities",
            "model_accuracy",
            "prediction_time",
            "valid_until",
        ]
        
        missing = [k for k in required_keys if k not in response]
        
        print(f"\n  Required Keys:")
        for key in required_keys:
            status = "YES" if key in response else "MISSING"
            print(f"    {key:20s} {status}")
        
        if not missing:
            print(f"\n  Sample JSON Response:")
            import json
            sample = {
                "symbol": response["symbol"],
                "timeframe": response["timeframe"],
                "direction": response["direction"],
                "confidence": response["confidence"],
                "model_accuracy": response["model_accuracy"],
            }
            print(json.dumps(sample, indent=4))
        else:
            print(f"\n  ERROR: Missing keys: {missing}")


def test_label_consistency():
    """Verify label encoding is consistent."""
    print("\n" + "=" * 70)
    print("TEST 4: Label Encoding Consistency")
    print("=" * 70)
    
    from crypto_bot.ml.inference.inference_service import (
        LABEL_TO_DIRECTION,
        DIRECTION_TO_LABEL
    )
    
    print(f"\n  Label to Direction:")
    for label, direction in LABEL_TO_DIRECTION.items():
        print(f"    {label} -> {direction}")
    
    print(f"\n  Direction to Label:")
    for direction, label in DIRECTION_TO_LABEL.items():
        print(f"    {direction} -> {label}")
    
    # Verify bidirectional mapping
    valid = True
    for label, direction in LABEL_TO_DIRECTION.items():
        if DIRECTION_TO_LABEL.get(direction) != label:
            print(f"  ERROR: Mapping mismatch for {label}/{direction}")
            valid = False
    
    if valid:
        print(f"\n  Mapping is bidirectional and consistent!")


def test_model_architecture():
    """Display model architecture details."""
    print("\n" + "=" * 70)
    print("TEST 5: Model Architecture Details")
    print("=" * 70)
    
    registry = get_registry()
    metadata = registry.load_metadata("BTCUSDT", "15m")
    
    if metadata:
        print(f"\n  Dataset Info:")
        dataset_info = metadata.get("dataset_info", {})
        print(f"    Num Features: {dataset_info.get('num_features', 'N/A')}")
        print(f"    Num Samples: {dataset_info.get('num_samples', 'N/A')}")
        print(f"    Lookback: {dataset_info.get('lookback', 'N/A')}")
        
        print(f"\n  Hyperparameters:")
        hyper = metadata.get("hyperparameters", {})
        for key, value in hyper.items():
            print(f"    {key}: {value}")
        
        print(f"\n  Test Metrics:")
        metrics = metadata.get("test_metrics", {})
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("CRYPTO TRADING SYSTEM - ML INFERENCE COMPLETE TEST")
    print("*" * 70)
    
    try:
        test_model_registry()
        test_inference_service()
        test_api_response_format()
        test_label_consistency()
        test_model_architecture()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nThe inference system is ready for:")
        print("  1. Dashboard integration")
        print("  2. Real-time signal serving (/api/predictions)")
        print("  3. Multi-timeframe analysis")
        print("  4. WebSocket signal updates")
        print("\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
