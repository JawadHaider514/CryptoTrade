#!/usr/bin/env python3
"""Test complete signal generation with ML integration"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
os.environ['USE_ML_PER_COIN'] = '1'
os.environ['ML_DEFAULT_TF'] = '15m'

from crypto_bot.services.market_data_service import MarketDataService
from crypto_bot.services.signal_engine_service import SignalEngineService

# Use a small test set
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

print("=" * 70)
print("TEST: ML SIGNAL GENERATION")
print("=" * 70)

try:
    # Initialize services
    market_data = MarketDataService(symbols=TEST_SYMBOLS, use_websocket=False)
    signal_engine = SignalEngineService(market_data=market_data)
    
    print(f"\nGenerating signal for BTCUSDT with ML enabled...")
    signal = signal_engine.generate_for_symbol("BTCUSDT", "15m")
    
    if signal:
        print(f"\nSIGNAL GENERATED")
        print(f"   Symbol: {signal.symbol}")
        print(f"   Direction: {signal.direction}")
        print(f"   Entry: {signal.entry_price:.2f}")
        print(f"   Stop Loss: {signal.stop_loss:.2f}")
        print(f"   Confidence Score: {signal.confidence_score}%")
        print(f"   Source: {signal.source}")
        
        if signal.source == "ML_PER_COIN_V1":
            print(f"\nML BEING USED AS PRIMARY SOURCE!")
        else:
            print(f"\nFallback being used: {signal.source}")
    else:
        print(f"\nNo signal generated")
    
    print(f"\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
