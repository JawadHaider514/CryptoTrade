#!/usr/bin/env python3
"""Test TP calculation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import datetime
from crypto_bot.server.advanced_web_server import PredictionMapper

print("Testing PredictionMapper TP calculation...\n")

# Test LONG signal
print("LONG signal test:")
tp_data = PredictionMapper._calculate_take_profits(
    entry=0.3624,
    stop_loss=0.3500,
    direction='LONG',
    timestamp=datetime.utcnow()
)
print(f"  Entry: 0.3624, SL: 0.3500")
print(f"  Risk: 0.0124 (entry - SL)")
print(f"  TP1: {tp_data['tp1']} (entry + 1R)")
print(f"  TP2: {tp_data['tp2']} (entry + 2R)")
print(f"  TP3: {tp_data['tp3']} (entry + 3R)")
print(f"  Take Profits Array: {tp_data['take_profits'][:1]}")

# Test SHORT signal
print("\nSHORT signal test:")
tp_data = PredictionMapper._calculate_take_profits(
    entry=0.3624,
    stop_loss=0.3750,
    direction='SHORT',
    timestamp=datetime.utcnow()
)
print(f"  Entry: 0.3624, SL: 0.3750")
print(f"  Risk: 0.0126 (SL - entry)")
print(f"  TP1: {tp_data['tp1']} (entry - 1R)")
print(f"  TP2: {tp_data['tp2']} (entry - 2R)")
print(f"  TP3: {tp_data['tp3']} (entry - 3R)")
print(f"  TP1_ETA: {tp_data['tp1_eta']}")

print("\n✅ TP Calculation Test Complete!")
