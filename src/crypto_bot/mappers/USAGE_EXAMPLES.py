#!/usr/bin/env python3
"""
PredictionMapper Usage Examples
===============================
Shows how to normalize different analyzer outputs to unified schema
"""

# Example 1: Convert SignalModel to unified prediction
# ====================================================
# from crypto_bot.mappers import PredictionMapper
# 
# # After signal_engine_service generates a signal
# signal = signal_engine.generate_for_symbol('BTCUSDT')
# 
# # Convert to unified prediction schema
# prediction = PredictionMapper.from_signal_model(signal)
# 
# # prediction now has unified schema:
# # {
# #   "symbol": "BTCUSDT",
# #   "timeframe": "15m",
# #   "source": "PRO",  # or "DASHBOARD" or "FALLBACK"
# #   "direction": "LONG",
# #   "entry_price": 42500.50,
# #   "stop_loss": 42000.00,
# #   "take_profits": [
# #     {"level": 1, "price": 43000.00, "eta": "2025-12-27T09:15:51Z"},
# #     {"level": 2, "price": 43500.00, "eta": "2025-12-27T09:25:51Z"},
# #     {"level": 3, "price": 44000.00, "eta": "2025-12-27T09:55:51Z"}
# #   ],
# #   "confidence_score": 77,
# #   "accuracy_percent": 90.0,
# #   "leverage": 20,
# #   "current_price": 42500.50,
# #   "timestamp": "2025-12-27T09:10:51Z",
# #   "valid_until": "2025-12-27T13:10:51Z",
# #   "reasons": ["Pattern ABC detected", "Volume spike"],
# #   "patterns": ["head_and_shoulders"],
# #   "market_context": "Bullish trend with support"
# # }


# Example 2: Validate prediction schema
# ======================================
# from crypto_bot.mappers import PredictionMapper
# 
# if PredictionMapper.validate_prediction(prediction):
#     print("✅ Prediction valid")
#     # Send to API, database, or frontend
# else:
#     print("❌ Prediction invalid - missing or malformed fields")


# Example 3: Use convenience mapper function
# ===========================================
# from crypto_bot.mappers import map_to_prediction
# 
# # From SignalModel
# prediction = map_to_prediction(signal_model)
# 
# # From ProfessionalAnalyzer
# prediction = map_to_prediction(
#     pro_setup,
#     source='PRO',
#     symbol='ETHUSDT',
#     timeframe='1h'
# )
# 
# # From Dashboard
# prediction = map_to_prediction(
#     dashboard_analysis,
#     source='DASHBOARD',
#     symbol='BNBUSDT',
#     current_price=300.50,
#     timeframe='15m'
# )


# Example 4: ISO Datetime Format
# ===============================
# from crypto_bot.mappers import PredictionMapper
# from datetime import datetime
# 
# dt = datetime(2025, 12, 27, 9, 10, 51)
# iso_str = PredictionMapper._to_iso_string(dt)
# print(iso_str)  # Output: 2025-12-27T09:10:51Z  (no space before Z)


# Example 5: Integration with API Response
# =========================================
# from crypto_bot.mappers import PredictionMapper
# import json
# 
# signal = signal_engine.generate_for_symbol('ATOMUSDT')
# prediction = PredictionMapper.from_signal_model(signal)
# 
# # Send as JSON to frontend
# json_response = json.dumps(prediction)
# # {"symbol": "ATOMUSDT", "timeframe": "15m", ...}


# ISO FORMAT RULE
# ===============
# All timestamps MUST be in format: YYYY-MM-DDTHH:MM:SSZ
# - No space before Z
# - T separates date and time
# - Always UTC time (Z suffix)
# - Examples:
#   2025-12-27T09:10:51Z  ✅
#   2025-12-27T09:10:51   ❌ (missing Z)
#   2025-12-27 09:10:51Z  ❌ (space instead of T)
#   2025-12-27T09:10:51 Z ❌ (space before Z)
