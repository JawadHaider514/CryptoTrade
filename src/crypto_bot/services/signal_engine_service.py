#!/usr/bin/env python3
"""
Signal Engine Service with ML Integration
=========================================
Decision engine pipeline with ML priority:
1. ML Per-Coin Models (if available & enabled)
2. Professional Analyzer (fallback)
3. Simple RSI+MA fallback
4. Momentum fallback
5. NO_TRADE (no fake neutral-long)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os

from crypto_bot.domain.signal_models import SignalModel, TakeProfit
from crypto_bot.services.market_data_service import MarketDataService
from crypto_bot.services.market_history_service import get_market_history_service

# Load feature flags from settings
try:
    from config.settings import USE_PRO_ANALYZER, SIGNAL_VALID_MINUTES, MIN_CONFIDENCE, MIN_ACCURACY
except ImportError:
    USE_PRO_ANALYZER = True
    SIGNAL_VALID_MINUTES = 240
    MIN_CONFIDENCE = 15
    MIN_ACCURACY = 0

# ML settings from environment
USE_ML_PER_COIN = int(os.environ.get("USE_ML_PER_COIN", "0")) == 1
ML_DEFAULT_TF = os.environ.get("ML_DEFAULT_TF", "15m")
ML_DEVICE = os.environ.get("ML_DEVICE", "cpu")
ML_MIN_CONFIDENCE = 0.5

# Market data quality settings
MAX_PRICE_AGE_SECONDS = int(os.environ.get("MAX_PRICE_AGE_SECONDS", "15"))

logger = logging.getLogger(__name__)

_prediction_service = None


def get_prediction_service():
    """Lazy-load PredictionService only if ML is enabled."""
    global _prediction_service
    if _prediction_service is None and USE_ML_PER_COIN:
        try:
            from crypto_bot.services.prediction_service import PredictionService
            _prediction_service = PredictionService(device=ML_DEVICE, min_confidence=ML_MIN_CONFIDENCE)
            logger.info(f"✅ ML PredictionService initialized (device={ML_DEVICE})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PredictionService: {e}")
    return _prediction_service


def _to_float(value: Any) -> Optional[float]:
    """
    Safe float conversion to satisfy Pylance and avoid runtime crashes.
    Handles:
      - float/int/str
      - dict {"price": x} or {"value": x}
      - list/tuple like (price, weight) or [price, weight]
    """
    if value is None:
        return None

    # handle (price, weight) or [price, weight]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]

    # handle {"price": 123.4} etc
    if isinstance(value, dict):
        value = value.get("price") or value.get("value")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SignalEngineService:
    """
    Decision Engine:
    - uses ML (optional)
    - then Professional Analyzer
    - then fallbacks
    - else returns NO_TRADE (no forced direction)
    """

    def __init__(self, market_data: MarketDataService, enhanced_dashboard=None, professional_analyzer=None):
        self.market_data = market_data
        self.dashboard = enhanced_dashboard  # deprecated
        self.pro_analyzer = professional_analyzer

        logger.info(f"\n{'='*60}")
        logger.info("🔧 SignalEngineService Initialized")
        logger.info(f"   MIN_CONFIDENCE: {MIN_CONFIDENCE}%")
        logger.info(f"   MIN_ACCURACY: {MIN_ACCURACY}%")
        logger.info(f"   SIGNAL_VALID_MINUTES: {SIGNAL_VALID_MINUTES}")
        logger.info(f"   USE_PRO_ANALYZER: {USE_PRO_ANALYZER}")
        logger.info(f"   USE_ML_PER_COIN: {USE_ML_PER_COIN}")
        logger.info(f"   MAX_PRICE_AGE_SECONDS: {MAX_PRICE_AGE_SECONDS}")
        logger.info(f"{'='*60}\n")

    # -------------------------
    # Helpers
    # -------------------------
    def _get_valid_current_price(self, symbol: str) -> Optional[float]:
        """
        Returns current price only if:
        - price is not None
        - timestamp exists
        - timestamp is fresh
        """
        info = self.market_data.get_price_with_timestamp(symbol)
        if not info:
            logger.warning(f"[{symbol}] ⚠️ No price info available")
            return None

        price = info.get("price", None)
        ts = info.get("timestamp", None)

        if price is None or ts is None:
            logger.warning(f"[{symbol}] ⚠️ Price or timestamp missing (price={price}, ts={ts})")
            return None

        age = (datetime.utcnow() - ts).total_seconds()
        if age > MAX_PRICE_AGE_SECONDS:
            logger.warning(f"[{symbol}] ⚠️ Stale price: age={age:.0f}s > {MAX_PRICE_AGE_SECONDS}s")
            return None

        price_f = _to_float(price)
        if price_f is None or price_f <= 0:
            return None
        return price_f

    def _ml_prediction_to_analysis(self, ml_pred: Dict, current_price: float) -> Optional[Dict]:
        """Convert ML prediction dict to analysis format."""
        try:
            direction = ml_pred.get("pred", "LONG")
            confidence = int((ml_pred.get("confidence", 0.5) or 0.5) * 100)

            if direction in ("NO_TRADE", "HOLD"):
                return None

            entry = current_price

            # SL based on confidence
            if direction == "LONG":
                sl = entry * (0.98 if confidence >= 70 else 0.95)
            else:  # SHORT
                sl = entry * (1.02 if confidence >= 70 else 1.05)

            # TPs (prices only)
            distance = abs(entry - sl)
            tp1 = entry + (distance if direction == "LONG" else -distance)
            tp2 = entry + (2 * distance if direction == "LONG" else -2 * distance)
            tp3 = entry + (3 * distance if direction == "LONG" else -3 * distance)

            return {
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "take_profits": [tp1, tp2, tp3],
                "confidence_score": confidence,
                "accuracy_percent": min(confidence + 10, 95),
                "source": "ML_PER_COIN_V1",
                "leverage": 20,
                "reasons": [f"ML prediction: {direction}"],
                "patterns": ["ML_MODEL"],
                "market_context": "ML model signal",
            }
        except Exception as e:
            logger.error(f"ML prediction conversion failed: {e}")
            return None

    def normalize_analysis(self, analysis: Optional[Dict], symbol: str) -> Optional[Dict]:
        """
        Normalize analyzer output to standard format.
        Ensures take_profits becomes list of prices [tp1,tp2,tp3].
        """
        if not analysis:
            return None

        setup = analysis.get("setup") if isinstance(analysis, dict) and "setup" in analysis else analysis
        if not isinstance(setup, dict):
            logger.warning(f"[{symbol}] Unexpected analysis structure: {type(setup)}")
            return None
        # ✅ FIX: if analyzer returned TradeSetup (dataclass/object), convert to dict
        if hasattr(setup, "__dict__") and not isinstance(setup, dict):
            setup = dict(setup.__dict__)


        # entry
        entry: Any = None
        if "entry_price" in setup:
            entry = setup["entry_price"]
        elif "entry_range" in setup:
            er = setup["entry_range"]
            if isinstance(er, (list, tuple)) and len(er) >= 2:
                entry = (er[0] + er[1]) / 2
            elif isinstance(er, dict):
                low = er.get("low", 0)
                high = er.get("high", 0)
                entry = er.get("mid") or er.get("entry") or ((low + high) / 2 if (low and high) else None)
        elif "entry" in setup:
            entry = setup["entry"]

        # sl
        sl = setup.get("stop_loss") if "stop_loss" in setup else setup.get("sl")

        entry_f = _to_float(entry)
        sl_f = _to_float(sl)
        # ✅ FIX: direction inference (stop-loss relation)
        direction = setup.get("direction")
        if not direction:
            if sl_f is not None and entry_f is not None:
                if sl_f < entry_f:
                    direction = "LONG"
                elif sl_f > entry_f:
                    direction = "SHORT"
                else:
                    direction = "NO_TRADE"
            else:
                direction = "NO_TRADE"

        direction = str(direction).upper()


        # take profits -> prices only (✅ safe conversion)
        tp_prices: List[float] = []
        tp_raw = setup.get("take_profits")

        if isinstance(tp_raw, (list, tuple)):
            for tp in tp_raw:
                price_val: Optional[float] = None
                if isinstance(tp, dict):
                    price_val = _to_float(tp.get("price") or tp.get("value"))
                elif isinstance(tp, (list, tuple)):
                    price_val = _to_float(tp[0] if len(tp) > 0 else None)
                else:
                    price_val = _to_float(tp)

                if price_val is not None:
                    tp_prices.append(price_val)

        # individual tp fields
        if not tp_prices:
            for i in (1, 2, 3):
                k = f"take_profit_{i}"
                price_val = _to_float(setup.get(k))
                if price_val is not None:
                    tp_prices.append(price_val)

        # confidence/accuracy (✅ safe conversion)
        confidence = int(_to_float(
            setup.get("confidence_score")
            or setup.get("confidence")
            or setup.get("confluence_score")
            or 50
        ) or 50)

        accuracy = float(_to_float(
            setup.get("accuracy_percent")
            or setup.get("accuracy")
            or setup.get("accuracy_estimate")
            or 50.0
        ) or 50.0)

        if entry_f is None or sl_f is None or len(tp_prices) < 1:
            logger.warning(
                f"[{symbol}] Missing fields after normalization: entry={entry_f}, sl={sl_f}, tp_count={len(tp_prices)}"
            )
            return None

        return {
            "direction": "direction",
            "entry": float(entry_f),
            "sl": float(sl_f),
            "take_profits": tp_prices,
            "confidence_score": confidence,
            "accuracy_percent": accuracy,
            "leverage": int(_to_float(setup.get("leverage", 5)) or 5),
            "reasons": setup.get("reasons", []),
            "patterns": setup.get("patterns", []),
            "market_context": setup.get("market_context", ""),
            "source": setup.get("source", "UNKNOWN"),
        }

    def _step1_professional_analyzer(self, symbol: str) -> Optional[Dict]:
        if not self.pro_analyzer:
            logger.info(f"[{symbol}] STEP 1 SKIPPED: Professional analyzer not available")
            return None

        logger.info(f"[{symbol}] STEP 1 STARTING: Professional Analyzer")

        try:
            hist_service = get_market_history_service(cache_ttl=45)
            dfs = hist_service.get_dataframes(symbol)

            if not dfs or all(df.empty for df in dfs.values()):
                logger.info(f"[{symbol}] STEP 1 FAILED: No historical data available")
                return None

            setup = self.pro_analyzer.analyze_complete_setup(symbol, dfs)
            if not setup:
                logger.info(f"[{symbol}] STEP 1 FAILED: analyze_complete_setup returned None")
                return None

            analysis = self.normalize_analysis(setup, symbol)
            if not analysis:
                logger.info(f"[{symbol}] STEP 1 FAILED: Normalization failed")
                return None

            analysis["source"] = "PRO"
            logger.info(f"✅ [{symbol}] STEP 1 SUCCESS: Professional Analyzer")
            return analysis

        except Exception as e:
            logger.info(f"[{symbol}] STEP 1 EXCEPTION: {type(e).__name__}: {str(e)[:120]}")
            return None

    # -------------------------
    # Simple indicators
    # -------------------------
    def _calculate_rsi(self, prices: list, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        seed = deltas[:period]

        up = sum(x for x in seed if x > 0) / period if period > 0 else 0
        down = sum(abs(x) for x in seed if x < 0) / period if period > 0 else 0

        rs = up / down if down != 0 else 1.0
        return float(100 - (100 / (1 + rs)))

    def _simple_fallback_signal(self, symbol: str, current_price: float, dfs: Dict) -> Optional[Dict]:
        try:
            if "1h" not in dfs or dfs["1h"] is None or len(dfs["1h"]) < 30:
                logger.info(f"[{symbol}] Fallback 1 SKIP: Insufficient 1h data")
                return None

            df_1h = dfs["1h"]
            closes = df_1h["close"].values.tolist()

            rsi = self._calculate_rsi(closes[-30:], period=14)
            ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current_price
            ma5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else current_price

            logger.info(f"[{symbol}] Fallback 1: RSI={rsi:.1f}, MA5={ma5:.2f}, MA20={ma20:.2f}, Price={current_price:.2f}")

            if rsi < 30 and ma5 > ma20:
                return {
                    "direction": "LONG",
                    "entry": current_price,
                    "sl": current_price * 0.97,
                    "take_profits": [current_price * 1.02, current_price * 1.04, current_price * 1.06],
                    "confidence_score": 60,
                    "accuracy_percent": 60.0,
                    "leverage": 5,
                    "reasons": ["Oversold (RSI < 30)", "MA bullish crossover"],
                    "patterns": ["RSI_OVERSOLD", "MA_CROSSOVER"],
                    "market_context": "Oversold + Bullish MA alignment",
                    "source": "FALLBACK_RSI_MA",
                }
            elif rsi > 70 and ma5 < ma20:
                return {
                    "direction": "SHORT",
                    "entry": current_price,
                    "sl": current_price * 1.03,
                    "take_profits": [current_price * 0.98, current_price * 0.96, current_price * 0.94],
                    "confidence_score": 60,
                    "accuracy_percent": 60.0,
                    "leverage": 5,
                    "reasons": ["Overbought (RSI > 70)", "MA bearish crossover"],
                    "patterns": ["RSI_OVERBOUGHT", "MA_CROSSOVER"],
                    "market_context": "Overbought + Bearish MA alignment",
                    "source": "FALLBACK_RSI_MA",
                }

            return None

        except Exception as e:
            logger.info(f"[{symbol}] Fallback 1 Exception: {type(e).__name__}: {str(e)[:80]}")
            return None

    def _momentum_signal(self, symbol: str, current_price: float, dfs: Dict) -> Optional[Dict]:
        try:
            if "15m" not in dfs or dfs["15m"] is None or len(dfs["15m"]) < 14:
                logger.info(f"[{symbol}] Fallback 2 SKIP: Insufficient 15m data")
                return None

            df_15m = dfs["15m"]
            closes = df_15m["close"].values.tolist()
            momentum = ((closes[-1] - closes[-14]) / closes[-14]) * 100 if closes[-14] != 0 else 0

            logger.info(f"[{symbol}] Fallback 2: Momentum={momentum:.2f}%")

            if momentum > 2.5:
                return {
                    "direction": "LONG",
                    "entry": current_price,
                    "sl": current_price * 0.96,
                    "take_profits": [current_price * 1.02, current_price * 1.04, current_price * 1.07],
                    "confidence_score": 55,
                    "accuracy_percent": 55.0,
                    "leverage": 3,
                    "reasons": [f"Positive momentum: {momentum:.2f}%"],
                    "patterns": ["MOMENTUM_UP"],
                    "market_context": "Strong upward momentum",
                    "source": "FALLBACK_MOMENTUM",
                }
            elif momentum < -2.5:
                return {
                    "direction": "SHORT",
                    "entry": current_price,
                    "sl": current_price * 1.04,
                    "take_profits": [current_price * 0.98, current_price * 0.96, current_price * 0.93],
                    "confidence_score": 55,
                    "accuracy_percent": 55.0,
                    "leverage": 3,
                    "reasons": [f"Negative momentum: {momentum:.2f}%"],
                    "patterns": ["MOMENTUM_DOWN"],
                    "market_context": "Strong downward momentum",
                    "source": "FALLBACK_MOMENTUM",
                }

            return None

        except Exception as e:
            logger.info(f"[{symbol}] Fallback 2 Exception: {type(e).__name__}: {str(e)[:80]}")
            return None

    def _no_trade(self, symbol: str, reason: str) -> None:
        logger.info(f"[{symbol}] ⛔ NO_TRADE: {reason}")
        return None

    # -------------------------
    # Main generator
    # -------------------------
    def generate_for_symbol(self, symbol: str, timeframe: str = "15m") -> Optional[SignalModel]:
        logger.info(f"\n{'='*60}")
        logger.info(f"[{symbol}] GENERATING SIGNAL")
        logger.info(f"{'='*60}")

        # ✅ MUST: valid + fresh current price
        current_price = self._get_valid_current_price(symbol)
        if current_price is None:
            return self._no_trade(symbol, "No valid/fresh price available")

        logger.info(f"[{symbol}] Current price: ${current_price:.2f}")

        analysis = None

        # TRY 1: ML
        if USE_ML_PER_COIN:
            prediction_service = get_prediction_service()
            if prediction_service:
                try:
                    ml_pred = prediction_service.predict_symbol(symbol, timeframe)
                    if ml_pred and ml_pred.get("pred") not in ("NO_TRADE", "HOLD"):
                        analysis = self._ml_prediction_to_analysis(ml_pred, current_price)
                        if analysis:
                            logger.info(
                                f"[{symbol}] ✅ ML prediction: {ml_pred.get('pred')} (conf={ml_pred.get('confidence', 0):.2f})"
                            )
                except Exception as e:
                    logger.error(f"[{symbol}] ML prediction failed: {e}")

        # TRY 2: PRO
        if analysis is None and USE_PRO_ANALYZER:
            analysis = self._step1_professional_analyzer(symbol)
        elif analysis is None and not USE_PRO_ANALYZER:
            logger.info(f"[{symbol}] Professional analyzer disabled via feature flag")

        # TRY 3/4: fallbacks
        if analysis is None:
            logger.info(f"[{symbol}] No PRO/ML signal, trying fallbacks...")

            try:
                hist_service = get_market_history_service(cache_ttl=45)
                dfs = hist_service.get_dataframes(symbol)

                if dfs and not all(df.empty if df is not None else True for df in dfs.values()):
                    analysis = self._simple_fallback_signal(symbol, current_price, dfs)
                    if analysis:
                        logger.info(f"[{symbol}] ✅ Fallback 1 (RSI+MA) signal")

                    if not analysis:
                        analysis = self._momentum_signal(symbol, current_price, dfs)
                        if analysis:
                            logger.info(f"[{symbol}] ✅ Fallback 2 (Momentum) signal")

            except Exception as e:
                logger.info(f"[{symbol}] Could not load historical data for fallbacks: {type(e).__name__}")

        # ❌ NO forced trade
        if analysis is None:
            return self._no_trade(symbol, "No strategy produced a valid setup")

        # Extract
        direction = str(analysis.get("direction") or "NO_TRADE").upper()
        if direction in ("NO_TRADE", "HOLD"):
            return self._no_trade(symbol, f"Direction={direction}")

        entry_price = float(_to_float(analysis.get("entry", current_price)) or current_price)
        stop_loss = float(_to_float(analysis.get("sl", entry_price * 0.95)) or (entry_price * 0.95))
        confidence = int(_to_float(analysis.get("confidence_score", 50)) or 50)
        accuracy = float(_to_float(analysis.get("accuracy_percent", 50.0)) or 50.0)
        leverage = int(_to_float(analysis.get("leverage", 10)) or 10)
        source = analysis.get("source", "UNKNOWN")

        logger.info(
            f"[{symbol}] Analysis extracted: dir={direction}, entry=${entry_price:.2f}, sl=${stop_loss:.2f}, "
            f"conf={confidence}%, acc={accuracy:.1f}%, src={source}"
        )

        # Thresholds
        if confidence < MIN_CONFIDENCE:
            return self._no_trade(symbol, f"Filtered: confidence {confidence}% < MIN_CONFIDENCE {MIN_CONFIDENCE}%")
        if accuracy < MIN_ACCURACY:
            return self._no_trade(symbol, f"Filtered: accuracy {accuracy:.1f}% < MIN_ACCURACY {MIN_ACCURACY}%")

        # Take profits
        tp_data = analysis.get("take_profits", [])
        if not tp_data or len(tp_data) < 3:
            logger.info(f"[{symbol}] Generating synthetic TPs (received {len(tp_data) if tp_data else 0})")
            distance = abs(entry_price - stop_loss)
            tp_data = [
                entry_price + (distance * 0.5),
                entry_price + (distance * 1.0),
                entry_price + (distance * 1.5),
            ]

        tp1 = _to_float(tp_data[0] if len(tp_data) > 0 else None)
        tp2 = _to_float(tp_data[1] if len(tp_data) > 1 else None)
        tp3 = _to_float(tp_data[2] if len(tp_data) > 2 else None)

        if tp1 is None or tp2 is None or tp3 is None:
            distance = abs(entry_price - stop_loss)
            tp1 = entry_price + (distance * 0.5)
            tp2 = entry_price + (distance * 1.0)
            tp3 = entry_price + (distance * 1.5)

        now = datetime.utcnow()
        take_profits = [
            TakeProfit(level=1, price=float(tp1), eta=now + timedelta(minutes=min(5, SIGNAL_VALID_MINUTES // 4))),
            TakeProfit(level=2, price=float(tp2), eta=now + timedelta(minutes=min(15, SIGNAL_VALID_MINUTES // 2))),
            TakeProfit(level=3, price=float(tp3), eta=now + timedelta(minutes=SIGNAL_VALID_MINUTES)),
        ]

        signal = SignalModel(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profits=take_profits,
            confidence_score=confidence,
            accuracy_percent=accuracy,
            leverage=leverage,
            timestamp=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(hours=4),
            current_price=current_price,
            reasons=analysis.get("reasons", []),
            patterns=analysis.get("patterns", []),
            market_context=analysis.get("market_context", ""),
            source=source,
        )

        logger.info(f"✅ [{symbol}] SIGNAL GENERATED: {direction} @ ${entry_price:.2f} (conf={confidence}%, src={source})")
        logger.info(f"{'='*60}\n")
        return signal

    def generate_for_all(self, symbols: List[str], timeframe: str = "15m") -> Dict[str, SignalModel]:
        signals: Dict[str, SignalModel] = {}
        for symbol in symbols:
            sig = self.generate_for_symbol(symbol, timeframe)
            if sig:
                signals[symbol] = sig
        return signals
