"""
PROFESSIONAL TRADING ANALYZER
5+ Years Experience Logic with 95% Accuracy Target
(Updated Version)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")


@dataclass
class TradeSetup:
    """Complete professional trade setup"""
    symbol: str
    direction: str  # LONG, SHORT
    entry_price: float
    entry_range: Tuple[float, float]
    stop_loss: float
    take_profits: List[Tuple[float, float, float]]  # price, percentage, risk_reward
    confidence: float
    confluence_score: float
    accuracy_estimate: float
    timestamp: datetime
    timeframes: Dict[str, str]
    patterns: List[str]
    indicators: Dict[str, Any]
    risk_reward_ratio: float
    position_size: float
    leverage: int
    risk_percentage: float
    setup_type: str  # breakout, retest, reversal, etc.
    market_regime: str
    trade_priority: int  # 1-5, 5 being highest


class ConfigObj:
    """Config object wrapper for attribute-based access to dict values"""
    def __init__(self, config_dict=None):
        if config_dict:
            self.__dict__.update(config_dict)

        # Defaults
        if not hasattr(self, "MIN_CONFLUENCE_SCORE"):
            self.MIN_CONFLUENCE_SCORE = 40  # reduced for more signals
        if not hasattr(self, "LEVERAGE"):
            self.LEVERAGE = 10
        if not hasattr(self, "MAX_RISK_PER_TRADE"):
            self.MAX_RISK_PER_TRADE = 0.02
        if not hasattr(self, "RSI_OVERBOUGHT"):
            self.RSI_OVERBOUGHT = 70
        if not hasattr(self, "RSI_OVERSOLD"):
            self.RSI_OVERSOLD = 30
        if not hasattr(self, "VOLUME_SURGE_REQUIREMENT"):
            self.VOLUME_SURGE_REQUIREMENT = 1.5
        if not hasattr(self, "CONFLUENCE_WEIGHTS"):
            self.CONFLUENCE_WEIGHTS = {
                "trend": 0.25,
                "price_action": 0.20,
                "indicators": 0.15,
                "volume": 0.15,
                "order_book": 0.15,
                "multi_timeframe": 0.10,
            }
        if not hasattr(self, "BANKROLL"):
            self.BANKROLL = 10000
        if not hasattr(self, "MIN_CATEGORY_SCORES"):
            self.MIN_CATEGORY_SCORES = {"trend": 0.6, "price_action": 0.6, "indicators": 0.6}

        # Volatility / SL safety
        if not hasattr(self, "MIN_ATR_PERCENTAGE"):
            self.MIN_ATR_PERCENTAGE = 0.0015
        if not hasattr(self, "SL_ATR_MULTIPLIER"):
            self.SL_ATR_MULTIPLIER = 3.5
        if not hasattr(self, "MIN_SL_PERCENTAGE"):
            self.MIN_SL_PERCENTAGE = 0.0025

        if not hasattr(self, "POSITION_SIZING"):
            self.POSITION_SIZING = "fixed"

        # EMA periods you actually use
        if not hasattr(self, "EMA_PERIODS"):
            self.EMA_PERIODS = [8, 20, 50]

        # Multi-TF weights (optional)
        if not hasattr(self, "MULTI_TF_WEIGHTS"):
            self.MULTI_TF_WEIGHTS = {"4h": 0.4, "1h": 0.35, "15m": 0.25}

        if not hasattr(self, "REQUIRE_ALL_TF_ALIGNMENT"):
            self.REQUIRE_ALL_TF_ALIGNMENT = False

        # Regime thresholds (optional)
        if not hasattr(self, "REGIME_RANGE_THRESHOLD"):
            self.REGIME_RANGE_THRESHOLD = 0.01
        if not hasattr(self, "REGIME_ATR_PERCENTAGE"):
            self.REGIME_ATR_PERCENTAGE = 0.015

        # Accuracy threshold fallback
        if not hasattr(self, "MIN_ACCEPTABLE_ACCURACY"):
            self.MIN_ACCEPTABLE_ACCURACY = 0.6


class ProfessionalAnalyzer:
    """Complete professional trading analysis system"""

    def __init__(self, config=None):
        if isinstance(config, dict):
            self.config = ConfigObj(config)
        elif config:
            self.config = config
        else:
            self.config = ConfigObj()

        self.setup_history = []
        self.performance_stats = {
            "total_setups": 0,
            "high_confidence_setups": 0,
            "winning_setups": 0,
            "losing_setups": 0,
            "avg_accuracy": 0.0,
            "best_setup_score": 0.0,
            "market_regimes": {},
        }

        self.analyzer_config = self.load_analyzer_config("config/analyzer_config.json")
        self.confluence_threshold = self.analyzer_config.get("confluence_threshold", 6)
        self.accuracy_threshold = self.analyzer_config.get("accuracy_threshold", 0.75)
        self.layer_weights = self.analyzer_config.get(
            "weights",
            {
                "layer_1": 0.15,
                "layer_2": 0.15,
                "layer_3": 0.12,
                "layer_4": 0.12,
                "layer_5": 0.10,
                "layer_6": 0.10,
                "layer_7": 0.08,
                "layer_8": 0.18,
            },
        )

        # Directional thresholds
        self.rsi_bull_threshold = 40
        self.rsi_bear_threshold = 60

    @staticmethod
    def load_analyzer_config(config_path: str = "config/analyzer_config.json") -> Dict[str, Any]:
        default_config = {
            "confluence_threshold": 6,
            "accuracy_threshold": 0.75,
            "weights": {
                "layer_1": 0.15,
                "layer_2": 0.15,
                "layer_3": 0.12,
                "layer_4": 0.12,
                "layer_5": 0.10,
                "layer_6": 0.10,
                "layer_7": 0.08,
                "layer_8": 0.18,
            },
        }
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    loaded = json.load(f)
                merged = default_config.copy()
                merged.update(loaded)
                return merged
            return default_config
        except Exception:
            return default_config

    # -------------------------
    # Main Entry
    # -------------------------
    def analyze_complete_setup(self, symbol: str, data_frames: Dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        """
        Returns TradeSetup if conditions met, else None.
        """

        try:
            print(f"\n🔍 PROFESSIONAL ANALYSIS: {symbol}")
            print("=" * 60)

            # Validate required TFs exist
            if "15m" not in data_frames or data_frames["15m"] is None or len(data_frames["15m"]) < 50:
                print("  ❌ Missing/insufficient 15m data")
                return None

            # ========== PHASE 1: MULTI-TF TREND ==========
            trend_analysis = self._analyze_higher_timeframe_trend(data_frames)
            self._debug_trend_analysis(symbol, data_frames, trend_analysis)

            if not trend_analysis.get("trend_clear", False):
                print("  ❌ No clear trend on higher timeframes")
                return None

            trend_dir = trend_analysis.get("direction", "NEUTRAL")  # BULLISH/BEARISH/NEUTRAL

            # ========== PHASE 1.2: MARKET REGIME ==========
            market_regime = self._detect_market_regime(data_frames["15m"])
            print(f"  📊 Market Regime: {market_regime}")

            # ========== PHASE 2: PRICE ACTION ==========
            price_action = self._analyze_price_action(data_frames["15m"])
            sr_levels = self._calculate_support_resistance(data_frames["15m"])
            breakout_analysis = self._analyze_breakout_retest(data_frames["15m"], sr_levels, trend_dir)

            # ========== PHASE 3: INDICATORS ==========
            indicator_signals = self._analyze_indicators(data_frames["15m"])
            _ = self._analyze_rsi(data_frames["15m"])
            _ = self._analyze_macd(data_frames["15m"])
            _ = self._analyze_bollinger_bands(data_frames["15m"])
            _ = self._analyze_ema_alignment(data_frames["15m"])

            # ========== PHASE 4: VOLUME ==========
            volume_analysis = self._analyze_volume(data_frames["15m"])
            _ = self._detect_volume_spike(data_frames["15m"])
            _ = self._detect_volume_divergence(data_frames["15m"])

            # ========== PHASE 5: ORDER FLOW ==========
            order_flow = self._analyze_order_flow(symbol)

            # ========== PHASE 6: CONFLUENCE ==========
            confluence = self._calculate_professional_confluence(
                trend_analysis=trend_analysis,
                price_action=price_action,
                breakout_analysis=breakout_analysis,
                indicator_signals=indicator_signals,
                volume_analysis=volume_analysis,
                order_flow=order_flow,
                market_regime=market_regime,
            )

            print(f"  Confluence Score: {confluence['total_score']:.1f}/100")
            print(f"  📈 Required: {self.config.MIN_CONFLUENCE_SCORE}/100")

            # ========== PHASE 7: THRESHOLDS ==========
            if confluence["total_score"] < self.config.MIN_CONFLUENCE_SCORE:
                print("  ❌ Confluence score below minimum threshold")
                return None

            if not self._check_minimum_category_scores(confluence):
                print("  ❌ Minimum category scores not met")
                return None

            # ========== PHASE 8: DIRECTION ==========
            direction = self._determine_trade_direction(confluence, trend_analysis, indicator_signals)
            if direction == "NEUTRAL":
                print("  ❌ No clear trade direction")
                return None

            # ========== PHASE 9: LEVELS ==========
            trade_levels = self._calculate_professional_levels(data_frames["15m"], direction, confluence)

            # ========== PHASE 10: POSITION SIZING ==========
            position_size = self._calculate_professional_position_size(
                trade_levels["entry_price"],
                trade_levels["stop_loss"],
                confluence["confidence"],
                direction,
            )

            # ========== PHASE 11: ACCURACY ==========
            accuracy_estimate = self._estimate_professional_accuracy(confluence)
            if accuracy_estimate < getattr(self.config, "MIN_ACCEPTABLE_ACCURACY", 0.6):
                print(
                    f"  ❌ Accuracy estimate too low: {accuracy_estimate:.1%} < "
                    f"{getattr(self.config, 'MIN_ACCEPTABLE_ACCURACY', 0.6):.1%}"
                )
                return None

            setup = TradeSetup(
                symbol=symbol,
                direction=direction,
                entry_price=trade_levels["entry_price"],
                entry_range=trade_levels["entry_range"],
                stop_loss=trade_levels["stop_loss"],
                take_profits=trade_levels["take_profits"],
                confidence=confluence["confidence"] * 100,
                confluence_score=confluence["total_score"],
                accuracy_estimate=accuracy_estimate * 100,
                timestamp=datetime.now(),
                timeframes=self._get_timeframe_summary(data_frames),
                patterns=price_action["patterns"],
                indicators=indicator_signals,
                risk_reward_ratio=trade_levels["risk_reward_ratio"],
                position_size=position_size,
                leverage=self.config.LEVERAGE,
                risk_percentage=self.config.MAX_RISK_PER_TRADE,
                setup_type=breakout_analysis.get("setup_type", "UNKNOWN"),
                market_regime=market_regime,
                trade_priority=self._calculate_trade_priority(confluence),
            )

            self.setup_history.append(setup)
            self.performance_stats["total_setups"] += 1
            if confluence["confidence"] > 0.8:
                self.performance_stats["high_confidence_setups"] += 1

            print("  ✅ PROFESSIONAL SETUP GENERATED!")
            print(f"  🎯 Direction: {direction}")
            print(f"  📊 Accuracy Estimate: {accuracy_estimate:.1%}")
            print(f"  ⚖️ Risk/Reward: {trade_levels['risk_reward_ratio']:.2f}")
            print(f"  🔢 Priority: {setup.trade_priority}/5")

            return setup

        except Exception as e:
            print(f"  ❌ Error in professional analysis: {e}")
            return None

    # -------------------------
    # Trend / Multi-TF
    # -------------------------
    def _analyze_higher_timeframe_trend(self, data_frames: Dict) -> Dict:
        """Wrap multi-timeframe structure analysis to maintain old format."""
        mtf = self._analyze_multi_timeframe_structure(data_frames)

        if mtf.get("trend_clear", False):
            alignment = "STRONG" if mtf.get("all_aligned", False) or mtf.get("weighted_alignment", 0) >= 0.7 else "WEAK"
            return {
                "trend_clear": True,
                "direction": mtf.get("direction", "NEUTRAL"),
                "alignment": alignment,
                "timeframe_directions": mtf.get("timeframe_directions", {}),
                "weighted_alignment": mtf.get("weighted_alignment", 0.0),
                "all_aligned": mtf.get("all_aligned", False),
            }

        return {
            "trend_clear": False,
            "direction": "NEUTRAL",
            "alignment": "WEAK",
            "timeframe_directions": mtf.get("timeframe_directions", {}),
            "weighted_alignment": mtf.get("weighted_alignment", 0.0),
            "all_aligned": mtf.get("all_aligned", False),
        }

    def _debug_trend_analysis(self, symbol: str, data_frames: Dict, trend_analysis: Dict) -> None:
        print("\n  📊 TREND DEBUG:")
        for timeframe, df in data_frames.items():
            if df is None or len(df) < 2:
                print(f"    [{timeframe}] ⚠️  Insufficient data: {len(df) if df is not None else 0} candles")
                continue

            closes = df["close"].values
            current = closes[-1]
            prev = closes[-2]
            direction = "UP" if current > prev else "DOWN"
            change_pct = abs((current - prev) / prev) * 100 if prev else 0.0
            tf_dir = trend_analysis.get("timeframe_directions", {}).get(timeframe, "N/A")

            print(f"    [{timeframe}] {direction} | Change: {change_pct:.2f}% | Trend Dir: {tf_dir}")

        print(f"  Trend Clear: {trend_analysis.get('trend_clear', False)}")
        print(f"  Overall Direction: {trend_analysis.get('direction', 'NEUTRAL')}")
        print(f"  Alignment: {trend_analysis.get('alignment', 'WEAK')}\n")

    def _analyze_multi_timeframe_structure(self, data_frames: Dict) -> Dict:
        weights = getattr(self.config, "MULTI_TF_WEIGHTS", {})
        timeframe_directions = {}
        timeframe_structures = {}
        total_weight = sum(weights.values()) if isinstance(weights, dict) else 0.0

        counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}

        for tf, w in weights.items():
            if tf in data_frames and data_frames[tf] is not None and len(data_frames[tf]) >= 30:
                df = data_frames[tf]
                direction = self._determine_trend_direction(df)
                structure = self._determine_structure(df)
                timeframe_directions[tf] = direction
                timeframe_structures[tf] = structure
                counts[direction] = counts.get(direction, 0) + 1

        non_neutral = {k: v for k, v in counts.items() if k != "NEUTRAL"}
        majority = max(non_neutral.items(), key=lambda x: x[1])[0] if non_neutral else None

        matched_weight = 0.0
        if total_weight > 0 and majority is not None:
            for tf, dir_ in timeframe_directions.items():
                if dir_ == majority:
                    matched_weight += weights.get(tf, 0)
            weighted_alignment = matched_weight / total_weight
        else:
            weighted_alignment = 0.0

        all_aligned = bool(timeframe_directions) and all(d == majority for d in timeframe_directions.values())

        if getattr(self.config, "REQUIRE_ALL_TF_ALIGNMENT", False) and not all_aligned:
            return {
                "trend_clear": False,
                "direction": "NEUTRAL",
                "timeframe_directions": timeframe_directions,
                "timeframe_structures": timeframe_structures,
                "weighted_alignment": weighted_alignment,
                "all_aligned": all_aligned,
            }

        trend_clear = majority is not None and weighted_alignment >= 0.6
        direction = majority if trend_clear else "NEUTRAL"

        return {
            "trend_clear": trend_clear,
            "direction": direction,
            "timeframe_directions": timeframe_directions,
            "timeframe_structures": timeframe_structures,
            "weighted_alignment": weighted_alignment,
            "all_aligned": all_aligned,
        }

    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        if len(df) < 50:
            return "NEUTRAL"
        ema_50 = df["close"].ewm(span=50, adjust=False).mean()
        ema_200 = df["close"].ewm(span=200, adjust=False).mean()
        price = df["close"].iloc[-1]

        if price > ema_50.iloc[-1] > ema_200.iloc[-1]:
            return "BULLISH"
        if price < ema_50.iloc[-1] < ema_200.iloc[-1]:
            return "BEARISH"
        return "NEUTRAL"

    def _determine_structure(self, df: pd.DataFrame) -> str:
        lookback = min(len(df), 200)
        if lookback < 20:
            return "NEUTRAL"

        recent = df.tail(lookback)
        highs = recent["high"].values
        lows = recent["low"].values
        peaks = self._find_peaks(highs, window=5)
        troughs = self._find_troughs(lows, window=5)

        if len(peaks) >= 2 and len(troughs) >= 2:
            last_peak, prev_peak = highs[peaks[-1]], highs[peaks[-2]]
            last_trough, prev_trough = lows[troughs[-1]], lows[troughs[-2]]

            if last_peak > prev_peak and last_trough > prev_trough:
                return "HH_HL"
            if last_peak < prev_peak and last_trough < prev_trough:
                return "LH_LL"
        return "NEUTRAL"

    # -------------------------
    # Market Regime
    # -------------------------
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        if len(df) < 50:
            return "UNKNOWN"

        current_price = df["close"].iloc[-1]
        sma_50 = df["close"].rolling(50).mean()
        trend_strength = abs((current_price - sma_50.iloc[-1]) / sma_50.iloc[-1]) if sma_50.iloc[-1] else 0.0

        range_pct = (df["high"].max() - df["low"].min()) / df["close"].mean() if df["close"].mean() else 0.0

        atr = self._calculate_atr(df, period=getattr(self.config, "ATR_PERIOD", 14))
        atr_pct = atr / current_price if current_price else 0.0

        range_threshold = getattr(self.config, "REGIME_RANGE_THRESHOLD", 0.01)
        atr_threshold = getattr(self.config, "REGIME_ATR_PERCENTAGE", 0.015)

        try:
            print(f"  Analyzer regime -> range_pct={range_pct:.4f}, atr_pct={atr_pct:.4f}, trend_strength={trend_strength:.4f}")
        except Exception:
            pass

        if range_pct < range_threshold and atr_pct < atr_threshold:
            return "CONSOLIDATION"
        if atr_pct > atr_threshold * 3:
            return "HIGH_VOLATILITY"
        if trend_strength > 0.05:
            return "TRENDING"
        return "RANGING"

    # -------------------------
    # Price Action / S&R
    # -------------------------
    def _analyze_price_action(self, df: pd.DataFrame) -> Dict:
        patterns = []
        confs = []

        patterns_detected = self._detect_all_patterns(df)
        for p in patterns_detected:
            patterns.append(p["name"])
            confs.append(p["confidence"])

        sr_quality = self._calculate_sr_quality(df)
        return {
            "patterns": patterns,
            "avg_confidence": float(np.mean(confs)) if confs else 0.0,
            "sr_quality": float(sr_quality),
            "pattern_count": len(patterns),
        }

    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        if len(df) < lookback:
            return {"support": [], "resistance": []}

        recent = df.tail(lookback)
        highs = recent["high"].values
        lows = recent["low"].values

        peaks = self._find_peaks(highs, window=5)
        troughs = self._find_troughs(lows, window=5)

        resistance_levels = [highs[i] for i in peaks]
        support_levels = [lows[i] for i in troughs]

        resistance = self._cluster_levels(resistance_levels, tolerance=0.005)
        support = self._cluster_levels(support_levels, tolerance=0.005)

        current_price = df["close"].iloc[-1]

        nearest_resistance = min(resistance, key=lambda x: abs(x - current_price)) if resistance else current_price
        nearest_support = min(support, key=lambda x: abs(x - current_price)) if support else current_price

        return {
            "resistance": resistance[:5],
            "support": support[:5],
            "nearest_resistance": float(nearest_resistance),
            "nearest_support": float(nearest_support),
            "distance_to_resistance": float(abs(current_price - nearest_resistance) / current_price) if resistance and current_price else 0.0,
            "distance_to_support": float(abs(current_price - nearest_support) / current_price) if support and current_price else 0.0,
        }

    def _analyze_breakout_retest(self, df: pd.DataFrame, sr_levels: Dict, trend: str) -> Dict:
        breakout = self._check_breakout(df, sr_levels)
        retest = self._check_retest(df, sr_levels, breakout)
        setup_type = self._determine_setup_type(breakout, retest, trend)

        return {
            "breakout_detected": breakout["detected"],
            "retest_detected": retest["detected"],
            "setup_type": setup_type,
            "breakout_strength": breakout["strength"],
            "retest_quality": retest["quality"],
        }

    # -------------------------
    # Indicators
    # -------------------------
    def _analyze_indicators(self, df: pd.DataFrame) -> Dict:
        indicators = {}

        rsi = self._calculate_rsi(df)
        indicators["rsi"] = rsi
        indicators["rsi_signal"] = "BULLISH" if rsi < self.rsi_bull_threshold else "BEARISH" if rsi > self.rsi_bear_threshold else "NEUTRAL"

        macd = self._calculate_macd(df)
        indicators["macd"] = macd
        indicators["macd_signal"] = "BULLISH" if macd["histogram"] > 0 else "BEARISH"

        bb = self._calculate_bollinger_bands(df)
        indicators["bb"] = bb

        emas = self._calculate_emas(df)
        indicators["emas"] = emas

        atr = self._calculate_atr(df)
        indicators["atr"] = atr

        stoch = self._calculate_stochastic(df)
        indicators["stochastic"] = stoch

        return indicators

    def _analyze_rsi(self, df: pd.DataFrame) -> Dict:
        rsi = self._calculate_rsi(df)
        return {
            "value": rsi,
            "signal": "BULLISH" if rsi < self.rsi_bull_threshold else "BEARISH" if rsi > self.rsi_bear_threshold else "NEUTRAL",
            "overbought": rsi > self.config.RSI_OVERBOUGHT,
            "oversold": rsi < self.config.RSI_OVERSOLD,
        }

    def _analyze_macd(self, df: pd.DataFrame) -> Dict:
        macd = self._calculate_macd(df)
        return {"value": macd["macd"], "signal": "BULLISH" if macd["histogram"] > 0 else "BEARISH", "histogram": macd["histogram"]}

    def _analyze_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        bb = self._calculate_bollinger_bands(df)
        return {"upper": bb["upper"], "middle": bb["middle"], "lower": bb["lower"], "width": bb["width"]}

    def _analyze_ema_alignment(self, df: pd.DataFrame) -> Dict:
        emas = self._calculate_emas(df)

        ema_8 = emas.get("ema_8")
        ema_20 = emas.get("ema_20")
        ema_50 = emas.get("ema_50")

        signal = "NEUTRAL"
        aligned = False

        if ema_8 is not None and ema_20 is not None and ema_50 is not None:
            if ema_8 > ema_20 > ema_50:
                signal = "BULLISH"
                aligned = True
            elif ema_8 < ema_20 < ema_50:
                signal = "BEARISH"
                aligned = True

        return {"signal": signal, "ema_8": ema_8, "ema_20": ema_20, "ema_50": ema_50, "aligned": aligned}

    # -------------------------
    # Volume / Order Flow
    # -------------------------
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {"volume_ratio": 1.0, "volume_trend": "NEUTRAL", "divergence": False}

        volume = df["volume"]
        close = df["close"]

        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = float(volume.iloc[-1] / volume_ma.iloc[-1]) if volume_ma.iloc[-1] > 0 else 1.0
        volume_trend = "BULLISH" if volume.iloc[-1] > volume_ma.iloc[-1] else "BEARISH"

        price_change = close.pct_change()
        volume_trend_value = float((volume * price_change).cumsum().iloc[-1]) if len(price_change) else 0.0

        return {
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "volume_trend_value": volume_trend_value,
            "volume_ma": float(volume_ma.iloc[-1]),
        }

    def _analyze_order_flow(self, symbol: str) -> Dict:
        # Simulated for now
        return {"bid_ask_ratio": 1.2, "order_imbalance": 0.3, "market_depth": "GOOD", "large_orders": True}

    # -------------------------
    # Confluence
    # -------------------------
    def _calculate_professional_confluence(self, **analysis_results) -> Dict:
        scores = {
            "trend": 0.0,
            "price_action": 0.0,
            "indicators": 0.0,
            "volume": 0.0,
            "order_book": 0.0,
            "multi_timeframe": 0.0,
            "total_score": 0.0,
            "confidence": 0.0,
        }

        trend = analysis_results.get("trend_analysis", {})
        trend_direction = trend.get("direction", "NEUTRAL")

        # 1) Trend score (max 20)
        if trend.get("trend_clear", False):
            scores["trend"] = 20 if trend.get("alignment") == "STRONG" else 10

        # 2) Price action (max 25)
        price_action = analysis_results.get("price_action", {})
        breakout = analysis_results.get("breakout_analysis", {})

        pattern_score = min(10, len(price_action.get("patterns", [])) * 2)

        setup_type = breakout.get("setup_type", "")
        if setup_type in ["BREAKOUT_RETEST", "TREND_CONTINUATION"]:
            setup_score = 10
        elif setup_type in ["REVERSAL", "BREAKOUT"]:
            setup_score = 7
        else:
            setup_score = 3

        sr_quality = float(price_action.get("sr_quality", 0))
        sr_score = min(5, sr_quality * 5)

        scores["price_action"] = pattern_score + setup_score + sr_score

        # 3) Indicators (max 20)
        indicators = analysis_results.get("indicator_signals", {})
        indicator_score = 0

        if indicators.get("rsi_signal") == trend_direction:
            indicator_score += 5
        if indicators.get("macd_signal") == trend_direction:
            indicator_score += 5

        emas = indicators.get("emas", {})
        if self._check_ema_alignment(emas, trend_direction):
            indicator_score += 5

        bb = indicators.get("bb", {})
        if self._check_bb_position(bb, trend_direction):
            indicator_score += 5

        scores["indicators"] = indicator_score

        # 4) Volume (max 15)
        volume = analysis_results.get("volume_analysis", {})
        volume_score = 0
        volume_ratio = float(volume.get("volume_ratio", 1.0))

        if volume_ratio > 1.5:
            volume_score += 10
        elif volume_ratio > 1.2:
            volume_score += 7
        elif volume_ratio >= self.config.VOLUME_SURGE_REQUIREMENT:
            volume_score += 3

        if volume.get("volume_trend", "NEUTRAL") == trend_direction:
            volume_score += 5

        scores["volume"] = volume_score

        # 5) Order book (max 10)
        order_flow = analysis_results.get("order_flow", {})
        order_book_score = 0
        bid_ask_ratio = float(order_flow.get("bid_ask_ratio", 1.0))

        if (trend_direction == "BULLISH" and bid_ask_ratio > 1.2) or (trend_direction == "BEARISH" and bid_ask_ratio < 0.8):
            order_book_score += 10

        scores["order_book"] = order_book_score

        # 6) Multi-TF (max 10)
        weighted_alignment = float(trend.get("weighted_alignment", 0.0))
        all_aligned = bool(trend.get("all_aligned", False))

        if all_aligned:
            scores["multi_timeframe"] = 10
        elif weighted_alignment >= 0.7:
            scores["multi_timeframe"] = 7
        elif weighted_alignment >= 0.4:
            scores["multi_timeframe"] = 5
        else:
            scores["multi_timeframe"] = 3

        # Normalize to 0-100 and weight
        normalized_scores = {
            "trend": min(20, scores["trend"]) / 20 * 100,
            "price_action": min(25, scores["price_action"]) / 25 * 100,
            "indicators": min(20, scores["indicators"]) / 20 * 100,
            "volume": min(15, scores["volume"]) / 15 * 100,
            "order_book": min(10, scores["order_book"]) / 10 * 100,
            "multi_timeframe": min(10, scores["multi_timeframe"]) / 10 * 100,
        }

        total_score = (
            normalized_scores["trend"] * self.config.CONFLUENCE_WEIGHTS["trend"]
            + normalized_scores["price_action"] * self.config.CONFLUENCE_WEIGHTS["price_action"]
            + normalized_scores["indicators"] * self.config.CONFLUENCE_WEIGHTS["indicators"]
            + normalized_scores["volume"] * self.config.CONFLUENCE_WEIGHTS["volume"]
            + normalized_scores["order_book"] * self.config.CONFLUENCE_WEIGHTS["order_book"]
            + normalized_scores["multi_timeframe"] * self.config.CONFLUENCE_WEIGHTS["multi_timeframe"]
        )

        scores["total_score"] = float(total_score)
        scores["confidence"] = float(total_score / 100.0)
        return scores

    def _check_minimum_category_scores(self, confluence: Dict) -> bool:
        max_scores = {
            "trend": 20,
            "price_action": 25,
            "indicators": 20,
            "volume": 15,
            "order_book": 10,
            "multi_timeframe": 10,
        }

        for category, min_score_pct in self.config.MIN_CATEGORY_SCORES.items():
            raw_score = float(confluence.get(category, 0))
            max_score = float(max_scores.get(category, 100))
            normalized_score = (raw_score / max_score * 100) if max_score > 0 else 0
            min_threshold = float(min_score_pct) * 100

            if normalized_score < min_threshold:
                print(f"    ⚠️  {category}: {normalized_score:.1f}/100 < {min_threshold:.1f} (required)")
                return False
        return True

    # ✅ FIXED: real direction logic (no more always LONG)
    def _determine_trade_direction(self, confluence: Dict, trend_analysis: Dict, indicators: Dict) -> str:
        trend_dir = trend_analysis.get("direction", "NEUTRAL")  # BULLISH/BEARISH/NEUTRAL

        # Strong indicator alignment?
        rsi_sig = indicators.get("rsi_signal", "NEUTRAL")
        macd_sig = indicators.get("macd_signal", "NEUTRAL")

        if trend_dir == "BULLISH":
            # if indicators not fighting trend
            if rsi_sig in ("BULLISH", "NEUTRAL") and macd_sig in ("BULLISH", "NEUTRAL"):
                return "LONG"
        elif trend_dir == "BEARISH":
            if rsi_sig in ("BEARISH", "NEUTRAL") and macd_sig in ("BEARISH", "NEUTRAL"):
                return "SHORT"

        return "NEUTRAL"

    # -------------------------
    # Levels / Risk
    # -------------------------
    def _calculate_professional_levels(self, df: pd.DataFrame, direction: str, confluence: Dict) -> Dict:
        current_price = float(df["close"].iloc[-1])
        atr = float(self._calculate_atr(df))

        min_atr_distance = current_price * float(self.config.MIN_ATR_PERCENTAGE)
        safe_atr = max(atr, min_atr_distance)

        sl_distance = safe_atr * float(self.config.SL_ATR_MULTIPLIER)
        min_sl_distance = current_price * float(self.config.MIN_SL_PERCENTAGE)
        sl_distance = max(sl_distance, min_sl_distance)

        if direction == "LONG":
            entry_price = current_price * 1.001
            stop_loss = current_price - sl_distance
            entry_range = (current_price * 0.999, current_price * 1.002)
        else:
            entry_price = current_price * 0.999
            stop_loss = current_price + sl_distance
            entry_range = (current_price * 0.998, current_price * 1.001)

        stop_loss_pct = sl_distance / current_price if current_price else 0.0

        risk_amount = abs(entry_price - stop_loss)
        tp_levels = [1.0, 2.0, 3.0]
        tp_percentages = [50, 30, 20]

        take_profits: List[Tuple[float, float, float]] = []
        for multiple, percentage in zip(tp_levels, tp_percentages):
            if direction == "LONG":
                tp_price = entry_price + (risk_amount * multiple)
            else:
                tp_price = entry_price - (risk_amount * multiple)
            take_profits.append((float(tp_price), float(percentage), float(multiple)))

        avg_tp = sum(tp[0] * (tp[1] / 100) for tp in take_profits) / 100
        risk_reward_ratio = abs(avg_tp - entry_price) / risk_amount if risk_amount else 0.0

        print(f"  🔒 Dynamic SL: entry={entry_price:.6f}, sl={stop_loss:.6f}, sl_pct={stop_loss_pct:.4f} (min={self.config.MIN_SL_PERCENTAGE:.4f})")

        return {
            "entry_price": float(entry_price),
            "entry_range": (float(entry_range[0]), float(entry_range[1])),
            "stop_loss": float(stop_loss),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profits": take_profits,
            "risk_reward_ratio": float(risk_reward_ratio),
        }

    def _calculate_professional_position_size(self, entry: float, stop_loss: float, confidence: float, direction: str) -> float:
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return 0.0

        if self.config.POSITION_SIZING == "kelly":
            win_prob = float(confidence)
            loss_prob = 1.0 - win_prob
            risk_reward = 3.0
            kelly_fraction = (win_prob * risk_reward - loss_prob) / risk_reward
            kelly_fraction = min(kelly_fraction, 0.25)
            risk_amount = self.config.BANKROLL * kelly_fraction
        else:
            risk_amount = self.config.BANKROLL * self.config.MAX_RISK_PER_TRADE

        position_size = (risk_amount / stop_distance) * float(confidence)
        return float(position_size)

    def _estimate_professional_accuracy(self, confluence: Dict, timeframe: Optional[str] = None) -> float:
        score = float(confluence.get("total_score", 0))

        if score >= 95:
            base = 0.95
        elif score >= 90:
            base = 0.90
        elif score >= 85:
            base = 0.85
        elif score >= 80:
            base = 0.80
        elif score >= 75:
            base = 0.75
        else:
            base = 0.65

        if not timeframe:
            return float(base)

        tf_factors = {
            "1m": 1.00, "3m": 0.985, "5m": 0.985, "15m": 0.95, "30m": 0.92,
            "1h": 0.90, "4h": 0.88, "1d": 0.85, "1w": 0.82, "1M": 0.80
        }

        tf_key = str(timeframe).lower()
        factor = tf_factors.get(tf_key)
        return float(round(base * factor, 4)) if factor is not None else float(base)

    def _calculate_trade_priority(self, confluence: Dict) -> int:
        score = float(confluence.get("total_score", 0))
        if score >= 95:
            return 5
        if score >= 90:
            return 4
        if score >= 85:
            return 3
        if score >= 80:
            return 2
        return 1

    # -------------------------
    # Helpers / Patterns / Indicators
    # -------------------------
    def _detect_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        return []

    def _calculate_sr_quality(self, df: pd.DataFrame) -> float:
        return 0.8

    def _find_peaks(self, data, window: int = 5) -> List[int]:
        if hasattr(data, "values"):
            data = data.values
        peaks = []
        for i in range(window, len(data) - window):
            if data[i] == max(data[i - window : i + window + 1]):
                peaks.append(i)
        return peaks

    def _find_troughs(self, data, window: int = 5) -> List[int]:
        if hasattr(data, "values"):
            data = data.values
        troughs = []
        for i in range(window, len(data) - window):
            if data[i] == min(data[i - window : i + window + 1]):
                troughs.append(i)
        return troughs

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.005) -> List[float]:
        if not levels:
            return []
        levels.sort()
        clusters = []
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(float(np.mean(current_cluster)))
                current_cluster = [level]
        if current_cluster:
            clusters.append(float(np.mean(current_cluster)))
        return clusters

    def _check_breakout(self, df: pd.DataFrame, sr_levels: Dict) -> Dict:
        return {"detected": False, "strength": 0}

    def _check_retest(self, df: pd.DataFrame, sr_levels: Dict, breakout: Dict) -> Dict:
        return {"detected": False, "quality": 0}

    def _determine_setup_type(self, breakout: Dict, retest: Dict, trend: str) -> str:
        return "BREAKOUT_RETEST"

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        close = df["close"]
        if len(close) < period:
            return 50.0
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    def _calculate_macd(self, df: pd.DataFrame) -> Dict:
        close = df["close"]
        if len(close) < 26:
            return {"histogram": 0.0, "signal": 0.0, "macd": 0.0}
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return {"histogram": float(histogram.iloc[-1]), "signal": float(signal_line.iloc[-1]), "macd": float(macd_line.iloc[-1])}

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        close = df["close"]
        if len(close) < 20:
            current = float(close.iloc[-1])
            return {"upper": current, "middle": current, "lower": current, "width": 0.0, "position": 0.5}

        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)

        current_price = float(close.iloc[-1])
        denom = float(upper.iloc[-1] - lower.iloc[-1]) if float(upper.iloc[-1] - lower.iloc[-1]) != 0 else 1e-10
        position = float((current_price - float(lower.iloc[-1])) / denom)

        return {
            "upper": float(upper.iloc[-1]),
            "middle": float(sma.iloc[-1]),
            "lower": float(lower.iloc[-1]),
            "width": float((upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1]) if float(sma.iloc[-1]) else 0.0,
            "position": position,
        }

    def _calculate_emas(self, df: pd.DataFrame) -> Dict:
        close = df["close"]
        emas = {}

        for period in self.config.EMA_PERIODS:
            key = f"ema_{period}"
            if len(close) >= period:
                ema = close.ewm(span=period, adjust=False).mean()
                emas[key] = float(ema.iloc[-1])
            else:
                # ✅ important: do not fake EMAs with close price
                emas[key] = None

        return emas

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period:
            return float(df["close"].iloc[-1] * 0.02)

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return float(atr.iloc[-1]) if not atr.empty else float(close.iloc[-1] * 0.02)

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        if len(close) < k_period + d_period:
            return {"k": 50.0, "d": 50.0}

        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()

        k = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
        d = k.rolling(window=d_period).mean()
        return {"k": float(k.iloc[-1]), "d": float(d.iloc[-1])}

    # ✅ FIXED: correct EMA keys
    def _check_ema_alignment(self, emas: Dict, trend: str) -> bool:
        e8 = emas.get("ema_8")
        e20 = emas.get("ema_20")
        e50 = emas.get("ema_50")

        if e8 is None or e20 is None or e50 is None:
            return False

        if trend == "BULLISH":
            return e8 > e20 > e50
        if trend == "BEARISH":
            return e8 < e20 < e50
        return False

    def _check_bb_position(self, bb: Dict, trend: str) -> bool:
        position = float(bb.get("position", 0.5))
        if trend == "BULLISH":
            return position < 0.3
        if trend == "BEARISH":
            return position > 0.7
        return False

    def _detect_volume_spike(self, df: pd.DataFrame) -> bool:
        if len(df) < 20:
            return False
        volume = df["volume"]
        volume_ma = volume.rolling(window=20).mean()
        return float(volume.iloc[-1]) > float(volume_ma.iloc[-1]) * 1.5 if float(volume_ma.iloc[-1]) else False

    def _detect_volume_divergence(self, df: pd.DataFrame) -> bool:
        return False

    def _get_timeframe_summary(self, data_frames: Dict) -> Dict:
        mtf = self._analyze_multi_timeframe_structure(data_frames)
        return {
            "directions": mtf.get("timeframe_directions", {}),
            "structures": mtf.get("timeframe_structures", {}),
            "weighted_alignment": mtf.get("weighted_alignment", 0.0),
            "all_aligned": mtf.get("all_aligned", False),
        }
