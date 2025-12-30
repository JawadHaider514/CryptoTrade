#!/usr/bin/env python3
"""
ENHANCED CRYPTO SCALPING DASHBOARD
⚡ With CSV Export & Auto Cleanup
🎯 University Final Year Project - COMPLETE VERSION
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import sys
from datetime import datetime, timedelta
import random
import warnings
from typing import Dict, List, Optional, Tuple
import csv
import json

warnings.filterwarnings('ignore')

# Import the trade tracker (place trade_tracker.py in same directory)
try:
    from trade_tracker import TradeTracker
    TRACKING_ENABLED = True
except ImportError:
    print("⚠️  Trade tracking disabled - trade_tracker.py not found")
    TRACKING_ENABLED = False

# ============================================================================
# ORIGINAL JAWAD'S CODE (PRESERVED)
# ============================================================================

class ScalpingConfig:
    """Configuration for crypto scalping system"""
    def __init__(self):
        # Load trading symbols from config
        try:
            import json as _json
            from pathlib import Path as _Path
            _config_path = _Path(__file__).parent.parent / "config" / "coins.json"
            _coins_config = _json.load(open(_config_path))
            self.SYMBOLS = _coins_config.get("symbols", [])[:8]  # Use first 8
        except Exception:
            # Fallback: 8 primary trading symbols
            self.SYMBOLS = [
                "XRPUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT",
                "ADAUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT"
            ]
        self.PRIMARY_SYMBOL = "XRPUSDT"
        
        # Timeframes for analysis
        self.TIMEFRAMES = ["1m", "5m", "15m"]
        
        # Signal generation thresholds
        self.MIN_CONFLUENCE_SCORE = 50
        self.MIN_PATTERN_SCORE = 30
        self.MIN_VOLUME_RATIO = 0.9
        
        # Trading parameters
        self.LEVERAGE = 20
        self.MAX_RISK_PER_TRADE = 0.5  # 0.5%
        self.BANKROLL = 10000
        
        # API settings
        self.API_TIMEOUT = 8
        self.MAX_RETRIES = 2
        
        # Pattern scoring
        self.PATTERN_SCORES = {
            "doji": 8,
            "hammer": 12,
            "shooting_star": 12,
            "bullish_engulfing": 15,
            "bearish_engulfing": 15,
            "morning_star": 20,
            "evening_star": 20
        }


class DataManager:
    """Fetch and manage market data"""
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.api_url = "https://api.binance.com/api/v3"
    
    def fetch_kline_data(self, symbol: str, interval: str, limit: int = 80):
        """Fetch candlestick data"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                url = f"{self.api_url}/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                }
                
                response = requests.get(url, params=params, timeout=self.config.API_TIMEOUT)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", 
                        "taker_buy_vol", "taker_buy_quote", "ignore"
                    ])
                    
                    # Convert to numeric
                    numeric_cols = ["open", "high", "low", "close", "volume"]
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
                    
            except Exception as e:
                if attempt == self.config.MAX_RETRIES - 1:
                    print(f"  Failed to fetch {symbol} data, using mock data")
                    return self._get_mock_data(symbol, interval, limit)
        
        return None
    
    def fetch_order_book(self, symbol: str):
        """Fetch order book data"""
        try:
            url = f"{self.api_url}/depth"
            params = {"symbol": symbol, "limit": 10}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                book_data = response.json()
                bids = book_data.get("bids", [])
                asks = book_data.get("asks", [])
                
                if bids and asks:
                    bid_volume = sum(float(bid[1]) for bid in bids[:5])
                    ask_volume = sum(float(ask[1]) for ask in asks[:5])
                    ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
                    return {"bid_ask_ratio": ratio, "score": 5}
                    
        except Exception:
            pass
        
        return {"bid_ask_ratio": 1.0, "score": 5}
    
    def _get_mock_data(self, symbol: str, interval: str, limit: int):
        """Generate realistic mock data"""
        end_date = datetime.now()
        
        # Set base price based on symbol
        if symbol == "BTCUSDT":
            base_price = 50000
        elif symbol == "ETHUSDT":
            base_price = 3000
        elif symbol == "XRPUSDT":
            base_price = 2.0
        elif symbol == "BNBUSDT":
            base_price = 300
        else:
            base_price = random.uniform(0.1, 10)
        
        # Generate time intervals
        if interval == '1m':
            freq = '1min'
        elif interval == '5m':
            freq = '5min'
        else:  # 15m
            freq = '15min'
        
        dates = pd.date_range(end=end_date, periods=limit, freq=freq)
        
        # Generate price movement
        returns = np.random.normal(0.0001, 0.008, limit)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create candles
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.003, 0.003, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, limit))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, limit)
        }, index=dates)
        
        return df
    
    def get_multi_timeframe_data(self, symbol: str):
        """Get data for multiple timeframes"""
        data = {}
        
        for timeframe in self.config.TIMEFRAMES:
            df = self.fetch_kline_data(symbol, timeframe, 80)
            if df is not None and len(df) >= 20:
                data[timeframe] = df
        
        return data


class IndicatorsCalculator:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_emas(close: pd.Series):
        """Calculate Exponential Moving Averages"""
        try:
            ema_9 = close.ewm(span=9, adjust=False).mean()
            ema_21 = close.ewm(span=21, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()
            
            return {
                "ema_9": float(ema_9.iloc[-1]),
                "ema_21": float(ema_21.iloc[-1]),
                "ema_50": float(ema_50.iloc[-1])
            }
        except:
            current = float(close.iloc[-1]) if len(close) > 0 else 1.0
            return {"ema_9": current, "ema_21": current, "ema_50": current}
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14):
        """Calculate RSI"""
        try:
            if len(close) < period:
                return 50.0
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not rsi.empty else 50.0
        except:
            return 50.0
    
    @staticmethod
    def calculate_macd(close: pd.Series):
        """Calculate MACD"""
        try:
            if len(close) < 26:
                return 0.0
            
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - macd_signal
            
            return float(macd_hist.iloc[-1]) if not macd_hist.empty else 0.0
        except:
            return 0.0
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series):
        """Calculate Average True Range"""
        try:
            if len(close) < 14:
                return float(close.iloc[-1] * 0.015)
            
            high_low = high - low
            high_close = (high - close.shift()).abs()
            low_close = (low - close.shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            return float(atr.iloc[-1]) if not atr.empty else float(close.iloc[-1] * 0.015)
        except:
            return 0.015


class PatternDetector:
    """Detect candlestick patterns"""
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
    
    def detect_patterns(self, df: pd.DataFrame):
        """Detect common candlestick patterns"""
        patterns = {}
        
        if len(df) < 3:
            return patterns
        
        recent = df.tail(3)
        current = recent.iloc[-1]
        
        # Doji detection
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        
        if total_range > 0 and body_size / total_range < 0.15:
            patterns["doji"] = {
                "type": "neutral",
                "score": self.config.PATTERN_SCORES["doji"],
                "confidence": 0.7
            }
        
        # Hammer detection
        if body_size > 0:
            lower_shadow = min(current['open'], current['close']) - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            
            if lower_shadow > body_size * 1.5 and upper_shadow < body_size * 0.5:
                patterns["hammer"] = {
                    "type": "bullish",
                    "score": self.config.PATTERN_SCORES["hammer"],
                    "confidence": 0.6
                }
            
            # Shooting star
            if upper_shadow > body_size * 1.5 and lower_shadow < body_size * 0.5:
                patterns["shooting_star"] = {
                    "type": "bearish",
                    "score": self.config.PATTERN_SCORES["shooting_star"],
                    "confidence": 0.6
                }
        
        # Engulfing patterns
        if len(recent) >= 2:
            prev = recent.iloc[-2]
            curr = recent.iloc[-1]
            
            prev_body = abs(prev['close'] - prev['open'])
            curr_body = abs(curr['close'] - curr['open'])
            
            # Bullish engulfing
            if (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
                curr['open'] <= prev['close'] and curr['close'] >= prev['open'] and
                curr_body > prev_body * 0.8):
                patterns["bullish_engulfing"] = {
                    "type": "bullish",
                    "score": self.config.PATTERN_SCORES["bullish_engulfing"],
                    "confidence": 0.7
                }
            
            # Bearish engulfing
            if (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
                curr['open'] >= prev['close'] and curr['close'] <= prev['open'] and
                curr_body > prev_body * 0.8):
                patterns["bearish_engulfing"] = {
                    "type": "bearish",
                    "score": self.config.PATTERN_SCORES["bearish_engulfing"],
                    "confidence": 0.7
                }
        
        return patterns


class SignalAnalyzer:
    """Analyze market data and generate signals"""
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.pattern_detector = PatternDetector(config)
        self.indicator_calc = IndicatorsCalculator()
    
    def analyze_symbol(self, symbol: str):
        """Generate trading signal for a symbol"""
        try:
            # Get data
            data = self.data_manager.get_multi_timeframe_data(symbol)
            if not data or "1m" not in data:
                return None
            
            df = data["1m"]
            if len(df) < 20:
                return None
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns(df)
            
            # Order book analysis
            orderbook = self.data_manager.fetch_order_book(symbol)
            
            # Multi-timeframe alignment
            mtf_score = self._check_mtf_alignment(data)
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(
                indicators, patterns, orderbook, mtf_score, df
            )
            
            # Check if score meets threshold
            if confluence_score["total"] < self.config.MIN_CONFLUENCE_SCORE:
                return None
            
            # Determine direction
            direction_data = self._determine_direction(indicators, patterns)
            if not direction_data:
                return None
            
            # Calculate trade levels
            levels = self._calculate_levels(direction_data, indicators, df)
            
            # Create signal
            signal = {
                "symbol": symbol,
                "direction": direction_data["direction"],
                "timestamp": datetime.now(),
                "entry_price": levels["entry"],
                "entry_range": (levels["entry_low"], levels["entry_high"]),
                "stop_loss": levels["stop_loss"],
                "take_profits": levels["take_profits"],
                "confluence_score": confluence_score["total"],
                "pattern_score": confluence_score["patterns"],
                "indicator_score": confluence_score["indicators"],
                "volume_score": confluence_score["volume"],
                "orderbook_score": confluence_score["orderbook"],
                "detected_patterns": list(patterns.keys()),
                "accuracy_estimate": self._estimate_accuracy(confluence_score["total"]),
                "risk_percentage": self.config.MAX_RISK_PER_TRADE,
                "leverage": self.config.LEVERAGE
            }
            
            return signal
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame):
        """Calculate all indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # EMAs
        emas = self.indicator_calc.calculate_emas(close)
        
        # RSI
        rsi = self.indicator_calc.calculate_rsi(close)
        
        # MACD
        macd_hist = self.indicator_calc.calculate_macd(close)
        
        # ATR
        atr = self.indicator_calc.calculate_atr(high, low, close)
        
        # Volume ratio
        try:
            volume_sma = volume.rolling(10).mean().iloc[-1] if len(volume) >= 10 else volume.iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_sma if volume_sma > 0 else 1.0
        except:
            volume_ratio = 1.0
        
        return {
            **emas,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "atr": atr,
            "volume_ratio": volume_ratio,
            "current_price": float(close.iloc[-1])
        }
    
    def _check_mtf_alignment(self, data: Dict):
        """Check multi-timeframe alignment"""
        alignment_score = 0
        
        for timeframe, df in data.items():
            if len(df) < 21:
                continue
            
            close = df['close']
            emas = self.indicator_calc.calculate_emas(close)
            current_price = float(close.iloc[-1])
            
            # Trend check
            if current_price > emas["ema_9"] > emas["ema_21"]:
                alignment_score += 5
            elif current_price < emas["ema_9"] < emas["ema_21"]:
                alignment_score += 5
            else:
                alignment_score += 2
        
        return min(alignment_score, 15)
    
    def _calculate_confluence_score(self, indicators: Dict, patterns: Dict, 
                                   orderbook: Dict, mtf_score: int, df: pd.DataFrame):
        """Calculate confluence score"""
        scores = {
            "indicators": 0,
            "patterns": 0,
            "volume": 0,
            "orderbook": 0,
            "mtf": mtf_score,
            "total": 0
        }
        
        current_price = indicators["current_price"]
        
        # Indicator Score (max 25)
        if indicators["ema_9"] > indicators["ema_21"]:
            scores["indicators"] += 8
        elif indicators["ema_9"] < indicators["ema_21"]:
            scores["indicators"] += 8
        
        if abs(current_price - indicators["ema_9"]) / current_price < 0.01:
            scores["indicators"] += 6
        
        if 20 < indicators["rsi"] < 80:
            scores["indicators"] += 6
        
        if abs(indicators["macd_hist"]) > 0.0001:
            scores["indicators"] += 5
        
        # Pattern Score (max 20)
        total_pattern_score = 0
        for pattern, data in patterns.items():
            total_pattern_score += int(data["score"] * data["confidence"])
        scores["patterns"] = min(20, total_pattern_score)
        
        # Volume Score (max 15)
        if indicators["volume_ratio"] >= self.config.MIN_VOLUME_RATIO:
            scores["volume"] = 15
        elif indicators["volume_ratio"] >= 0.7:
            scores["volume"] = 10
        else:
            scores["volume"] = 5
        
        # Order Book Score (max 10)
        scores["orderbook"] = min(10, orderbook.get("score", 5))
        
        # Calculate total
        scores["total"] = sum([scores["indicators"], scores["patterns"], 
                              scores["volume"], scores["orderbook"], scores["mtf"]])
        
        return scores
    
    def _determine_direction(self, indicators: Dict, patterns: Dict):
        """Determine trade direction"""
        bullish_points = 0
        bearish_points = 0
        
        # EMA trend
        if indicators["ema_9"] > indicators["ema_21"]:
            bullish_points += 2
        else:
            bearish_points += 2
        
        # Price position
        if indicators["current_price"] > indicators["ema_9"]:
            bullish_points += 1
        else:
            bearish_points += 1
        
        # RSI bias
        if indicators["rsi"] < 50:
            bullish_points += 1
        else:
            bearish_points += 1
        
        # MACD
        if indicators["macd_hist"] > 0:
            bullish_points += 1
        else:
            bearish_points += 1
        
        # Pattern bias
        for pattern, data in patterns.items():
            if data["type"] == "bullish":
                bullish_points += 1
            elif data["type"] == "bearish":
                bearish_points += 1
        
        # Determine direction
        if bullish_points > bearish_points + 1:
            return {"direction": "LONG", "confidence": bullish_points / (bullish_points + bearish_points)}
        elif bearish_points > bullish_points + 1:
            return {"direction": "SHORT", "confidence": bearish_points / (bullish_points + bearish_points)}
        
        return None
    
    def _calculate_levels(self, signal_data: Dict, indicators: Dict, df: pd.DataFrame):
        """Calculate trade levels"""
        current_price = indicators["current_price"]
        atr = indicators["atr"]
        direction = signal_data["direction"]
        
        # Entry calculations
        if direction == "LONG":
            entry = current_price + (atr * 0.1)
            entry_low = current_price - (atr * 0.3)
            entry_high = current_price + (atr * 0.2)
            stop_loss = current_price - (atr * 2.0)
        else:
            entry = current_price - (atr * 0.1)
            entry_low = current_price - (atr * 0.2)
            entry_high = current_price + (atr * 0.3)
            stop_loss = current_price + (atr * 2.0)
        
        # Take profits
        risk_amount = abs(entry - stop_loss)
        tp_ratios = [0.4, 0.8, 1.2, 2.0]
        tp_percentages = [40, 35, 20, 5]
        
        take_profits = []
        for ratio, percentage in zip(tp_ratios, tp_percentages):
            if direction == "LONG":
                tp_price = entry + (risk_amount * ratio)
            else:
                tp_price = entry - (risk_amount * ratio)
            
            take_profits.append((tp_price, percentage))
        
        return {
            "entry": entry,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop_loss": stop_loss,
            "take_profits": take_profits
        }
    
    def _estimate_accuracy(self, score: float):
        """Estimate accuracy based on score"""
        if score >= 80:
            return 88.0
        elif score >= 70:
            return 82.0
        elif score >= 60:
            return 78.0
        elif score >= 50:
            return 75.0
        else:
            return 70.0


def create_realistic_timeline(signal: Dict) -> List[str]:
    """Create realistic trade timeline"""
    timeline = []
    start_time = signal["timestamp"]
    direction = signal["direction"]
    entry_price = signal["entry_price"]
    take_profits = signal["take_profits"]
    
    # Entry (15-45 seconds)
    entry_delay = random.randint(15, 45)
    entry_time = start_time + timedelta(seconds=entry_delay)
    timeline.append(f"{entry_time.strftime('%H:%M:%S')}: Enter {direction} at {entry_price:.5f}")
    
    # Progressive TP hits
    current_time = entry_time
    for i, (tp_price, percentage) in enumerate(take_profits[:3], 1):
        delay = random.randint(45 + i*30, 120 + i*45)
        current_time += timedelta(seconds=delay)
        
        if i < 3:
            timeline.append(f"{current_time.strftime('%H:%M:%S')}: Hit TP{i} at {tp_price:.5f} ({percentage}% profit)")
        else:
            timeline.append(f"{current_time.strftime('%H:%M:%S')}: Hit TP{i} at {tp_price:.5f} - Trade Complete!")
    
    # Duration
    total_duration = current_time - entry_time
    minutes = int(total_duration.total_seconds() / 60)
    seconds = int(total_duration.total_seconds() % 60)
    timeline.append(f"\nTotal Duration: {minutes} minutes {seconds} seconds")
    
    return timeline


class SignalFormatter:
    """Format signals for display"""
    
    @staticmethod
    def format_detailed_signal(signal: Dict) -> str:
        """Format detailed signal"""
        if not signal:
            return "No signal available right now."
        
        timeline = create_realistic_timeline(signal)
        
        output = f"""# Futures : {signal['symbol']} - Advanced Analysis
TYPE : {signal['direction']} Signal
ENTRY : {signal['entry_price']:.5f}
LIMIT RANGE : {signal['entry_range'][0]:.5f} - {signal['entry_range'][1]:.5f}
EXCHANGE : Any
LEVERAGE : {signal['leverage']}x

TAKE PROFIT"""
        
        for i, (tp_price, percentage) in enumerate(signal['take_profits'], 1):
            output += f"\n{i} > {tp_price:.5f} ({percentage}%)"
        
        output += f"\n\nSTOPLOSS {signal['stop_loss']:.5f}"
        output += f"\n\nRisk {signal['risk_percentage']}%"
        
        # Analysis details
        output += f"\n\nANALYSIS RESULTS:"
        output += f"\nConfluence Score: {signal['confluence_score']}/100"
        output += f"\nAccuracy Estimate: {signal['accuracy_estimate']:.1f}%"
        output += f"\nIndicators: {signal['indicator_score']}/25"
        output += f"\nPatterns: {signal['pattern_score']}/20"
        output += f"\nVolume: {signal['volume_score']}/15"
        output += f"\nOrderBook: {signal['orderbook_score']}/10"
        
        if signal['detected_patterns']:
            output += f"\n\nDETECTED PATTERNS:"
            for pattern in signal['detected_patterns']:
                output += f"\n• {pattern.replace('_', ' ').title()}"
        
        # Timeline
        output += f"\n\nEXECUTION TIMELINE:"
        output += f"\n{signal['timestamp'].strftime('%H:%M:%S')}: Signal Generated ({signal['accuracy_estimate']:.1f}%)"
        
        for item in timeline:
            output += f"\n{item}"
        
        return output
    
    @staticmethod
    def format_simple_signal(signal: Dict) -> str:
        """Format simple signal"""
        if not signal:
            return "No signal available."
        
        timeline = create_realistic_timeline(signal)
        
        output = f"""# Futures : {signal['symbol']} - Scalping Trade
TYPE : {signal['direction']} Signal  
LIMIT ENTRY : {signal['entry_range'][0]:.5f} - {signal['entry_range'][1]:.5f}
EXCHANGE : Any
LEVERAGE : {signal['leverage']}x

TAKE PROFIT"""
        
        for i, (tp_price, percentage) in enumerate(signal['take_profits'], 1):
            output += f"\n{i} > {tp_price:.5f}"
        
        output += f"\n\nSTOPLOSS {signal['stop_loss']:.5f}"
        output += f"\n\nRisk {signal['risk_percentage']}%"
        output += f"\n\n{signal['timestamp'].strftime('%H:%M')} ✅"
        
        # Simple timeline
        output += f"\n\n{signal['timestamp'].strftime('%H:%M:%S')}: Signal Generated ({signal['accuracy_estimate']:.1f}%)"
        
        for item in timeline[:4]:
            output += f"\n{item}"
        
        return output


# ============================================================================
# NEW: ENHANCED DASHBOARD WITH CSV TRACKING
# ============================================================================

class EnhancedScalpingDashboard:
    """
    ENHANCED Dashboard with CSV Export & Auto Cleanup
    Combines Jawad's original system with trade tracking
    """
    
    def __init__(self):
        self.config = ScalpingConfig()
        self.analyzer = SignalAnalyzer(self.config)
        self.formatter = SignalFormatter()
        self.signals = {}
        self.active_trades = {}  # Track active trade IDs
        
        # Initialize trade tracker if available
        if TRACKING_ENABLED:
            self.tracker = TradeTracker(data_dir="./trade_data")
            print("✅ CSV tracking enabled")
        else:
            self.tracker = None
            print("⚠️  CSV tracking disabled")
        
        print("🎯 ENHANCED CRYPTO SCALPING DASHBOARD")
        print("⚡ Multi-Coin Monitoring + CSV Export + Auto Cleanup")
        print("="*60)
        print(f"INFO: System initialized")
        print(f"INFO: Confluence threshold: {self.config.MIN_CONFLUENCE_SCORE}/100")
    
    def generate_all_signals(self):
        """Generate signals for all coins"""
        self.signals = {}
        
        print(f"\nAnalyzing {len(self.config.SYMBOLS)} coins...")
        print("-"*60)
        
        for symbol in self.config.SYMBOLS:
            print(f"  Analyzing {symbol}...", end=" ")
            
            signal = self.analyzer.analyze_symbol(symbol)
            
            if signal:
                self.signals[symbol] = signal
                print(f"✅ Signal generated (Score: {signal['confluence_score']})")
                
                # NEW: Log to CSV if tracking enabled
                if self.tracker and symbol not in self.active_trades:
                    trade_id = self._log_signal_to_csv(signal)
                    self.active_trades[symbol] = trade_id
            else:
                print("❌ No signal")
        
        return len(self.signals)
    
    def _log_signal_to_csv(self, signal: Dict) -> str:
        """NEW: Log signal to CSV"""
        if not self.tracker:
            return None
        
        # Get coin name
        coin_name = self._get_coin_name(signal['symbol'])
        
        # Prepare trade data for CSV
        trade_data = {
            'pair': signal['symbol'],
            'coin_name': coin_name,
            'entry_time': signal['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'direction': signal['direction'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profits'][0][0] if signal['take_profits'] else 0,
            'tp_timeframe': '5m',  # Default
            'predicted_accuracy': signal['accuracy_estimate'],
            'status': 'OPEN',
            'timeframe': '5m',
            'strategy': f"Confluence {signal['confluence_score']}/100"
        }
        
        trade_id = self.tracker.log_trade(trade_data)
        print(f"    📝 Logged to CSV: {trade_id}")
        
        return trade_id
    
    def _get_coin_name(self, symbol: str) -> str:
        """Get full coin name from symbol"""
        coin_map = {
            'BTCUSDT': 'Bitcoin',
            'ETHUSDT': 'Ethereum',
            'XRPUSDT': 'Ripple',
            'BNBUSDT': 'Binance Coin',
            'ADAUSDT': 'Cardano',
            'SOLUSDT': 'Solana',
            'DOGEUSDT': 'Dogecoin',
            'DOTUSDT': 'Polkadot'
        }
        return coin_map.get(symbol, symbol.replace('USDT', ''))
    
    def display_dashboard(self):
        """Display the main dashboard"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*80)
        print("ENHANCED CRYPTO SCALPING DASHBOARD".center(80))
        print("="*80)
        
        if not self.signals:
            print("\n" + "NO ACTIVE SIGNALS".center(80))
            print("-"*80)
            print("Market conditions may not be favorable for trading.")
            print("Try again in 30-60 seconds.")
            return
        
        print(f"\nACTIVE SIGNALS: {len(self.signals)}/{len(self.config.SYMBOLS)} coins")
        
        if self.tracker:
            stats = self.tracker.get_trade_statistics()
            print(f"TOTAL TRADES LOGGED: {stats.get('total_trades', 0)}")
        
        print("-"*80)
        print(f"{'#':<2} {'Coin':<10} {'Direction':<10} {'Score':<8} {'Entry':<12} {'Stop Loss':<12} {'R/R':<6}")
        print("-"*80)
        
        for i, (symbol, signal) in enumerate(self.signals.items(), 1):
            direction_color = "\033[92m" if signal['direction'] == 'LONG' else "\033[91m"
            reset_color = "\033[0m"
            
            # Calculate risk/reward
            risk = abs(signal['entry_price'] - signal['stop_loss'])
            reward = abs(signal['take_profits'][0][0] - signal['entry_price'])
            rr_ratio = reward / risk if risk > 0 else 0
            
            print(f"{i:<2} {symbol:<10} "
                  f"{direction_color}{signal['direction']:<10}{reset_color} "
                  f"{signal['confluence_score']:<8.0f} "
                  f"{signal['entry_price']:<12.5f} "
                  f"{signal['stop_loss']:<12.5f} "
                  f"{rr_ratio:<6.2f}")
    
    def display_single_signal(self, symbol: str, format_type: str = "simple"):
        """Display signal for a specific coin"""
        if symbol not in self.signals:
            print(f"\nNo signal available for {symbol}")
            return
        
        signal = self.signals[symbol]
        
        if format_type == "simple":
            output = self.formatter.format_simple_signal(signal)
        else:
            output = self.formatter.format_detailed_signal(signal)
        
        print("\n" + "="*80)
        print(f"SIGNAL DETAILS: {symbol}".center(80))
        print("="*80)
        print(output)
        
        if self.tracker and symbol in self.active_trades:
            print(f"\n📝 Trade ID: {self.active_trades[symbol]}")
        
        print("\n⚠️  Educational use only - Not financial advice")
    
    def display_quick_signals(self):
        """Display quick signals for all coins with signals"""
        if not self.signals:
            print("\nNo signals available right now.")
            return
        
        for symbol, signal in self.signals.items():
            print(f"\n{'='*60}")
            print(f"📈 {symbol} - {signal['direction']} Signal")
            print(f"   Entry: {signal['entry_range'][0]:.5f} - {signal['entry_range'][1]:.5f}")
            print(f"   Stop: {signal['stop_loss']:.5f}")
            print(f"   Score: {signal['confluence_score']}/100 | Accuracy: {signal['accuracy_estimate']:.1f}%")
            print(f"   Patterns: {', '.join(signal['detected_patterns'][:2]) if signal['detected_patterns'] else 'None'}")
            
            if self.tracker and symbol in self.active_trades:
                print(f"   📝 Trade ID: {self.active_trades[symbol]}")
    
    def show_csv_statistics(self):
        """NEW: Show CSV statistics"""
        if not self.tracker:
            print("\n⚠️  CSV tracking not enabled")
            return
        
        print("\n" + "="*60)
        print("CSV TRADE STATISTICS".center(60))
        print("="*60)
        
        stats = self.tracker.get_trade_statistics()
        
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📁 CSV File: {self.tracker.csv_file}")
    
    def run_interactive(self):
        """Run interactive dashboard"""
        while True:
            # Generate signals
            signal_count = self.generate_all_signals()
            
            # Display dashboard
            self.display_dashboard()
            
            # Show menu
            print("\n" + "="*80)
            print("MENU:")
            print("  1-8. View detailed signal for coin")
            print("  A. View all quick signals")
            print("  S. Show CSV statistics")
            print("  R. Refresh dashboard")
            print("  Q. Quit")
            print("="*80)
            
            # Get user input
            choice = input("\nEnter choice: ").strip().upper()
            
            if choice == 'Q':
                print("\nExiting dashboard...")
                
                # Run cleanup before exiting
                if self.tracker:
                    print("\nRunning cleanup...")
                    self.tracker.cleanup_old_data(hours=24)
                
                break
            
            elif choice == 'R':
                continue
            
            elif choice == 'A':
                self.display_quick_signals()
                input("\nPress Enter to continue...")
            
            elif choice == 'S':
                self.show_csv_statistics()
                input("\nPress Enter to continue...")
            
            elif choice.isdigit() and 1 <= int(choice) <= len(self.signals):
                symbols = list(self.signals.keys())
                selected_symbol = symbols[int(choice) - 1]
                
                print("\nChoose format:")
                print("1. Simple Signal")
                print("2. Detailed Analysis")
                format_choice = input("\nEnter choice (1-2): ").strip()
                
                if format_choice == "2":
                    self.display_single_signal(selected_symbol, "detailed")
                else:
                    self.display_single_signal(selected_symbol, "simple")
                
                input("\nPress Enter to return to dashboard...")
            
            else:
                print("\nInvalid choice. Refreshing...")
                time.sleep(2)
    
    def run_single_analysis(self, symbol: str = None, format_type: str = "simple"):
        """Run single coin analysis"""
        if symbol is None:
            symbol = self.config.PRIMARY_SYMBOL
        
        print(f"\nChoose format:")
        print("1. Detailed Analysis")
        print("2. Simple Signal")
        
        try:
            choice = input("\nEnter choice (1-2): ").strip()
            print()
        except:
            choice = "2"
        
        if choice == "1":
            format_type = "detailed"
        else:
            format_type = "simple"
        
        print(f"INFO: Starting analysis for {symbol}")
        
        signal = self.analyzer.analyze_symbol(symbol)
        
        if not signal:
            print(f"No high-quality confluence signal available for {symbol}")
            return
        
        print(f"INFO: Confluence score: {signal['confluence_score']}/100")
        print(f"INFO: ✅ Signal generated: {signal['direction']} with {signal['confluence_score']}/100")
        
        # Log to CSV
        if self.tracker:
            trade_id = self._log_signal_to_csv(signal)
            self.active_trades[symbol] = trade_id
        
        if format_type == "simple":
            output = self.formatter.format_simple_signal(signal)
        else:
            output = self.formatter.format_detailed_signal(signal)
        
        print(output)
        print("\n⚠️  Educational use only - Not financial advice")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "="*80)
    print("ENHANCED CRYPTO SCALPING DASHBOARD".center(80))
    print("University Project with CSV Export & Auto Cleanup".center(80))
    print("="*80)
    
    print("\nChoose mode:")
    print("1. Multi-Coin Dashboard (Monitor all coins + CSV export)")
    print("2. Single Coin Analysis")
    print("3. Quick Signal Generation")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except:
        choice = "1"
    
    dashboard = EnhancedScalpingDashboard()
    
    if choice == "2":
        print("\n" + "="*60)
        print("SINGLE COIN ANALYSIS MODE")
        print("="*60)
        
        print(f"\nAvailable coins: {', '.join(dashboard.config.SYMBOLS)}")
        symbol = input(f"Enter symbol (default: {dashboard.config.PRIMARY_SYMBOL}): ").strip().upper()
        
        if not symbol:
            symbol = dashboard.config.PRIMARY_SYMBOL
        
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        dashboard.run_single_analysis(symbol)
    
    elif choice == "3":
        print("\n" + "="*60)
        print("QUICK SIGNAL GENERATION")
        print("="*60)
        
        dashboard.generate_all_signals()
        dashboard.display_quick_signals()
        
        if dashboard.tracker:
            dashboard.show_csv_statistics()
        
        print("\n⚠️  Educational use only - Not financial advice")
    
    else:
        print("\n" + "="*60)
        print("MULTI-COIN DASHBOARD MODE")
        print("="*60)
        print("CSV export enabled - All trades logged automatically")
        print("Press Ctrl+C to exit")
        print("="*60)
        
        dashboard.run_interactive()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()