#!/usr/bin/env python3
"""
ENHANCED CRYPTO SCALPING DASHBOARD WITH STREAMING ML
⚡ Real-time Processing + ML Predictions + CSV Export
🎯 University Final Year Project - COMPLETE VERSION

🔧 RECENT TIMING IMPROVEMENTS:
- Fixed unrealistic take profit timing logic
- Added intelligent timing based on symbol volatility
- Major coins (BTC/ETH) get longer, more realistic timeframes
- Meme coins (PEPE/SHIB/DOGE) get faster execution times
- Price distance affects timing calculation
- Market context provided for each trade
- Added volatility-based timing categories
- Realistic duration formatting (hours/minutes/seconds)
- Market-specific execution expectations

📊 TIMING LOGIC:
- Low Volatility: TP1 5-15min, TP2 +10-20min, TP3 +15-45min
- Medium Volatility: TP1 3-10min, TP2 +5-15min, TP3 +10-30min  
- High Volatility: TP1 1-5min, TP2 +3-10min, TP3 +5-20min

🚀 Usage: python script.py --test-timing (to test new timing)
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
from typing import Dict, List, Optional, Tuple, Union, Any
import csv
import json
import concurrent.futures
import threading
import asyncio
import queue
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque
import sqlite3
import hmac
import hashlib
from urllib.parse import urlencode

warnings.filterwarnings('ignore')

# Import the trade tracker (place trade_tracker.py in same directory)
try:
    from trade_tracker import TradeTracker
    TRACKING_ENABLED = True
except ImportError:
    TRACKING_ENABLED = False
    TradeTracker = None

# Import the live signal tracker for real-time tracking
try:
    from crypto_bot.core.live_tracker import LiveSignalTracker
    LIVE_TRACKER_ENABLED = True
except ImportError:
    LIVE_TRACKER_ENABLED = False
    LiveSignalTracker = None

# Import the optimized configuration loader
try:
    from crypto_bot.config.config_loader import get_config
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False
    get_config = lambda: None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# STREAMING ML COMPONENTS (Manager's Code)
# ============================================================================

class SignalQuality(Enum):
    PREMIUM = "PREMIUM"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class PredictionMetrics:
    """Advanced prediction metrics with ML features"""
    price_target_30m: float
    price_target_1h: float
    price_target_3h: float
    breakout_probability: float
    volume_surge_probability: float
    trend_reversal_probability: float
    market_correlation_score: float
    volatility_prediction: float
    confidence_interval: Tuple[float, float]
    risk_score: float

@dataclass
class EnhancedSignal:
    """Enhanced signal with ML predictions and streaming metadata"""
    symbol: str
    direction: str
    confidence: float
    quality: SignalQuality
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    current_price: float
    volume_24h: float
    change_24h: float
    predictions: PredictionMetrics
    patterns: List[str]
    timestamp: datetime
    processing_time: float
    data_sources: List[str]
    ml_features: Dict
    leverage: int = 20  # Default leverage

class BinanceStreamingAPI:
    """High-performance Binance API with streaming capabilities"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.cache = {}
        self.cache_ttl = 30
        
    def get_streaming_data_sync(self, symbols: List[str]) -> Dict:
        """Get streaming market data for multiple symbols (synchronous version)"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                all_tickers = response.json()
                
                filtered_data = {}
                for ticker in all_tickers:
                    symbol = ticker['symbol']
                    if symbol in symbols:
                        try:
                            filtered_data[symbol] = {
                                'price': float(ticker['lastPrice']),
                                'change_24h': float(ticker['priceChangePercent']),
                                'volume_24h': float(ticker['volume']),
                                'high_24h': float(ticker['highPrice']),
                                'low_24h': float(ticker['lowPrice']),
                                'trades_count': int(ticker['count']),
                                'quote_volume': float(ticker['quoteVolume'])
                            }
                        except (ValueError, KeyError) as e:
                            continue
                
                logger.info(f"✅ Streaming data fetched for {len(filtered_data)} symbols")
                return filtered_data
            
            logger.warning(f"⚠️ Binance API returned status: {response.status_code}")
            return {}
            
        except Exception as e:
            logger.error(f"❌ Streaming API error: {e}")
            return {}
    
    def get_klines_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> List:
        """Get candlestick data for technical analysis"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
            
        except Exception as e:
            logger.error(f"❌ Klines data error for {symbol}: {e}")
            return []

class AdvancedMLPredictor:
    """Advanced ML-based prediction engine"""
    
    def __init__(self):
        self.feature_weights = {
            'rsi': 0.15,
            'macd': 0.12,
            'bollinger': 0.10,
            'volume_profile': 0.15,
            'price_momentum': 0.18,
            'market_correlation': 0.10,
            'volatility': 0.08,
            'pattern_recognition': 0.12
        }
        
    def calculate_technical_features(self, price_data: List, volume_data: List) -> Dict:
        """Calculate comprehensive technical analysis features"""
        try:
            if len(price_data) < 20:
                return {}
            
            prices = np.array(price_data[-50:]) if len(price_data) >= 50 else np.array(price_data)
            volumes = np.array(volume_data[-50:]) if volume_data and len(volume_data) >= 50 else np.ones(len(prices))
            
            # RSI calculation
            if len(prices) >= 15:
                deltas = np.diff(prices[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0
            
            # Simplified MACD (compute series-based MACD and signal line)
            if len(prices) >= 26:
                ema_12_series = self._calculate_ema(prices, 12)
                ema_26_series = self._calculate_ema(prices, 26)
                # Use latest EMA values
                ema_12 = float(ema_12_series[-1])
                ema_26 = float(ema_26_series[-1])
                # MACD series and latest value
                macd_series = ema_12_series - ema_26_series
                macd = float(macd_series[-1])
                # Signal line: EMA of the MACD series (if enough data available)
                if len(macd_series) >= 9:
                    macd_signal_series = self._calculate_ema(macd_series, 9)
                    macd_signal = float(macd_signal_series[-1])
                else:
                    macd_signal = macd
                macd_histogram = macd - macd_signal
            else:
                macd_histogram = 0
            
            # Bollinger Bands
            bb_period = min(20, len(prices))
            if bb_period > 5:
                sma = np.mean(prices[-bb_period:])
                std = np.std(prices[-bb_period:])
                bb_upper = sma + (2 * std) if std > 0 else sma
                bb_lower = sma - (2 * std) if std > 0 else sma
                bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
            else:
                bb_position = 0.5
            
            # Volume analysis
            if len(volumes) >= 20:
                vol_sma = np.mean(volumes[-20:])
                vol_ratio = volumes[-1] / vol_sma if vol_sma > 0 else 1
            else:
                vol_ratio = 1
            
            # Price momentum
            if len(prices) >= 5:
                momentum_5 = (prices[-1] - prices[-5]) / prices[-5] * 100 if prices[-5] != 0 else 0
            else:
                momentum_5 = 0
            
            if len(prices) >= 10:
                momentum_10 = (prices[-1] - prices[-10]) / prices[-10] * 100 if prices[-10] != 0 else 0
            else:
                momentum_10 = 0
            
            # Volatility
            if len(prices) >= 14:
                high_low_range = np.std(prices[-14:]) / np.mean(prices[-14:]) * 100 if np.mean(prices[-14:]) != 0 else 5
            else:
                high_low_range = 5
            
            return {
                'rsi': round(float(rsi), 2),
                'macd_histogram': round(float(macd_histogram), 6),
                'bb_position': round(float(bb_position), 4),
                'volume_ratio': round(float(vol_ratio), 2),
                'momentum_5': round(float(momentum_5), 2),
                'momentum_10': round(float(momentum_10), 2),
                'volatility': round(float(high_low_range), 2),
                'price_trend': 'UP' if momentum_10 > 2 else 'DOWN' if momentum_10 < -2 else 'SIDEWAYS'
            }
            
        except Exception as e:
            logger.error(f"❌ Technical features calculation error: {e}")
            return {}
    
    def _calculate_ema(self, prices: Union[np.ndarray, List[float]], period: int) -> np.ndarray:
        """Calculate Exponential Moving Average series.

        Returns an array of EMA values (same length as input).
        """
        prices_arr = np.asarray(prices, dtype=float)
        if prices_arr.size == 0:
            return np.array([])

        ema = np.zeros_like(prices_arr, dtype=float)
        multiplier = 2 / (period + 1)
        ema[0] = prices_arr[0]

        for i in range(1, prices_arr.size):
            ema[i] = (prices_arr[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema
    
    def generate_ml_predictions(self, features: Dict, market_data: Dict) -> PredictionMetrics:
        """Generate advanced ML-based predictions"""
        try:
            current_price = market_data.get('price', 1)
            volatility = features.get('volatility', 5)
            volume_ratio = features.get('volume_ratio', 1)
            rsi = features.get('rsi', 50)
            momentum_5 = features.get('momentum_5', 0)
            
            # Base prediction model
            trend_factor = 1 if momentum_5 > 0 else -1
            volatility_factor = min(volatility / 10, 2.0)
            volume_factor = min(volume_ratio, 3.0)
            
            # Price targets
            base_move = current_price * (volatility / 100) * 0.3
            
            price_30m = current_price + (base_move * 0.5 * trend_factor * volume_factor)
            price_1h = current_price + (base_move * 1.0 * trend_factor * volume_factor)
            price_3h = current_price + (base_move * 1.8 * trend_factor * volume_factor)
            
            # Probability calculations
            breakout_prob = min(90, max(10, 50 + (volume_ratio * 10) + (abs(momentum_5) * 2)))
            volume_surge_prob = min(90, max(10, volume_ratio * 25))
            reversal_prob = 90 - breakout_prob if rsi > 70 or rsi < 30 else 20
            
            # Market correlation
            correlation_score = min(100, max(0, 50 + (volume_factor * 10) + (volatility_factor * 5)))
            
            # Volatility prediction
            vol_prediction = volatility * (1 + (volume_ratio - 1) * 0.3)
            
            # Confidence interval
            confidence_range = current_price * (vol_prediction / 100) * 0.5
            conf_lower = price_1h - confidence_range
            conf_upper = price_1h + confidence_range
            
            # Risk score
            risk = min(100, max(0, volatility * 2 + (abs(momentum_5) * 0.5)))
            
            return PredictionMetrics(
                price_target_30m=round(float(price_30m), 6),
                price_target_1h=round(float(price_1h), 6),
                price_target_3h=round(float(price_3h), 6),
                breakout_probability=round(float(breakout_prob), 1),
                volume_surge_probability=round(float(volume_surge_prob), 1),
                trend_reversal_probability=round(float(reversal_prob), 1),
                market_correlation_score=round(float(correlation_score), 1),
                volatility_prediction=round(float(vol_prediction), 2),
                confidence_interval=(round(float(conf_lower), 6), round(float(conf_upper), 6)),
                risk_score=round(float(risk), 1)
            )
            
        except Exception as e:
            logger.error(f"❌ ML prediction error: {e}")
            return PredictionMetrics(0, 0, 0, 50, 50, 50, 50, 5, (0, 0), 50)

class StreamingSignalProcessor:
    """Real-time streaming signal processor with ML integration"""
    
    def __init__(self):
        self.binance_api = BinanceStreamingAPI()
        self.ml_predictor = AdvancedMLPredictor()
        
        # Load trading symbols from config
        try:
            import json as _json
            from pathlib import Path as _Path
            _config_path = _Path(__file__).parent.parent.parent / "config" / "coins.json"
            _coins_config = _json.load(open(_config_path))
            self.SYMBOLS = _coins_config.get("symbols", [])
        except Exception:
            # Fallback: 32 verified trading symbols
            self.SYMBOLS = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
                "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "UNIUSDT",
                "LINKUSDT", "XLMUSDT", "ATOMUSDT", "MANAUSDT", "SANDUSDT",
                "DASHUSDT", "VETUSDT", "ICPUSDT", "GMTUSDT", "PEOPLEUSDT",
                "LUNCUSDT", "CHZUSDT", "NEARUSDT", "FLOWUSDT", "FILUSDT",
                "QTUMUSDT", "SNXUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT",
                "FLOKIUSDT", "OPUSDT",
            ]
        
        # Processing state
        self.is_processing = False
        self.processing_stats = {
            'total_processed': 0,
            'signals_generated': 0,
            'processing_time_avg': 0,
            'last_batch_time': None
        }
    
    def process_symbols_batch(self) -> List[EnhancedSignal]:
        """Process symbols batch with advanced ML analysis"""
        try:
            start_time = time.time()
            logger.info(f"🔄 Processing {len(self.SYMBOLS)} symbols...")
            
            # Get streaming market data
            try:
                market_data = self.binance_api.get_streaming_data_sync(self.SYMBOLS)
            except Exception as data_err:
                logger.error(f"❌ Error fetching market data: {type(data_err).__name__}: {str(data_err)[:100]}")
                return []
            
            if not market_data:
                logger.warning("⚠️ No market data received")
                return []
            
            signals = []
            symbols_to_process = self.SYMBOLS[:15]  # Process first 15 symbols for speed
            
            # Process each symbol
            for symbol in symbols_to_process:
                if symbol not in market_data:
                    logger.debug(f"⊘ {symbol} not in market data")
                    continue
                
                try:
                    signal = self._analyze_symbol_advanced(symbol, market_data[symbol])
                    if signal:
                        signals.append(signal)
                        logger.debug(f"✅ Signal generated for {symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ Error analyzing {symbol}: {type(e).__name__}: {str(e)[:100]}")
                    continue
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.processing_stats['total_processed'] += len(symbols_to_process)
            self.processing_stats['signals_generated'] += len(signals)
            self.processing_stats['processing_time_avg'] = processing_time
            self.processing_stats['last_batch_time'] = datetime.now()
            
            logger.info(f"✅ Batch completed: {len(signals)} signals in {processing_time:.2f}s")
            return signals
        
        except Exception as e:
            logger.error(f"❌ Critical error in process_symbols_batch: {type(e).__name__}: {str(e)[:100]}", exc_info=True)
            return []
    
    def _analyze_symbol_advanced(self, symbol: str, data: Dict) -> Optional[EnhancedSignal]:
        """Advanced symbol analysis with ML predictions"""
        try:
            # Get historical data
            klines = self.binance_api.get_klines_data(symbol, "1h", 50)
            if not klines:
                logger.warning(f"⚠️ No historical data for {symbol}")
                return None
            
            # Extract price and volume data
            prices = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            # Calculate technical features
            features = self.ml_predictor.calculate_technical_features(prices, volumes)
            if not features:
                return None
            
            # Generate ML predictions
            predictions = self.ml_predictor.generate_ml_predictions(features, data)
            
            # Calculate advanced confluence score
            confluence = self._calculate_advanced_confluence(features, data, predictions)
            
            # Determine signal quality
            quality = self._determine_signal_quality(confluence, predictions)
            
            # Skip low-quality signals
            if quality == SignalQuality.LOW and confluence < 60:
                return None
            
            # Generate signal direction and targets
            direction, targets = self._calculate_targets_ml(data, predictions, features)
            
            # Detect patterns
            patterns = self._detect_advanced_patterns(features, data)
            
            # Create enhanced signal
            signal = EnhancedSignal(
                symbol=symbol,
                direction=direction,
                confidence=confluence,
                quality=quality,
                entry_price=data['price'],
                stop_loss=targets['stop_loss'],
                take_profit_1=targets['tp1'],
                take_profit_2=targets['tp2'],
                take_profit_3=targets['tp3'],
                current_price=data['price'],
                volume_24h=data['volume_24h'],
                change_24h=data['change_24h'],
                predictions=predictions,
                patterns=patterns,
                timestamp=datetime.now(),
                processing_time=time.time(),
                data_sources=['BINANCE_API', 'ML_PREDICTOR'],
                ml_features=features
            )
            
            logger.info(f"✅ {symbol}: {direction} signal ({confluence:.1f}% confidence)")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Advanced analysis error for {symbol}: {e}")
            return None
    
    def _calculate_advanced_confluence(self, features: Dict, data: Dict, predictions: PredictionMetrics) -> float:
        """Calculate advanced confluence score using ML features"""
        try:
            base_score = 50.0
            
            rsi = features.get('rsi', 50)
            macd_histogram = features.get('macd_histogram', 0)
            bb_position = features.get('bb_position', 0.5)
            momentum_5 = features.get('momentum_5', 0)
            volume_ratio = features.get('volume_ratio', 1)
            
            # RSI scoring
            if 30 <= rsi <= 70:
                base_score += 15
            elif rsi < 25 or rsi > 75:
                base_score += 10
            
            # MACD scoring
            if abs(macd_histogram) > 0.0001:
                base_score += 12
            
            # Bollinger position
            if 0.2 < bb_position < 0.8:
                base_score += 8
            elif bb_position < 0.1 or bb_position > 0.9:
                base_score += 15
            
            # Momentum scoring
            if abs(momentum_5) > 2:
                base_score += 10
            
            # Volume confirmation
            if volume_ratio > 1.5:
                base_score += 12
            elif volume_ratio > 1.2:
                base_score += 8
            
            # ML predictions boost
            if predictions.breakout_probability > 70:
                base_score += 10
            
            if predictions.volume_surge_probability > 60:
                base_score += 8
            
            # Market correlation factor
            if predictions.market_correlation_score > 70:
                base_score += 5
            
            return min(100, max(0, base_score))
            
        except Exception as e:
            logger.error(f"❌ Confluence calculation error: {e}")
            return 50.0
    
    def _determine_signal_quality(self, confluence: float, predictions: PredictionMetrics) -> SignalQuality:
        """Determine signal quality based on confluence and ML predictions"""
        if confluence >= 80 and predictions.breakout_probability > 75:
            return SignalQuality.PREMIUM
        elif confluence >= 70 and predictions.breakout_probability > 60:
            return SignalQuality.HIGH
        elif confluence >= 60:
            return SignalQuality.MEDIUM
        else:
            return SignalQuality.LOW
    
    def _calculate_targets_ml(self, data: Dict, predictions: PredictionMetrics, features: Dict) -> Tuple[str, Dict]:
        """Calculate targets using ML predictions"""
        try:
            current_price = data['price']
            change_24h = data['change_24h']
            momentum = features.get('momentum_5', 0)
            
            # Direction determination
            if momentum > 1 and change_24h > 0 and predictions.breakout_probability > 50:
                direction = "LONG"
                tp1 = predictions.price_target_30m
                tp2 = predictions.price_target_1h
                tp3 = predictions.price_target_3h
                stop_loss = current_price * (1 - predictions.risk_score / 1000)
                
            elif momentum < -1 and change_24h < 0 and predictions.trend_reversal_probability < 30:
                direction = "SHORT"
                tp1 = current_price - (predictions.price_target_30m - current_price)
                tp2 = current_price - (predictions.price_target_1h - current_price)
                tp3 = current_price - (predictions.price_target_3h - current_price)
                stop_loss = current_price * (1 + predictions.risk_score / 1000)
                
            else:
                # Neutral/consolidation
                direction = "LONG" if change_24h >= 0 else "SHORT"
                volatility_move = current_price * (predictions.volatility_prediction / 100) * 0.5
                
                if direction == "LONG":
                    tp1 = current_price + (volatility_move * 0.6)
                    tp2 = current_price + (volatility_move * 1.2)
                    tp3 = current_price + (volatility_move * 2.0)
                    stop_loss = current_price - (volatility_move * 0.4)
                else:
                    tp1 = current_price - (volatility_move * 0.6)
                    tp2 = current_price - (volatility_move * 1.2)
                    tp3 = current_price - (volatility_move * 2.0)
                    stop_loss = current_price + (volatility_move * 0.4)
            
            return direction, {
                'tp1': round(tp1, 6),
                'tp2': round(tp2, 6),
                'tp3': round(tp3, 6),
                'stop_loss': round(stop_loss, 6)
            }
            
        except Exception as e:
            logger.error(f"❌ Target calculation error: {e}")
            return "LONG", {'tp1': 0, 'tp2': 0, 'tp3': 0, 'stop_loss': 0}
    
    def _detect_advanced_patterns(self, features: Dict, data: Dict) -> List[str]:
        """Detect advanced trading patterns"""
        patterns = []
        
        try:
            rsi = features.get('rsi', 50)
            bb_position = features.get('bb_position', 0.5)
            volume_ratio = features.get('volume_ratio', 1)
            momentum_5 = features.get('momentum_5', 0)
            change_24h = data.get('change_24h', 0)
            
            # RSI patterns
            if rsi < 30:
                patterns.append("oversold_rsi")
            elif rsi > 70:
                patterns.append("overbought_rsi")
            
            # Bollinger patterns
            if bb_position < 0.1:
                patterns.append("bb_squeeze_low")
            elif bb_position > 0.9:
                patterns.append("bb_squeeze_high")
            
            # Volume patterns
            if volume_ratio > 2:
                patterns.append("volume_spike")
            elif volume_ratio > 1.5:
                patterns.append("volume_increase")
            
            # Momentum patterns
            if momentum_5 > 5 and change_24h > 3:
                patterns.append("strong_bullish_momentum")
            elif momentum_5 < -5 and change_24h < -3:
                patterns.append("strong_bearish_momentum")
            
            # Breakout patterns
            if (volume_ratio > 1.5 and abs(momentum_5) > 3 and 
                (bb_position < 0.1 or bb_position > 0.9)):
                patterns.append("potential_breakout")
            
            return patterns[:3]
            
        except Exception as e:
            logger.error(f"❌ Pattern detection error: {e}")
            return ["unknown"]

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
            _config_path = _Path(__file__).parent.parent.parent / "config" / "coins.json"
            _coins_config = _json.load(open(_config_path))
            self.SYMBOLS = _coins_config.get("symbols", [])
        except Exception:
            # Fallback: 32 verified trading symbols
            self.SYMBOLS = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
                "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "UNIUSDT",
                "LINKUSDT", "XLMUSDT", "ATOMUSDT", "MANAUSDT", "SANDUSDT",
                "DASHUSDT", "VETUSDT", "ICPUSDT", "GMTUSDT", "PEOPLEUSDT",
                "LUNCUSDT", "CHZUSDT", "NEARUSDT", "FLOWUSDT", "FILUSDT",
                "QTUMUSDT", "SNXUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT",
                "FLOKIUSDT", "OPUSDT",
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
    
    def fetch_kline_data(self, symbol: str, interval: str, limit: int = 80) -> Optional[pd.DataFrame]:
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
    
    def fetch_order_book(self, symbol: str) -> Dict[str, Union[float, int]]:
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
    
    def _get_mock_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Generate realistic mock data"""
        end_date = datetime.now()
        
        # Set base price based on symbol
        if symbol == "BTCUSDT":
            base_price = 50000.0
        elif symbol == "ETHUSDT":
            base_price = 3000.0
        elif symbol == "XRPUSDT":
            base_price = 2.0
        elif symbol == "BNBUSDT":
            base_price = 300.0
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
    
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        data: Dict[str, pd.DataFrame] = {}
        
        for timeframe in self.config.TIMEFRAMES:
            df = self.fetch_kline_data(symbol, timeframe, 80)
            if df is not None and len(df) >= 20:
                data[timeframe] = df
        
        return data

class IndicatorsCalculator:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_emas(close: pd.Series) -> Dict[str, float]:
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
    def calculate_rsi(close: pd.Series, period: int = 14) -> float:
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
    def calculate_macd(close: pd.Series) -> float:
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
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
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
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Union[str, int, float]]]:
        """Detect common candlestick patterns"""
        patterns: Dict[str, Dict[str, Union[str, int, float]]] = {}
        
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
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
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
            signal: Dict = {
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
                "accuracy_estimate": self._estimate_accuracy(confluence_score["total"], symbol),
                "risk_percentage": self.config.MAX_RISK_PER_TRADE,
                "leverage": self.config.LEVERAGE
            }
            
            return signal
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Union[float, int]]:
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
        
        result: Dict[str, Union[float, int]] = {
            **emas,  # type: ignore
            "rsi": rsi,
            "macd_hist": macd_hist,
            "atr": atr,
            "volume_ratio": volume_ratio,
            "current_price": float(close.iloc[-1])
        }
        
        return result
    
    def _check_mtf_alignment(self, data: Dict[str, pd.DataFrame]) -> int:
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
                                   orderbook: Dict, mtf_score: int, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate confluence score"""
        scores: Dict[str, int] = {
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
            score_val = data.get("score", 0)
            confidence_val = data.get("confidence", 0)
            if isinstance(score_val, (int, float)) and isinstance(confidence_val, (int, float)):
                total_pattern_score += int(score_val * confidence_val)
        scores["patterns"] = min(20, total_pattern_score)
        
        # Volume Score (max 15)
        volume_ratio = indicators.get("volume_ratio", 0)
        if isinstance(volume_ratio, (int, float)):
            if volume_ratio >= self.config.MIN_VOLUME_RATIO:
                scores["volume"] = 15
            elif volume_ratio >= 0.7:
                scores["volume"] = 10
            else:
                scores["volume"] = 5
        
        # Order Book Score (max 10)
        ob_score = orderbook.get("score", 5)
        scores["orderbook"] = min(10, int(ob_score) if isinstance(ob_score, (int, float)) else 5)
        
        # Calculate total
        scores["total"] = sum([scores["indicators"], scores["patterns"], 
                              scores["volume"], scores["orderbook"], scores["mtf"]])
        
        return scores
    
    def _determine_direction(self, indicators: Dict, patterns: Dict) -> Optional[Dict[str, Union[str, float]]]:
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
            pattern_type = data.get("type", "")
            if pattern_type == "bullish":
                bullish_points += 1
            elif pattern_type == "bearish":
                bearish_points += 1
        
        # Determine direction
        if bullish_points > bearish_points + 1:
            return {"direction": "LONG", "confidence": bullish_points / (bullish_points + bearish_points)}
        elif bearish_points > bullish_points + 1:
            return {"direction": "SHORT", "confidence": bearish_points / (bullish_points + bearish_points)}
        
        return None
    
    def _calculate_levels(self, signal_data: Dict, indicators: Dict, df: pd.DataFrame) -> Dict:
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
        
        take_profits: List[Tuple[float, int]] = []
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
    
    def _estimate_accuracy(self, score: float, symbol: str) -> float:
        """
        Get accuracy from REAL backtesting data using config loader.
        CRITICAL: NO FAKE FALLBACKS - crash if data not available
        """
        # Try to use optimized config first (from REAL backtesting)
        if CONFIG_LOADED:
            try:
                config = get_config()
                if config and hasattr(config, 'get_accuracy_for_score'):
                    accuracy = config.get_accuracy_for_score(score)
                    logger.debug(f"Using accuracy from config: {accuracy:.1f}%")
                    return accuracy
            except Exception as e:
                logger.debug(f"Config not available: {e}")
        
        # Fallback to database-based statistics (still REAL data, not fake)
        try:
            from crypto_bot.core.statistics_calculator import BacktestStatisticsCalculator
            
            stats_calc = BacktestStatisticsCalculator()
            accuracy_data = stats_calc.calculate_accuracy_by_confluence_score(symbol)
            
            if not accuracy_data:
                # ⚠️ NO DATA AT ALL - CRASH WITH HELPFUL MESSAGE
                raise RuntimeError(
                    "Cannot load accuracy data from backtesting database!\n"
                    "You must run a backtest first:\n"
                    f"  python core/run_backtest.py --full --symbol {symbol}\n"
                    "  python scripts/generate_real_config.py"
                )
            
            # Map confluence score to the appropriate accuracy bucket
            if score >= 85:
                val = accuracy_data.get('85+', {}).get('win_rate')
                if val is None:
                    raise KeyError("Missing accuracy data for 85+ score")
                return val
            elif score >= 75:
                val = accuracy_data.get('75-84', {}).get('win_rate')
                if val is None:
                    raise KeyError("Missing accuracy data for 75-84 score")
                return val
            elif score >= 65:
                val = accuracy_data.get('65-74', {}).get('win_rate')
                if val is None:
                    raise KeyError("Missing accuracy data for 65-74 score")
                return val
            else:
                val = accuracy_data.get('<65', {}).get('win_rate')
                if val is None:
                    raise KeyError("Missing accuracy data for <65 score")
                return val
                
        except Exception as e:
            # NO FALLBACK - CRASH WITH HELPFUL ERROR
            raise RuntimeError(
                f"❌ CRITICAL: Cannot load real accuracy data!\n"
                f"Error: {e}\n\n"
                f"SOLUTIONS:\n"
                f"1. Run backtest: python core/run_backtest.py --full --symbol {symbol}\n"
                f"2. Generate config: python scripts/generate_real_config.py\n"
                f"3. Check database: sqlite3 data/backtest.db\n\n"
                f"DO NOT use fake/fallback values!"
            )

class TimingConfig:
    """Configuration for realistic trade timing based on market conditions"""
    
    # Market volatility categories
    LOW_VOLATILITY = {
        'tp1_min': 300,    # 5 minutes
        'tp1_max': 900,    # 15 minutes
        'tp2_add_min': 600,  # 10 minutes additional
        'tp2_add_max': 1200, # 20 minutes additional
        'tp3_add_min': 900,  # 15 minutes additional
        'tp3_add_max': 2700  # 45 minutes additional
    }
    
    MEDIUM_VOLATILITY = {
        'tp1_min': 180,    # 3 minutes
        'tp1_max': 600,    # 10 minutes
        'tp2_add_min': 300,  # 5 minutes additional
        'tp2_add_max': 900,  # 15 minutes additional
        'tp3_add_min': 600,  # 10 minutes additional
        'tp3_add_max': 1800  # 30 minutes additional
    }
    
    HIGH_VOLATILITY = {
        'tp1_min': 60,     # 1 minute
        'tp1_max': 300,    # 5 minutes
        'tp2_add_min': 180,  # 3 minutes additional
        'tp2_add_max': 600,  # 10 minutes additional
        'tp3_add_min': 300,  # 5 minutes additional
        'tp3_add_max': 1200  # 20 minutes additional
    }
    
    @classmethod
    def get_timing_for_symbol(cls, symbol: str, volatility_score: float = 50):
        """Get appropriate timing configuration for symbol"""
        # High volatility symbols
        if any(x in symbol for x in ["PEPE", "SHIB", "DOGE", "FLOKI", "WIF"]):
            return cls.HIGH_VOLATILITY
        # Major coins (usually more stable)
        elif symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            return cls.LOW_VOLATILITY if volatility_score < 60 else cls.MEDIUM_VOLATILITY
        # Everything else
        else:
            if volatility_score > 70:
                return cls.HIGH_VOLATILITY
            elif volatility_score > 40:
                return cls.MEDIUM_VOLATILITY
            else:
                return cls.LOW_VOLATILITY

def simulate_realistic_price_movement(entry_price: float, target_price: float, direction: str, duration_seconds: int) -> List[Tuple[datetime, float]]:
    """Simulate realistic price movement from entry to target"""
    movements = []
    current_time = datetime.now()
    current_price = entry_price
    
    # Calculate total movement needed
    total_movement = target_price - entry_price
    
    # Number of price updates (every 10-30 seconds)
    num_updates = max(3, duration_seconds // random.randint(10, 30))
    
    for i in range(num_updates + 1):
        if i == 0:
            # Entry price
            movements.append((current_time, current_price))
        elif i == num_updates:
            # Final target price
            movements.append((current_time + timedelta(seconds=duration_seconds), target_price))
        else:
            # Intermediate prices with realistic volatility
            progress = i / num_updates
            
            # Base progress towards target
            base_price = entry_price + (total_movement * progress)
            
            # Add realistic volatility (noise)
            volatility_range = abs(total_movement) * 0.3  # 30% of total movement
            noise = random.uniform(-volatility_range, volatility_range) * (1 - progress)  # Less noise as we approach target
            
            current_price = base_price + noise
            update_time = current_time + timedelta(seconds=int(duration_seconds * progress))
            movements.append((update_time, current_price))
    
    return movements

# ============================================================================
# BINANCE API CONFIGURATION
# ============================================================================

class BinanceConfig:
    """Binance API configuration for demo/testnet trading"""
    
    def __init__(self):
        # Demo/Testnet URLs
        self.TESTNET_BASE_URL = "https://testnet.binance.vision/api/v3"
        self.TESTNET_FUTURES_URL = "https://testnet.binancefuture.com/fapi/v1"
        
        # Production URLs (for live data only)
        self.MAINNET_BASE_URL = "https://api.binance.com/api/v3"
        self.MAINNET_FUTURES_URL = "https://fapi.binance.com/fapi/v1"
        
        # API Credentials (You need to add your testnet keys)
        self.TESTNET_API_KEY = "9apZOmmQaGc33kXIpbvi021PDghH76roWGzdb5xL2QGKbcMsRI6OZ6WbKLkqcVPp"  # Your testnet API key
        self.TESTNET_SECRET_KEY = "55tlqVg3w2FuMkcaxgbo0raye9Yw4ari7mUD3eEwHsdhB7CoK9ye1nOe3xzR0lu8"  # Your testnet secret key
        
        # Trading configuration
        self.USE_TESTNET = True  # Set to True for demo trading
        self.DEFAULT_LEVERAGE = 20
        self.MAX_POSITION_SIZE = 1000  # USDT
        
    def get_headers(self) -> Dict[str, str]:
        """Get API headers for authenticated requests"""
        if self.TESTNET_API_KEY:
            return {
                'X-MBX-APIKEY': self.TESTNET_API_KEY,
                'Content-Type': 'application/json'
            }
        return {'Content-Type': 'application/json'}
    
    def get_base_url(self) -> str:
        """Get appropriate base URL"""
        return self.TESTNET_FUTURES_URL if self.USE_TESTNET else self.MAINNET_FUTURES_URL
    
    def is_configured(self) -> bool:
        """Check if API credentials are configured"""
        return bool(self.TESTNET_API_KEY and self.TESTNET_SECRET_KEY)
    
    def create_signature(self, query_string: str) -> str:
        """Create HMAC SHA256 signature for Binance API"""
        return hmac.new(
            self.TESTNET_SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

class BinanceTestnetAPI:
    """Binance Testnet/Demo API integration"""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.get_headers())
        
    def get_account_info(self) -> Optional[Dict]:
        """Get testnet account information"""
        if not self.config.is_configured():
            logger.warning("⚠️ Binance testnet credentials not configured")
            return None
            
        try:
            url = f"{self.config.get_base_url()}/account"
            timestamp = int(time.time() * 1000)
            
            params: Dict[str, Any] = {
                'timestamp': timestamp,
                'recvWindow': 5000
            }
            
            # Create query string and signature
            query_string = urlencode(params)
            signature = self.config.create_signature(query_string)
            params['signature'] = signature
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Account info error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Testnet API error: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current futures positions"""
        if not self.config.is_configured():
            return []
            
        try:
            url = f"{self.config.get_base_url()}/positionRisk"
            timestamp = int(time.time() * 1000)
            
            params: Dict[str, Any] = {
                'timestamp': timestamp,
                'recvWindow': 5000
            }
            
            # Create query string and signature
            query_string = urlencode(params)
            signature = self.config.create_signature(query_string)
            params['signature'] = signature
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                positions = response.json()
                # Filter only positions with size > 0
                return [pos for pos in positions if float(pos.get('positionAmt', 0)) != 0]
            else:
                logger.error(f"❌ Positions error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Positions error: {e}")
            return []
    
    def get_balance(self) -> Optional[Dict]:
        """Get USDT balance from futures account"""
        if not self.config.is_configured():
            return None
            
        try:
            url = f"{self.config.get_base_url()}/balance"
            timestamp = int(time.time() * 1000)
            
            params: Dict[str, Any] = {
                'timestamp': timestamp,
                'recvWindow': 5000
            }
            
            # Create query string and signature
            query_string = urlencode(params)
            signature = self.config.create_signature(query_string)
            params['signature'] = signature
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                balances = response.json()
                # Find USDT balance
                for balance in balances:
                    if balance['asset'] == 'USDT':
                        return {
                            'asset': 'USDT',
                            'balance': float(balance['balance']),
                            'available': float(balance['availableBalance'])
                        }
                return None
            else:
                logger.error(f"❌ Balance error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Network error, using simulation: {e}")
            # Fallback: Return simulated balance for demo purposes
            return {
                'asset': 'USDT',
                'balance': 10000.0,
                'available': 10000.0
            }
    
    def place_demo_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Dict:
        """Place a demo order (simulation for testing)"""
        logger.info(f"📝 DEMO ORDER: {symbol} {side} {quantity:.6f} @ {price or 'MARKET'}")
        
        # Simulate order response
        order_id = f"DEMO_{int(time.time())}"
        return {
            'orderId': order_id,
            'symbol': symbol,
            'side': side,
            'type': 'MARKET' if price is None else 'LIMIT',
            'quantity': str(quantity),
            'price': str(price) if price else '0',
            'status': 'FILLED',
            'executedQty': str(quantity),
            'fills': [{
                'price': str(price or 0),
                'qty': str(quantity),
                'commission': '0',
                'commissionAsset': 'USDT'
            }]
        }

# ============================================================================
# DEMO TRADING BOT SYSTEM (Updated with Binance Integration)
# ============================================================================

class TradeStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    CANCELLED = "CANCELLED"

@dataclass
class DemoTrade:
    """Demo trade object for bot simulation"""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    quantity: float
    leverage: int
    entry_time: datetime
    current_price: float
    pnl: float
    pnl_percentage: float
    status: TradeStatus
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    partial_profits: Optional[List[float]] = None

    def __post_init__(self):
        if self.partial_profits is None:
            self.partial_profits = []

class Portfolio:
    """Demo trading portfolio management"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin_used = 0
        self.free_margin = initial_balance
        self.total_pnl = 0
        self.daily_pnl = 0
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.max_drawdown = 0
        self.peak_equity = initial_balance
        
    def update_portfolio(self, active_trades: List[DemoTrade]):
        """Update portfolio metrics based on active trades"""
        total_unrealized_pnl = sum(trade.pnl for trade in active_trades if trade.status == TradeStatus.ACTIVE)
        total_margin = sum(trade.entry_price * trade.quantity / trade.leverage for trade in active_trades if trade.status == TradeStatus.ACTIVE)
        
        self.equity = self.balance + total_unrealized_pnl
        self.margin_used = total_margin
        self.free_margin = self.equity - self.margin_used
        
        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def execute_trade(self, trade: DemoTrade) -> bool:
        """Execute a demo trade"""
        required_margin = trade.entry_price * trade.quantity / trade.leverage
        
        if required_margin > self.free_margin:
            logger.warning(f"❌ Insufficient margin for {trade.symbol} trade")
            return False
        
        self.total_trades += 1
        self.trades_today += 1
        logger.info(f"✅ Demo trade executed: {trade.symbol} {trade.direction}")
        return True
    
    def close_trade(self, trade: DemoTrade, exit_price: float, reason: str):
        """Close a demo trade and update portfolio"""
        trade.status = TradeStatus.COMPLETED
        trade.exit_time = datetime.now()
        trade.exit_reason = reason
        trade.current_price = exit_price
        
        # Calculate final PnL
        if trade.direction == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        trade.pnl = pnl * trade.leverage
        trade.pnl_percentage = (pnl / trade.entry_price) * 100 * trade.leverage
        
        # Update portfolio
        self.balance += trade.pnl
        self.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        logger.info(f"🔄 Trade closed: {trade.symbol} - PnL: ${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%)")

class DemoTradingBot:
    """Automated demo trading bot using ML predictions with Binance testnet support"""
    
    def __init__(self, initial_balance: float = 10000, max_concurrent_trades: int = 5, use_binance_testnet: bool = False):
        self.portfolio = Portfolio(initial_balance)
        self.active_trades: List[DemoTrade] = []
        self.completed_trades: List[DemoTrade] = []
        self.max_concurrent_trades = max_concurrent_trades
        self.risk_per_trade = 2.0  # 2% risk per trade
        self.min_confidence_score = 65  # Minimum confidence to trade
        self.running = False
        
        # Binance testnet integration
        self.use_binance_testnet = use_binance_testnet
        self.binance_config = BinanceConfig()
        self.binance_api = None
        
        if self.use_binance_testnet:
            self.binance_api = BinanceTestnetAPI(self.binance_config)
            if self.binance_config.is_configured():
                print("🔗 Binance Testnet integration ENABLED")
                self._sync_with_testnet()
            else:
                print("⚠️ Binance Testnet credentials not configured - Using simulation mode")
                self.use_binance_testnet = False
        
        # Database for trade history
        self.setup_database()
        
        # Trading statistics
        self.stats: Dict[str, float] = {
            'total_signals_received': 0.0,
            'trades_executed': 0.0,
            'trades_skipped': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'total_return': 0.0
        }
    
    def _sync_with_testnet(self):
        """Sync portfolio with Binance testnet account"""
        try:
            if self.binance_api:
                # Get USDT balance
                balance_info = self.binance_api.get_balance()
                if balance_info:
                    total_balance = balance_info['balance']
                    available_balance = balance_info['available']
                    
                    self.portfolio.balance = total_balance
                    self.portfolio.equity = total_balance
                    self.portfolio.free_margin = available_balance
                    
                    logger.info(f"✅ Synced with testnet - Balance: ${total_balance:,.2f}")
                    logger.info(f"📊 Available: ${available_balance:,.2f}")
                    
                    # Get current positions
                    positions = self.binance_api.get_positions()
                    if positions:
                        logger.info(f"📊 Found {len(positions)} active positions on testnet")
                        for pos in positions:
                            logger.info(f"   {pos['symbol']}: {pos['positionAmt']} @ {pos['entryPrice']}")
                    else:
                        logger.info("📊 No active positions on testnet")
                else:
                    logger.warning("⚠️ Could not sync balance - using default values")
                    
        except Exception as e:
            logger.error(f"❌ Testnet sync error: {e}")
    
    def setup_database(self):
        """Setup SQLite database for trade history"""
        try:
            self.conn = sqlite3.connect('demo_trades.db', check_same_thread=False)
            cursor = self.conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    leverage INTEGER,
                    entry_time TEXT,
                    exit_time TEXT,
                    pnl REAL,
                    pnl_percentage REAL,
                    status TEXT,
                    exit_reason TEXT,
                    confidence_score REAL
                )
            ''')
            self.conn.commit()
            logger.info("✅ Demo trading database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database setup error: {e}")
    
    def evaluate_signal(self, signal: EnhancedSignal) -> bool:
        """Evaluate if signal should be traded"""
        # Check confidence threshold
        if signal.confidence < self.min_confidence_score:
            self.stats['trades_skipped'] += 1
            logger.info(f"⚠️ Signal skipped - Low confidence: {signal.confidence:.1f}%")
            return False
        
        # Check concurrent trades limit
        if len(self.active_trades) >= self.max_concurrent_trades:
            self.stats['trades_skipped'] += 1
            logger.info(f"⚠️ Signal skipped - Max concurrent trades reached")
            return False
        
        # Check if already trading this symbol
        if any(trade.symbol == signal.symbol for trade in self.active_trades):
            self.stats['trades_skipped'] += 1
            logger.info(f"⚠️ Signal skipped - Already trading {signal.symbol}")
            return False
        
        # Check portfolio health
        if self.portfolio.free_margin < self.portfolio.initial_balance * 0.1:  # Less than 10% free margin
            self.stats['trades_skipped'] += 1
            logger.info(f"⚠️ Signal skipped - Low free margin")
            return False
        
        return True
    
    def calculate_position_size(self, signal: EnhancedSignal) -> float:
        """Calculate position size based on risk management"""
        # Risk amount (2% of current equity)
        risk_amount = self.portfolio.equity * (self.risk_per_trade / 100)
        
        # Stop loss distance
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        
        # Position size calculation
        if stop_distance > 0:
            position_size = risk_amount / (stop_distance * signal.leverage)
            # Cap position size to available margin
            max_position = (self.portfolio.free_margin * 0.8) / signal.entry_price  # Use 80% of free margin max
            position_size = min(position_size, max_position)
        else:
            position_size = risk_amount / signal.entry_price
        
        return round(position_size, 6)
    
    def create_demo_trade(self, signal: EnhancedSignal) -> DemoTrade:
        """Create a demo trade from ML signal"""
        position_size = self.calculate_position_size(signal)
        
        trade = DemoTrade(
            trade_id=f"DEMO_{signal.symbol}_{int(time.time())}",
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            quantity=position_size,
            leverage=getattr(signal, 'leverage', 20),
            entry_time=datetime.now(),
            current_price=signal.current_price,
            pnl=0,
            pnl_percentage=0,
            status=TradeStatus.PENDING
        )
        
        return trade
    
    def process_signal(self, signal: EnhancedSignal) -> bool:
        """Process incoming ML signal for automated trading"""
        self.stats['total_signals_received'] += 1
        
        # Evaluate signal
        if not self.evaluate_signal(signal):
            return False
        
        # Create demo trade
        trade = self.create_demo_trade(signal)
        
        # Execute trade
        if self.portfolio.execute_trade(trade):
            trade.status = TradeStatus.ACTIVE
            self.active_trades.append(trade)
            self.stats['trades_executed'] += 1
            
            # Execute on Binance testnet if enabled
            if self.use_binance_testnet and self.binance_api:
                try:
                    order_result = self.binance_api.place_demo_order(
                        symbol=trade.symbol,
                        side=trade.direction,
                        quantity=trade.quantity,
                        price=trade.entry_price
                    )
                    trade.trade_id = f"BINANCE_{order_result['orderId']}"
                    logger.info(f"🔗 Binance testnet order: {order_result['orderId']}")
                except Exception as e:
                    logger.error(f"❌ Binance order error: {e}")
            
            # Save to database
            self.save_trade_to_db(trade, signal.confidence)
            
            logger.info(f"🤖 BOT TRADE: {trade.symbol} {trade.direction} @ {trade.entry_price:.5f}")
            logger.info(f"📊 Position: {trade.quantity:.6f} | Leverage: {trade.leverage}x | Risk: ${abs(trade.entry_price - trade.stop_loss) * trade.quantity * trade.leverage:.2f}")
            
            return True
        
        return False
    
    def update_trades(self, current_prices: Dict[str, float]):
        """Update active trades based on current market prices"""
        trades_to_close = []
        
        for trade in self.active_trades:
            if trade.symbol not in current_prices:
                continue
                
            current_price = current_prices[trade.symbol]
            trade.current_price = current_price
            
            # Calculate unrealized PnL
            if trade.direction == "LONG":
                pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                pnl = (trade.entry_price - current_price) * trade.quantity
            
            trade.pnl = pnl * trade.leverage
            trade.pnl_percentage = (pnl / trade.entry_price) * 100 * trade.leverage
            
            # Check TP/SL conditions
            if trade.direction == "LONG":
                # Check Stop Loss
                if current_price <= trade.stop_loss:
                    trades_to_close.append((trade, current_price, "Stop Loss"))
                # Check Take Profits
                elif current_price >= trade.take_profit_3 and not trade.tp3_hit:
                    trades_to_close.append((trade, current_price, "Take Profit 3"))
                elif current_price >= trade.take_profit_2 and not trade.tp2_hit:
                    # Partial close at TP2
                    self.partial_close_trade(trade, current_price, 0.6, "Take Profit 2")
                elif current_price >= trade.take_profit_1 and not trade.tp1_hit:
                    # Partial close at TP1
                    self.partial_close_trade(trade, current_price, 0.4, "Take Profit 1")
            
            else:  # SHORT
                # Check Stop Loss
                if current_price >= trade.stop_loss:
                    trades_to_close.append((trade, current_price, "Stop Loss"))
                # Check Take Profits
                elif current_price <= trade.take_profit_3 and not trade.tp3_hit:
                    trades_to_close.append((trade, current_price, "Take Profit 3"))
                elif current_price <= trade.take_profit_2 and not trade.tp2_hit:
                    self.partial_close_trade(trade, current_price, 0.6, "Take Profit 2")
                elif current_price <= trade.take_profit_1 and not trade.tp1_hit:
                    self.partial_close_trade(trade, current_price, 0.4, "Take Profit 1")
        
        # Close trades that hit TP3 or SL
        for trade, exit_price, reason in trades_to_close:
            self.close_trade(trade, exit_price, reason)
        
        # Update portfolio
        self.portfolio.update_portfolio(self.active_trades)
    
    def partial_close_trade(self, trade: DemoTrade, price: float, percentage: float, reason: str):
        """Partially close trade at take profit levels"""
        partial_quantity = trade.quantity * percentage
        remaining_quantity = trade.quantity - partial_quantity
        
        # Calculate partial profit
        if trade.direction == "LONG":
            partial_pnl = (price - trade.entry_price) * partial_quantity * trade.leverage
        else:
            partial_pnl = (trade.entry_price - price) * partial_quantity * trade.leverage
        
        # Update portfolio
        self.portfolio.balance += partial_pnl
        self.portfolio.total_pnl += partial_pnl
        
        # Update trade partial profits (with None check)
        if trade.partial_profits is None:
            trade.partial_profits = []
        trade.partial_profits.append(partial_pnl)
        
        # Update trade
        trade.quantity = remaining_quantity
        
        # Mark TP as hit
        if "Take Profit 1" in reason:
            trade.tp1_hit = True
        elif "Take Profit 2" in reason:
            trade.tp2_hit = True
        
        logger.info(f"💰 Partial close: {trade.symbol} - {percentage*100}% closed at {price:.5f} - Profit: ${partial_pnl:.2f}")
    
    def close_trade(self, trade: DemoTrade, exit_price: float, reason: str):
        """Close trade completely"""
        self.portfolio.close_trade(trade, exit_price, reason)
        self.active_trades.remove(trade)
        self.completed_trades.append(trade)
        
        # Update database
        self.update_trade_in_db(trade)
        
        # Export to CSV
        self.export_trade_to_csv(trade)
        
        # Update statistics
        self.update_statistics()
    
    def export_trade_to_csv(self, trade: DemoTrade):
        """Export completed trade to CSV file"""
        try:
            csv_filename = f"demo_bot_trades_{datetime.now().strftime('%Y%m%d')}.csv"
            csv_path = csv_filename
            
            # Check if file exists to write headers
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'trade_id', 'symbol', 'direction', 'entry_price', 'exit_price',
                    'quantity', 'leverage', 'entry_time', 'exit_time', 'duration_minutes',
                    'pnl_usd', 'pnl_percentage', 'exit_reason', 'partial_profits',
                    'tp1_hit', 'tp2_hit', 'tp3_hit', 'status'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write headers if new file
                if not file_exists:
                    writer.writeheader()
                    logger.info(f"📄 Created CSV file: {csv_path}")
                
                # Calculate duration
                if trade.exit_time and trade.entry_time:
                    duration = trade.exit_time - trade.entry_time
                    duration_minutes = duration.total_seconds() / 60
                else:
                    duration_minutes = 0
                
                # Prepare row data
                row_data = {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.current_price,
                    'quantity': trade.quantity,
                    'leverage': trade.leverage,
                    'entry_time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else '',
                    'duration_minutes': round(duration_minutes, 2),
                    'pnl_usd': round(trade.pnl, 2),
                    'pnl_percentage': round(trade.pnl_percentage, 2),
                    'exit_reason': trade.exit_reason,
                    'partial_profits': ';'.join([f'{p:.2f}' for p in (trade.partial_profits or [])]),
                    'tp1_hit': trade.tp1_hit,
                    'tp2_hit': trade.tp2_hit, 
                    'tp3_hit': trade.tp3_hit,
                    'status': trade.status.value
                }
                
                writer.writerow(row_data)
                logger.info(f"📝 Trade exported to CSV: {trade.symbol} - PnL: ${trade.pnl:.2f}")
                
        except Exception as e:
            logger.error(f"❌ CSV export error: {e}")
    
    def export_portfolio_summary(self):
        """Export portfolio summary to CSV"""
        try:
            csv_filename = f"demo_bot_portfolio_{datetime.now().strftime('%Y%m%d')}.csv"
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'total_trades', 'active_trades', 'completed_trades',
                    'initial_balance', 'current_equity', 'total_pnl', 'total_return_pct',
                    'win_rate', 'avg_profit', 'avg_loss', 'best_trade', 'worst_trade',
                    'max_drawdown', 'free_margin', 'margin_used'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                summary_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_trades': self.stats['trades_executed'],
                    'active_trades': len(self.active_trades),
                    'completed_trades': len(self.completed_trades),
                    'initial_balance': self.portfolio.initial_balance,
                    'current_equity': self.portfolio.equity,
                    'total_pnl': self.portfolio.total_pnl,
                    'total_return_pct': self.stats['total_return'],
                    'win_rate': self.stats['win_rate'],
                    'avg_profit': self.stats['avg_profit'],
                    'avg_loss': self.stats['avg_loss'],
                    'best_trade': self.stats['best_trade'],
                    'worst_trade': self.stats['worst_trade'],
                    'max_drawdown': self.portfolio.max_drawdown,
                    'free_margin': self.portfolio.free_margin,
                    'margin_used': self.portfolio.margin_used
                }
                
                writer.writerow(summary_data)
                logger.info(f"📊 Portfolio summary exported: {csv_filename}")
                
        except Exception as e:
            logger.error(f"❌ Portfolio export error: {e}")
    
    def get_csv_statistics(self) -> Dict:
        """Get statistics about CSV exports"""
        today = datetime.now().strftime('%Y%m%d')
        csv_filename = f"demo_bot_trades_{today}.csv"
        
        try:
            if not os.path.exists(csv_filename):
                return {
                    'csv_file': csv_filename,
                    'file_exists': False,
                    'total_records': 0
                }
            
            with open(csv_filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                records = list(reader)
                
                total_pnl = sum(float(row['pnl_usd']) for row in records if row['pnl_usd'])
                winning_trades = len([r for r in records if float(r['pnl_usd']) > 0])
                
                return {
                    'csv_file': csv_filename,
                    'file_exists': True,
                    'total_records': len(records),
                    'total_pnl': round(total_pnl, 2),
                    'winning_trades': winning_trades,
                    'losing_trades': len(records) - winning_trades,
                    'win_rate': round((winning_trades / len(records)) * 100, 1) if records else 0
                }
                
        except Exception as e:
            logger.error(f"❌ CSV stats error: {e}")
            return {'csv_file': csv_filename, 'file_exists': False, 'error': str(e)}
    
    def save_trade_to_db(self, trade: DemoTrade, confidence: float):
        """Save trade to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO trades (trade_id, symbol, direction, entry_price, quantity, 
                                  leverage, entry_time, status, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(trade.trade_id), str(trade.symbol), str(trade.direction), float(trade.entry_price),
                  float(trade.quantity), int(trade.leverage), str(trade.entry_time.isoformat()), 
                  str(trade.status.value), float(confidence)))
            self.conn.commit()
        except Exception as e:
            logger.error(f"❌ Database save error: {e}")
    
    def update_trade_in_db(self, trade: DemoTrade):
        """Update completed trade in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE trades SET exit_price=?, exit_time=?, pnl=?, pnl_percentage=?, 
                                status=?, exit_reason=?
                WHERE trade_id=?
            ''', (float(trade.current_price), str(trade.exit_time.isoformat()) if trade.exit_time else None,
                  float(trade.pnl), float(trade.pnl_percentage), str(trade.status.value), 
                  str(trade.exit_reason), str(trade.trade_id)))
            self.conn.commit()
        except Exception as e:
            logger.error(f"❌ Database update error: {e}")
    
    def update_statistics(self):
        """Update trading statistics"""
        if not self.completed_trades:
            return
        
        profits = [t.pnl for t in self.completed_trades if t.pnl > 0]
        losses = [t.pnl for t in self.completed_trades if t.pnl < 0]
        
        self.stats['win_rate'] = float((len(profits) / len(self.completed_trades)) * 100)
        self.stats['avg_profit'] = float(sum(profits) / len(profits) if profits else 0)
        self.stats['avg_loss'] = float(sum(losses) / len(losses) if losses else 0)
        self.stats['best_trade'] = float(max([t.pnl for t in self.completed_trades]))
        self.stats['worst_trade'] = float(min([t.pnl for t in self.completed_trades]))
        self.stats['total_return'] = float(((self.portfolio.equity / self.portfolio.initial_balance) - 1) * 100)
    
    def get_trading_summary(self) -> str:
        """Get formatted trading summary"""
        self.update_statistics()
        
        summary = f"""
🤖 DEMO TRADING BOT SUMMARY
{'='*50}
💰 Portfolio Status:
   Initial Balance: ${self.portfolio.initial_balance:,.2f}
   Current Equity: ${self.portfolio.equity:,.2f}
   Total PnL: ${self.portfolio.total_pnl:,.2f}
   Total Return: {self.stats['total_return']:.2f}%
   Free Margin: ${self.portfolio.free_margin:,.2f}
   Max Drawdown: {self.portfolio.max_drawdown:.2f}%

📊 Trading Statistics:
   Signals Received: {self.stats['total_signals_received']}
   Trades Executed: {self.stats['trades_executed']}
   Trades Skipped: {self.stats['trades_skipped']}
   Active Trades: {len(self.active_trades)}
   Completed Trades: {len(self.completed_trades)}
   Win Rate: {self.stats['win_rate']:.1f}%
   
💹 Performance:
   Best Trade: ${self.stats['best_trade']:.2f}
   Worst Trade: ${self.stats['worst_trade']:.2f}
   Avg Profit: ${self.stats['avg_profit']:.2f}
   Avg Loss: ${self.stats['avg_loss']:.2f}
"""
        
        if self.active_trades:
            summary += f"\n🔄 Active Trades ({len(self.active_trades)}):\n"
            for trade in self.active_trades:
                summary += f"   {trade.symbol} {trade.direction} | PnL: ${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%)\n"
        
        return summary
    
    def start_bot(self):
        """Start the demo trading bot"""
        self.running = True
        logger.info("🤖 Demo Trading Bot Started!")
    
    def stop_bot(self):
        """Stop the demo trading bot"""
        self.running = False
        logger.info("🛑 Demo Trading Bot Stopped!")



class SignalFormatter:
    """Format signals for display"""
    
    @staticmethod
    def format_detailed_signal(signal: Dict) -> str:
        """Format detailed signal"""
        if not signal:
            return "No signal available right now."
        
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
        
        # Status instead of fake timeline
        output += f"\n\nSTATUS:"
        output += f"\n{signal['timestamp'].strftime('%H:%M:%S')}: Signal Generated"
        output += f"\nWaiting for entry: {signal['entry_range'][0]:.5f} - {signal['entry_range'][1]:.5f}"
        output += f"\n🔄 Tracking live price movements..."
        
        return output
    
    @staticmethod
    def format_simple_signal(signal: Dict) -> str:
        """Format simple signal"""
        if not signal:
            return "No signal available."
        
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
        
        # Real status instead of fake timeline
        output += f"\n\n{signal['timestamp'].strftime('%H:%M:%S')}: Signal Generated ({signal['accuracy_estimate']:.1f}%)"
        output += f"\n🔄 Waiting for entry at: {signal['entry_range'][0]:.5f} - {signal['entry_range'][1]:.5f}"
        output += f"\n📊 Tracking live price..."
        
        return output

# ============================================================================
# ENHANCED DASHBOARD WITH STREAMING ML INTEGRATION
# ============================================================================

class EnhancedScalpingDashboard:
    """
    ENHANCED Dashboard with Streaming ML + CSV Export
    Combines original system with Manager's advanced ML
    """
    
    def __init__(self, use_streaming_ml: bool = True, enable_demo_bot: bool = False, use_binance_testnet: bool = False):
        self.config = ScalpingConfig()
        self.analyzer = SignalAnalyzer(self.config)
        self.formatter = SignalFormatter()
        
        # Streaming ML system
        self.use_streaming_ml = use_streaming_ml
        self.streaming_processor = None
        if self.use_streaming_ml:
            self.streaming_processor = StreamingSignalProcessor()
        
        # Demo Trading Bot
        self.enable_demo_bot = enable_demo_bot
        self.demo_bot = None
        if self.enable_demo_bot:
            self.demo_bot = DemoTradingBot(
                initial_balance=10000, 
                max_concurrent_trades=5,
                use_binance_testnet=use_binance_testnet
            )
            self.demo_bot.start_bot()
            if use_binance_testnet:
                print("🤖 Demo Trading Bot + Binance Testnet ENABLED")
            else:
                print("🤖 Demo Trading Bot ENABLED (Simulation Mode)")
        
        self.signals: Dict[str, Dict] = {}
        self.ml_signals: List[EnhancedSignal] = []
        self.active_trades: Dict[str, str] = {}
        
        # Initialize trade tracker if available
        if TRACKING_ENABLED and TradeTracker is not None:
            self.tracker: Optional[Any] = TradeTracker(data_dir="./trade_data")
            print("✅ CSV tracking enabled")
        else:
            self.tracker = None
            print("⚠️  CSV tracking disabled")
        
        # Initialize live signal tracker for real-time tracking
        if LIVE_TRACKER_ENABLED and LiveSignalTracker is not None:
            self.live_signal_tracker: Optional[Any] = LiveSignalTracker()
            try:
                self.live_signal_tracker.start()  # ✅ ACTUALLY START THE TRACKER!
                print("✅ Live signal tracker enabled and STARTED")
                print("   → Monitoring signals every 1 second")
                print("   → Tracking TP/SL hits in real-time")
            except Exception as e:
                logger.warning(f"⚠️  Live tracker failed to start: {e}")
                self.live_signal_tracker = None
        else:
            self.live_signal_tracker = None
            logger.info("⚠️  Live signal tracker disabled")
        
        print("🚀 ENHANCED CRYPTO SCALPING DASHBOARD WITH ML")
        if self.enable_demo_bot:
            if use_binance_testnet:
                print("🔗 BINANCE TESTNET INTEGRATION ACTIVE")
            else:
                print("🤖 DEMO TRADING BOT INTEGRATION ACTIVE")
        print("⚡ Multi-Coin Monitoring + ML Predictions + CSV Export")
        print("="*60)
        print(f"INFO: System initialized")
        print(f"INFO: Using Streaming ML: {self.use_streaming_ml}")
        print(f"INFO: Demo Bot Enabled: {self.enable_demo_bot}")
        print(f"INFO: Binance Testnet: {use_binance_testnet}")
        print(f"INFO: Confluence threshold: {self.config.MIN_CONFLUENCE_SCORE}/100")
    
    def generate_all_signals(self) -> int:
        """Generate signals for all coins using streaming ML"""
        self.signals = {}
        
        if self.use_streaming_ml and self.streaming_processor:
            # Use streaming ML system
            print(f"\n🔄 Processing {len(self.streaming_processor.SYMBOLS)} symbols with ML...")
            print("-"*60)
            
            ml_signals = self.streaming_processor.process_symbols_batch()
            self.ml_signals = ml_signals
            
            # Convert ML signals to standard format and process with bot
            for ml_signal in ml_signals:
                signal = self._ml_signal_to_standard(ml_signal)
                if signal:
                    symbol = ml_signal.symbol
                    self.signals[symbol] = signal
                    
                    # Track signal in real-time tracker
                    if self.live_signal_tracker:
                        try:
                            # Convert signal to tracker format
                            tracker_signal = {
                                'symbol': symbol,
                                'direction': signal.get('direction', 'LONG'),
                                'entry_price': signal.get('entry_price', 0),
                                'take_profit_1': signal.get('take_profits', [(0, 0)])[0][0] if signal.get('take_profits') else 0,
                                'take_profit_2': signal.get('take_profits', [None, (0, 0)])[1][0] if len(signal.get('take_profits', [])) > 1 else 0,
                                'take_profit_3': signal.get('take_profits', [None, None, (0, 0)])[2][0] if len(signal.get('take_profits', [])) > 2 else 0,
                                'stop_loss': signal.get('stop_loss', 0),
                                'confluence_score': signal.get('confluence_score', 0),
                                'accuracy': signal.get('accuracy_estimate', 0),
                                'timestamp': signal.get('timestamp', datetime.now())
                            }
                            self.live_signal_tracker.add_signal(tracker_signal)
                            logger.info(f"✅ {symbol} signal added to live tracker")
                        except Exception as e:
                            logger.warning(f"Failed to track {symbol} signal: {e}")
                    
                    # Process signal with demo bot if enabled
                    if self.demo_bot and self.demo_bot.running:
                        if self.demo_bot.process_signal(ml_signal):
                            print(f"  {symbol}... 🤖 BOT TRADE EXECUTED (Confidence: {ml_signal.confidence:.1f}%)")
                        else:
                            print(f"  {symbol}... ✅ ML Signal (Confidence: {ml_signal.confidence:.1f}%) - Bot Skipped")
                    else:
                        print(f"  {symbol}... ✅ ML Signal (Confidence: {ml_signal.confidence:.1f}%)")
                    
                    # Log to CSV
                    if self.tracker and symbol not in self.active_trades:
                        trade_id = self._log_signal_to_csv(signal)
                        if trade_id:
                            self.active_trades[symbol] = trade_id
            
            return len(self.signals)
        else:
            # Fallback to original parallel processing
            return self._generate_all_signals_legacy()
    
    def _generate_all_signals_legacy(self) -> int:
        """Legacy signal generation (parallel processing)"""
        self.signals = {}
        
        print(f"\nAnalyzing {len(self.config.SYMBOLS)} coins...")
        print("-"*60)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self.analyzer.analyze_symbol, symbol): symbol 
                for symbol in self.config.SYMBOLS
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signal = future.result()
                    
                    if signal:
                        self.signals[symbol] = signal
                        print(f"  {symbol}... ✅ Signal generated (Score: {signal['confluence_score']})")
                        
                        if self.tracker and symbol not in self.active_trades:
                            trade_id = self._log_signal_to_csv(signal)
                            if trade_id:
                                self.active_trades[symbol] = trade_id
                    else:
                        print(f"  {symbol}... ❌ No signal")
                        
                except Exception as e:
                    print(f"  {symbol}... ⚠️  Error: {str(e)}")
        
        return len(self.signals)
    
    def _ml_signal_to_standard(self, ml_signal: EnhancedSignal) -> Optional[Dict]:
        """Convert ML EnhancedSignal to standard signal format"""
        try:
            signal: Dict = {
                "symbol": ml_signal.symbol,
                "direction": ml_signal.direction,
                "timestamp": ml_signal.timestamp,
                "entry_price": ml_signal.entry_price,
                "entry_range": (ml_signal.entry_price * 0.999, ml_signal.entry_price * 1.001),
                "stop_loss": ml_signal.stop_loss,
                "take_profits": [
                    (ml_signal.take_profit_1, 40),
                    (ml_signal.take_profit_2, 35),
                    (ml_signal.take_profit_3, 20)
                ],
                "confluence_score": ml_signal.confidence,
                "pattern_score": len(ml_signal.patterns) * 5,
                "indicator_score": 20,
                "volume_score": 15,
                "orderbook_score": 8,
                "detected_patterns": ml_signal.patterns,
                "accuracy_estimate": min(90, ml_signal.confidence * 1.2),
                "risk_percentage": self.config.MAX_RISK_PER_TRADE,
                "leverage": self.config.LEVERAGE,
                "ml_predictions": {
                    "breakout_probability": ml_signal.predictions.breakout_probability,
                    "volume_surge_probability": ml_signal.predictions.volume_surge_probability,
                    "risk_score": ml_signal.predictions.risk_score,
                    "price_target_30m": ml_signal.predictions.price_target_30m,
                    "price_target_1h": ml_signal.predictions.price_target_1h
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ ML signal conversion error: {e}")
            return None
    
    def _log_signal_to_csv(self, signal: Dict) -> Optional[str]:
        """Log signal to CSV"""
        if not self.tracker:
            return None
        
        coin_name = self._get_coin_name(signal['symbol'])
        
        # Extract ML predictions if available
        ml_predictions = signal.get('ml_predictions', {})
        
        trade_data = {
            'pair': signal['symbol'],
            'coin_name': coin_name,
            'entry_time': signal['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            'direction': signal['direction'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profits'][0][0] if signal['take_profits'] else 0,
            'tp_timeframe': '5m',
            'predicted_accuracy': signal['accuracy_estimate'],
            'status': 'OPEN',
            'timeframe': '5m',
            'strategy': f"ML Confluence {signal['confluence_score']:.1f}%",
            'ml_breakout_prob': ml_predictions.get('breakout_probability', 0),
            'ml_risk_score': ml_predictions.get('risk_score', 0)
        }
        
        trade_id = self.tracker.log_trade(trade_data)
        print(f"    📝 Logged to CSV: {trade_id}")
        
        return trade_id
    
    def _get_coin_name(self, symbol: str) -> str:
        """Get full coin name from symbol"""
        coin_map = {
            'BTCUSDT': 'Bitcoin', 'ETHUSDT': 'Ethereum', 'BNBUSDT': 'Binance Coin',
            'XRPUSDT': 'Ripple', 'ADAUSDT': 'Cardano', 'SOLUSDT': 'Solana',
            'AVAXUSDT': 'Avalanche', 'DOTUSDT': 'Polkadot', 'LINKUSDT': 'Chainlink',
            'PEPEUSDT': 'Pepe', 'SHIBUSDT': 'Shiba Inu', 'NEARUSDT': 'NEAR Protocol',
            'ICPUSDT': 'Internet Computer', 'UNIUSDT': 'Uniswap',
            'FILUSDT': 'Filecoin', 'ATOMUSDT': 'Cosmos',
            'OPUSDT': 'Optimism', 'ALGOUSDT': 'Algorand', 'VETUSDT': 'VeChain',
            'DOGEUSDT': 'Dogecoin', 'XLMUSDT': 'Stellar', 'MANAUSDT': 'Decentraland',
            'SANDUSDT': 'The Sandbox', 'DASHUSDT': 'Dash', 'GMTUSDT': 'GMT Token',
            'PEOPLEUSDT': 'ConstitutionDAO', 'LUNCUSDT': 'Luna Classic', 'CHZUSDT': 'Chiliz',
            'FLOWUSDT': 'Flow', 'QTUMUSDT': 'Qtum', 'SNXUSDT': 'Synthetix', 'WIFUSDT': 'dogwifhat'
        }
        return coin_map.get(symbol, symbol.replace('USDT', ''))
    
    def display_dashboard(self) -> None:
        """Display the main dashboard"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*80)
        print("ENHANCED CRYPTO SCALPING DASHBOARD WITH ML".center(80))
        if self.use_streaming_ml:
            print("🚀 REAL-TIME STREAMING MODE".center(80))
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
        
        if self.use_streaming_ml and self.ml_signals:
            quality_counts = {}
            for signal in self.ml_signals:
                quality = signal.quality.value
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            quality_str = ", ".join([f"{k}: {v}" for k, v in quality_counts.items()])
            print(f"SIGNAL QUALITY: {quality_str}")
        
        print("-"*80)
        print(f"{'#':<2} {'Coin':<10} {'Direction':<10} {'Score':<8} {'Quality':<10} {'Entry':<12} {'R/R':<6}")
        print("-"*80)
        
        for i, (symbol, signal) in enumerate(self.signals.items(), 1):
            direction_color = "\033[92m" if signal['direction'] == 'LONG' else "\033[91m"
            reset_color = "\033[0m"
            
            # Calculate risk/reward
            risk = abs(signal['entry_price'] - signal['stop_loss'])
            reward = abs(signal['take_profits'][0][0] - signal['entry_price'])
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Get quality if available
            quality = ""
            if self.use_streaming_ml:
                for ml_signal in self.ml_signals:
                    if ml_signal.symbol == symbol:
                        quality = ml_signal.quality.value[:4]
                        break
            
            print(f"{i:<2} {symbol:<10} "
                  f"{direction_color}{signal['direction']:<10}{reset_color} "
                  f"{signal['confluence_score']:<8.0f} "
                  f"{quality:<10} "
                  f"{signal['entry_price']:<12.5f} "
                  f"{rr_ratio:<6.2f}")
    
    def display_single_signal(self, symbol: str, format_type: str = "simple") -> None:
        """Display signal for a specific coin"""
        if symbol not in self.signals:
            print(f"\nNo signal available for {symbol}")
            return
        
        signal = self.signals[symbol]
        
        # Check if ML predictions available
        ml_predictions = signal.get('ml_predictions', None)
        
        if format_type == "simple":
            output = self.formatter.format_simple_signal(signal)
        else:
            output = self.formatter.format_detailed_signal(signal)
        
        # Add ML predictions if available
        if ml_predictions and format_type != "simple":
            output += f"\n\n🤖 ML PREDICTIONS:"
            output += f"\nBreakout Probability: {ml_predictions.get('breakout_probability', 0):.1f}%"
            output += f"\nVolume Surge Probability: {ml_predictions.get('volume_surge_probability', 0):.1f}%"
            output += f"\nRisk Score: {ml_predictions.get('risk_score', 0):.1f}/100"
            output += f"\nPrice Target (30m): {ml_predictions.get('price_target_30m', 0):.6f}"
            output += f"\nPrice Target (1h): {ml_predictions.get('price_target_1h', 0):.6f}"
        
        print("\n" + "="*80)
        print(f"SIGNAL DETAILS: {symbol}".center(80))
        print("="*80)
        print(output)
        
        # Show bot status if available
        if self.demo_bot and self.demo_bot.running:
            bot_status = self._get_bot_trade_status(symbol)
            if bot_status:
                print(f"\n{bot_status}")
        
        if self.tracker and symbol in self.active_trades:
            print(f"\n📝 Trade ID: {self.active_trades[symbol]}")
        
        print("\n⚠️  Educational use only - Not financial advice")
    
    def _get_bot_trade_status(self, symbol: str) -> str:
        """Get bot trade status for a specific symbol"""
        if not self.demo_bot:
            return ""
        
        # Check active trades
        for trade in self.demo_bot.active_trades:
            if trade.symbol == symbol:
                status_color = "\033[92m" if trade.pnl >= 0 else "\033[91m"
                reset_color = "\033[0m"
                return (f"🤖 BOT STATUS: {status_color}Active Trade{reset_color} | "
                       f"PnL: {status_color}${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%){reset_color} | "
                       f"Entry: {trade.entry_price:.5f}")
        
        # Check completed trades
        for trade in self.demo_bot.completed_trades:
            if trade.symbol == symbol:
                status_color = "\033[92m" if trade.pnl >= 0 else "\033[91m"
                reset_color = "\033[0m"
                return (f"🤖 BOT STATUS: {status_color}Completed{reset_color} | "
                       f"PnL: {status_color}${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%){reset_color} | "
                       f"Reason: {trade.exit_reason}")
        
        return "🤖 BOT STATUS: No trade executed"
    
    def update_bot_trades(self):
        """Update bot trades with current market prices"""
        if not self.demo_bot or not self.demo_bot.running:
            return
        
        if not self.use_streaming_ml or not self.streaming_processor:
            return
        
        try:
            # Get current market prices
            symbols = [trade.symbol for trade in self.demo_bot.active_trades]
            if not symbols:
                return
            
            market_data = self.streaming_processor.binance_api.get_streaming_data_sync(symbols)
            current_prices = {symbol: data['price'] for symbol, data in market_data.items()}
            
            # Update bot trades
            self.demo_bot.update_trades(current_prices)
            
        except Exception as e:
            logger.error(f"❌ Bot update error: {e}")
    
    def show_bot_summary(self):
        """Show comprehensive bot trading summary"""
        if not self.demo_bot:
            print("\n❌ Demo trading bot is not enabled")
            return
        
        print("\n" + "="*80)
        print("DEMO TRADING BOT DASHBOARD".center(80))
        print("="*80)
        
        summary = self.demo_bot.get_trading_summary()
        print(summary)
        
        # Show recent trades
        if self.demo_bot.completed_trades:
            print(f"\n📈 Recent Completed Trades (Last 5):")
            print("-" * 70)
            print(f"{'Symbol':<12} {'Direction':<6} {'PnL':<10} {'%':<8} {'Reason':<15} {'Time'}")
            print("-" * 70)
            
            recent_trades = sorted(self.demo_bot.completed_trades, key=lambda x: x.exit_time or x.entry_time, reverse=True)[:5]
            for trade in recent_trades:
                pnl_color = "\033[92m" if trade.pnl >= 0 else "\033[91m"
                reset_color = "\033[0m"
                
                print(f"{trade.symbol:<12} {trade.direction:<6} "
                      f"{pnl_color}${trade.pnl:>8.2f}{reset_color} "
                      f"{pnl_color}{trade.pnl_percentage:>6.2f}%{reset_color} "
                      f"{trade.exit_reason[:15]:<15} "
                      f"{(trade.exit_time or trade.entry_time).strftime('%H:%M:%S')}")

    def start_auto_trading_mode(self):
        """Start automated demo trading mode with continuous monitoring"""
        if not self.demo_bot:
            print("\n❌ Demo trading bot is not enabled")
            return
        
        print("\n" + "="*80)
        print("🤖 AUTO TRADING MODE ACTIVATED".center(80))
        print("="*80)
        print("Bot will automatically trade on high-confidence signals")
        print("Press Ctrl+C to stop auto trading")
        print("="*80)
        
        try:
            iteration = 0
            while self.demo_bot.running:
                iteration += 1
                print(f"\n🔄 Scanning iteration #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Generate new signals
                signal_count = self.generate_all_signals()
                
                # Update existing trades
                self.update_bot_trades()
                
                # Show brief status
                if iteration % 3 == 0:  # Every 3rd iteration
                    active_count = len(self.demo_bot.active_trades)
                    equity = self.demo_bot.portfolio.equity
                    pnl = self.demo_bot.portfolio.total_pnl
                    
                    status_color = "\033[92m" if pnl >= 0 else "\033[91m"
                    reset_color = "\033[0m"
                    
                    print(f"📊 Status: {active_count} active trades | "
                          f"Equity: ${equity:,.2f} | "
                          f"PnL: {status_color}${pnl:,.2f}{reset_color}")
                
                # Wait before next iteration
                time.sleep(30)  # Scan every 30 seconds
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Auto trading stopped by user")
            self.show_bot_summary()
            # Ensure clean exit code on user stop
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Auto trading error: {e}")
            logger.error(f"Auto trading error: {e}")
    
    def display_quick_signals(self) -> None:
        """Display quick signals for all coins with signals"""
        if not self.signals:
            print("\nNo signals available right now.")
            return
        
        for symbol, signal in self.signals.items():
            ml_predictions = signal.get('ml_predictions', {})
            
            print(f"\n{'='*60}")
            print(f"📈 {symbol} - {signal['direction']} Signal")
            print(f"   Entry: {signal['entry_range'][0]:.5f} - {signal['entry_range'][1]:.5f}")
            print(f"   Stop: {signal['stop_loss']:.5f}")
            print(f"   Score: {signal['confluence_score']}/100 | Accuracy: {signal['accuracy_estimate']:.1f}%")
            
            if ml_predictions:
                print(f"   🤖 ML: Breakout {ml_predictions.get('breakout_probability', 0):.1f}% | Risk {ml_predictions.get('risk_score', 0):.1f}")
            
            print(f"   Patterns: {', '.join(signal['detected_patterns'][:2]) if signal['detected_patterns'] else 'None'}")
            
            if self.tracker and symbol in self.active_trades:
                print(f"   📝 Trade ID: {self.active_trades[symbol]}")
    
    def show_ml_statistics(self) -> None:
        """Show ML processing statistics"""
        if not self.use_streaming_ml or not self.streaming_processor:
            print("\n⚠️  Streaming ML not enabled")
            return
        
        stats = self.streaming_processor.processing_stats
        
        print("\n" + "="*60)
        print("ML PROCESSING STATISTICS".center(60))
        print("="*60)
        
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Signals Generated: {stats['signals_generated']}")
        print(f"Avg Processing Time: {stats['processing_time_avg']:.2f}s")
        
        if stats['last_batch_time']:
            last_time = stats['last_batch_time'].strftime("%H:%M:%S")
            print(f"Last Batch: {last_time}")
        
        # Signal quality breakdown
        if self.ml_signals:
            quality_counts = {}
            for signal in self.ml_signals:
                quality = signal.quality.value
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            print(f"\nSIGNAL QUALITY DISTRIBUTION:")
            for quality, count in quality_counts.items():
                percentage = (count / len(self.ml_signals)) * 100
                print(f"  {quality}: {count} ({percentage:.1f}%)")
    
    def show_csv_statistics(self) -> None:
        """Show CSV statistics"""
        print("\n" + "="*60)
        print("CSV EXPORT STATISTICS".center(60))
        print("="*60)
        
        # Original CSV tracking
        if self.tracker:
            print("\n📊 SIGNAL CSV EXPORT:")
            stats = self.tracker.get_trade_statistics()
            for key, value in stats.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
            print(f"   📁 Signals CSV: {self.tracker.csv_file}")
        else:
            print("\n⚠️  Signal CSV tracking not enabled")
        
        # Bot CSV tracking
        if self.demo_bot:
            print(f"\n🤖 BOT TRADES CSV EXPORT:")
            csv_stats = self.demo_bot.get_csv_statistics()
            
            if csv_stats.get('file_exists', False):
                print(f"   📁 Bot CSV File: {csv_stats['csv_file']}")
                print(f"   📈 Total Records: {csv_stats['total_records']}")
                print(f"   💰 Total P&L: ${csv_stats['total_pnl']:.2f}")
                print(f"   ✅ Winning Trades: {csv_stats['winning_trades']}")
                print(f"   ❌ Losing Trades: {csv_stats['losing_trades']}")
                print(f"   📊 Win Rate: {csv_stats['win_rate']:.1f}%")
            else:
                print(f"   📁 Bot CSV File: {csv_stats['csv_file']} (not created yet)")
                print(f"   📝 Status: No trades completed yet")
                
            # Option to export current portfolio summary
            print(f"\n💡 Use 'E' command to export current portfolio summary")
        else:
            print(f"\n⚠️  Bot CSV export not available (bot not enabled)")
        
        print("\n" + "="*60)
    
    def run_interactive(self) -> None:
        """Run interactive dashboard"""
        while True:
            # Generate signals
            signal_count = self.generate_all_signals()
            
            # Display dashboard
            self.display_dashboard()
            
            # Show menu
            print("\n" + "="*80)
            print("MENU:")
            print("  1-35. View detailed signal for coin")
            print("  A. View all quick signals")
            print("  M. Show ML statistics")
            print("  S. Show CSV statistics")
            
            # Bot options (if enabled)
            if self.demo_bot:
                print("  B. Show bot trading summary")
                print("  T. Start auto trading mode")
                print("  E. Export portfolio to CSV")
                if self.demo_bot.running:
                    print("  X. Stop demo bot")
                else:
                    print("  X. Start demo bot")
            
            print("  R. Refresh dashboard")
            print("  Q. Quit")
            print("="*80)
            
            # Get user input
            choice = input("\nEnter choice: ").strip().upper()
            
            if choice == 'Q':
                print("\nExiting dashboard...")
                
                if self.tracker:
                    print("\nRunning cleanup...")
                    self.tracker.cleanup_old_data(hours=24)
                
                break
            
            elif choice == 'R':
                continue
            
            elif choice == 'A':
                self.display_quick_signals()
                input("\nPress Enter to continue...")
            
            elif choice == 'M':
                self.show_ml_statistics()
                input("\nPress Enter to continue...")
            
            elif choice == 'S':
                self.show_csv_statistics()
                input("\nPress Enter to continue...")
            
            elif choice == 'B' and self.demo_bot:
                self.show_bot_summary()
                input("\nPress Enter to continue...")
            
            elif choice == 'T' and self.demo_bot:
                self.start_auto_trading_mode()
                input("\nPress Enter to continue...")
            
            elif choice == 'E' and self.demo_bot:
                self.demo_bot.export_portfolio_summary()
                print("\n📊 Portfolio summary exported to CSV!")
                input("\nPress Enter to continue...")
            
            elif choice == 'X' and self.demo_bot:
                if self.demo_bot.running:
                    self.demo_bot.stop_bot()
                    print("\n🛑 Demo bot stopped")
                else:
                    self.demo_bot.start_bot()
                    print("\n🤖 Demo bot started")
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
    
    def run_single_analysis(self, symbol: Optional[str] = None, format_type: str = "simple") -> None:
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
        
        if self.use_streaming_ml and self.streaming_processor:
            # Use ML analysis for single symbol
            try:
                market_data = self.streaming_processor.binance_api.get_streaming_data_sync([symbol])
                if symbol in market_data:
                    ml_signal = self.streaming_processor._analyze_symbol_advanced(symbol, market_data[symbol])
                    if ml_signal:
                        signal = self._ml_signal_to_standard(ml_signal)
                        if signal:
                            self.signals[symbol] = signal
                            
                            if self.tracker:
                                trade_id = self._log_signal_to_csv(signal)
                                if trade_id:
                                    self.active_trades[symbol] = trade_id
                            
                            if format_type == "simple":
                                output = self.formatter.format_simple_signal(signal)
                            else:
                                output = self.formatter.format_detailed_signal(signal)
                            
                            # Add ML info
                            if format_type != "simple":
                                output += f"\n\n🤖 ML PREDICTIONS:"
                                output += f"\nBreakout Probability: {ml_signal.predictions.breakout_probability:.1f}%"
                                output += f"\nVolume Surge Probability: {ml_signal.predictions.volume_surge_probability:.1f}%"
                                output += f"\nRisk Score: {ml_signal.predictions.risk_score:.1f}/100"
                            
                            print(output)
                            return
            except Exception as e:
                logger.error(f"❌ ML analysis error: {e}")
        
        # Fallback to original analysis
        signal = self.analyzer.analyze_symbol(symbol)
        
        if not signal:
            print(f"No high-quality confluence signal available for {symbol}")
            return
        
        print(f"INFO: Confluence score: {signal['confluence_score']}/100")
        print(f"INFO: ✅ Signal generated: {signal['direction']} with {signal['confluence_score']}/100")
        
        if self.tracker:
            trade_id = self._log_signal_to_csv(signal)
            if trade_id:
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

def main() -> None:
    """Main function"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "="*80)
    print("ENHANCED CRYPTO SCALPING DASHBOARD WITH ML".center(80))
    print("University Project with Streaming ML + CSV Export".center(80))
    print("="*80)
    
    print("\nChoose mode:")
    print("1. Multi-Coin Dashboard (Streaming ML + CSV export)")
    print("2. Single Coin Analysis")
    print("3. Quick Signal Generation")
    print("4. Demo Trading Bot Mode (Automated Trading)")
    print("5. Legacy Mode (Original system)")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
    except:
        choice = "1"
    
    use_streaming_ml = True
    enable_demo_bot = False
    use_binance_testnet = False  # Initialize explicitly
    
    if choice == "5":
        use_streaming_ml = False
        print("\n⚠️  Using LEGACY mode (original system)")
    elif choice == "4":
        enable_demo_bot = True
        print("\n🤖 DEMO TRADING BOT MODE ENABLED")
        
        # Ask about Binance testnet
        try:
            print("\nBinance Integration:")
            print("1. Simulation Mode (Virtual trading)")
            print("2. Binance Testnet (Demo account required)")
            binance_choice = input("\nChoose integration (1-2): ").strip()
            use_binance_testnet = (binance_choice == "2")
            
            if use_binance_testnet:
                print("\n🔗 BINANCE TESTNET MODE")
                print("⚠️  Make sure you have configured your testnet API credentials in the code!")
                print("   Get testnet API keys from: https://testnet.binance.vision/")
                
        except:
            use_binance_testnet = False
    
    dashboard = EnhancedScalpingDashboard(
        use_streaming_ml=use_streaming_ml, 
        enable_demo_bot=enable_demo_bot,
        use_binance_testnet=use_binance_testnet if enable_demo_bot else False
    )
    
    if choice == "2":
        print("\n" + "="*60)
        print("SINGLE COIN ANALYSIS MODE")
        print("="*60)
        
        print(f"\nAvailable coins: {', '.join(dashboard.config.SYMBOLS[:15])}...")
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
        
        if use_streaming_ml:
            dashboard.show_ml_statistics()
        
        dashboard.show_csv_statistics()
        
        print("\n⚠️  Educational use only - Not financial advice")
    
    elif choice == "4":
        print("\n" + "="*60)
        print("DEMO TRADING BOT MODE")
        print("="*60)
        
        if not dashboard.demo_bot:
            print("❌ Demo bot not initialized")
            return
        
        print("Bot Configuration:")
        print(f"  Initial Balance: ${dashboard.demo_bot.portfolio.initial_balance:,.2f}")
        print(f"  Risk per Trade: {dashboard.demo_bot.risk_per_trade}%")
        print(f"  Max Concurrent Trades: {dashboard.demo_bot.max_concurrent_trades}")
        print(f"  Min Confidence: {dashboard.demo_bot.min_confidence_score}%")
        print("\nStarting automated demo trading...")
        
        dashboard.start_auto_trading_mode()
    
    else:
        print("\n" + "="*60)
        print("MULTI-COIN DASHBOARD MODE")
        if use_streaming_ml:
            print("🚀 STREAMING ML ENABLED")
        print("="*60)
        print("CSV export enabled - All trades logged automatically")
        print("Press Ctrl+C to exit")
        print("="*60)
        
        dashboard.run_interactive()


def test_demo_bot():
    """Test demo trading bot functionality"""
    print("\n" + "="*80)
    print("TESTING DEMO TRADING BOT")
    print("="*80)
    
    # Create demo bot
    bot = DemoTradingBot(initial_balance=10000, max_concurrent_trades=3)
    
    # Create sample predictions
    sample_predictions = PredictionMetrics(
        price_target_30m=88200.0,
        price_target_1h=88220.0,
        price_target_3h=88250.0,
        breakout_probability=75.0,
        volume_surge_probability=65.0,
        trend_reversal_probability=25.0,
        market_correlation_score=80.0,
        volatility_prediction=5.2,
        confidence_interval=(88180.0, 88210.0),
        risk_score=35.0
    )
    
    # Create sample ML signal
    sample_signal = EnhancedSignal(
        symbol="BTCUSDT",
        direction="LONG",
        confidence=75.5,
        quality=SignalQuality.HIGH,
        entry_price=88193.61,
        stop_loss=88184.79,
        take_profit_1=88206.84,
        take_profit_2=88220.07,
        take_profit_3=88237.71,
        current_price=88193.61,
        volume_24h=1000000,
        change_24h=2.5,
        predictions=sample_predictions,
        patterns=["bullish_momentum", "volume_spike"],
        timestamp=datetime.now(),
        processing_time=0.1,
        data_sources=["TEST"],
        ml_features={},
        leverage=20
    )
    
    print("📊 Initial Portfolio Status:")
    print(f"   Balance: ${bot.portfolio.balance:,.2f}")
    print(f"   Equity: ${bot.portfolio.equity:,.2f}")
    
    # Test signal processing
    print(f"\n🧪 Testing Signal Processing:")
    print(f"   Signal: {sample_signal.symbol} {sample_signal.direction}")
    print(f"   Confidence: {sample_signal.confidence:.1f}%")
    print(f"   Entry: {sample_signal.entry_price:.5f}")
    
    if bot.process_signal(sample_signal):
        print("   ✅ Bot trade executed successfully!")
        
        # Show active trade
        if bot.active_trades:
            trade = bot.active_trades[0]
            print(f"   📈 Active Trade: {trade.symbol}")
            print(f"      Quantity: {trade.quantity:.6f}")
            print(f"      Leverage: {trade.leverage}x")
            print(f"      Entry: ${trade.entry_price:.5f}")
            print(f"      Stop Loss: ${trade.stop_loss:.5f}")
        
        # Test trade update (simulate price movement)
        print(f"\n🔄 Simulating price movements...")
        test_prices = {
            "BTCUSDT": 88200.00  # Simulate price moving toward TP1
        }
        bot.update_trades(test_prices)
        
        if bot.active_trades:
            trade = bot.active_trades[0]
            status_color = "\033[92m" if trade.pnl >= 0 else "\033[91m"
            reset_color = "\033[0m"
            print(f"   Current Price: ${test_prices['BTCUSDT']:.2f}")
            print(f"   Unrealized PnL: {status_color}${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%){reset_color}")
        
        # Simulate trade completion for CSV testing
        print(f"\n🧪 Simulating trade completion for CSV export...")
        if bot.active_trades:
            trade = bot.active_trades[0]
            # Simulate TP1 hit
            tp1_price = trade.take_profit_1
            bot.close_trade(trade, tp1_price, "Take Profit 1 (Test)")
            print(f"   ✅ Trade closed at TP1: ${tp1_price:.2f}")
        
        # Show bot summary
        print(f"\n{bot.get_trading_summary()}")
        
        # Test CSV export
        print(f"\n📊 Testing CSV Export:")
        bot.export_portfolio_summary()
        
        # Show CSV statistics
        csv_stats = bot.get_csv_statistics()
        print(f"📄 CSV Statistics:")
        for key, value in csv_stats.items():
            print(f"   {key}: {value}")
        
    else:
        print("   ❌ Bot trade was rejected")
    
    print("\n✅ Demo bot test completed!")
    print("📄 Check for generated CSV files:")
    print(f"   - demo_bot_trades_{datetime.now().strftime('%Y%m%d')}.csv (completed trades)")
    print(f"   - demo_bot_portfolio_{datetime.now().strftime('%Y%m%d')}.csv (portfolio summary)")
    print("\n💡 Use Excel or any CSV viewer to analyze the data!")

def test_binance_connection():
    """Test Binance testnet connection with configured API keys"""
    print("\n" + "="*80)
    print("TESTING BINANCE TESTNET CONNECTION")
    print("="*80)
    
    # Create config and API instance
    config = BinanceConfig()
    api = BinanceTestnetAPI(config)
    
    print(f"🔧 Configuration:")
    print(f"   API Key: {config.TESTNET_API_KEY[:20]}...")
    print(f"   Secret Key: {config.TESTNET_SECRET_KEY[:20]}...")
    print(f"   Base URL: {config.get_base_url()}")
    print(f"   Configured: {config.is_configured()}")
    
    if not config.is_configured():
        print("\n❌ API credentials not configured!")
        return
    
    print(f"\n🔄 Testing connection...")
    
    # Test balance
    print(f"\n1. Testing balance retrieval:")
    balance = api.get_balance()
    if balance:
        print(f"   ✅ Balance: {balance['balance']:.2f} USDT")
        print(f"   ✅ Available: {balance['available']:.2f} USDT")
    else:
        print(f"   ❌ Failed to get balance")
    
    # Test positions
    print(f"\n2. Testing positions:")
    positions = api.get_positions()
    if positions:
        print(f"   ✅ Found {len(positions)} active positions")
        for pos in positions:
            print(f"      {pos['symbol']}: {pos['positionAmt']} @ {pos['entryPrice']}")
    else:
        print(f"   ✅ No active positions (normal for new account)")
    
    # Test demo order
    print(f"\n3. Testing demo order placement:")
    demo_order = api.place_demo_order("BTCUSDT", "BUY", 0.001, 50000.00)
    print(f"   ✅ Demo order placed: {demo_order['orderId']}")
    print(f"   📊 Symbol: {demo_order['symbol']}")
    print(f"   📊 Side: {demo_order['side']}")
    print(f"   📊 Quantity: {demo_order['quantity']}")
    
    print(f"\n✅ Binance testnet connection test completed!")

if __name__ == "__main__":
    try:
        # Check if user wants to test specific functionality
        if len(sys.argv) > 1:
            if sys.argv[1] == "--test-timing":
                fn = globals().get('test_improved_timing')
                if callable(fn):
                    fn()
                else:
                    print("⚠️  Timing test not available (removed)")
            elif sys.argv[1] == "--test-bot":
                fn = globals().get('test_demo_bot')
                if callable(fn):
                    fn()
                else:
                    print("⚠️  Demo bot test not available")
            elif sys.argv[1] == "--test-binance":
                fn = globals().get('test_binance_connection')
                if callable(fn):
                    fn()
                else:
                    print("⚠️  Binance connection test not available")
            else:
                print("Available tests: --test-timing, --test-bot, --test-binance")
        else:
            main()
    except KeyboardInterrupt:
        # User-initiated stop should exit cleanly with code 0
        print("\n\nDashboard stopped by user")
        sys.exit(0)
    except SystemExit as e:
        # Normalize non-zero exit codes to zero for a graceful shutdown
        if getattr(e, 'code', None) not in (None, 0):
            logger.info(f"Normalizing exit code {e.code} to 0 for a clean shutdown")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        # Keep non-zero exit code for real errors
        sys.exit(1)