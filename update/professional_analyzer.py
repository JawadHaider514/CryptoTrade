"""
PROFESSIONAL TRADING ANALYZER
5+ Years Experience Logic with 95% Accuracy Target
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

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


class ProfessionalAnalyzer:
    """Complete professional trading analysis system"""
    
    def __init__(self, config):
        self.config = config
        self.setup_history = []
        self.performance_stats = {
            'total_setups': 0,
            'high_confidence_setups': 0,
            'winning_setups': 0,
            'losing_setups': 0,
            'avg_accuracy': 0.0,
            'best_setup_score': 0.0,
            'market_regimes': {}
        }
        # 8-Layer Scoring Configuration
        self.layer_weights = {
            "indicators": 25,
            "patterns": 15,
            "volume": 15,
            "orderbook": 10,
            "mtf": 10,
            "price_action": 10,
            "risk_management": 10,
            "execution": 5
        }
        # Tighter directional thresholds
        self.rsi_bull_threshold = 40
        self.rsi_bear_threshold = 60
        # Risk and SL safety parameters
        self.min_sl_pct = 0.0025
        self.sl_multiplier = 3.5
        self.min_atr_pct = 0.0015
    
    def analyze_complete_setup(self, symbol: str, data_frames: Dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        """
        Complete professional analysis following 5+ year trader framework
        Returns TradeSetup if all conditions met for 95% accuracy target
        """
        try:
            # ========== PHASE 1: MULTI-TIMEFRAME ANALYSIS ==========
            print(f"\n🔍 PROFESSIONAL ANALYSIS: {symbol}")
            print("="*60)
            
            # 1.1 Higher timeframe trend analysis
            trend_analysis = self._analyze_higher_timeframe_trend(data_frames)
            if not trend_analysis['trend_clear']:
                print("  ❌ No clear trend on higher timeframes")
                return None
            
            # 1.2 Market regime detection
            market_regime = self._detect_market_regime(data_frames['15m'])
            print(f"  📊 Market Regime: {market_regime}")
            
            # ========== PHASE 2: PRICE ACTION ANALYSIS ==========
            price_action = self._analyze_price_action(data_frames['15m'])
            
            # 2.1 Support and Resistance
            sr_levels = self._calculate_support_resistance(data_frames['15m'])
            
            # 2.2 Breakout/Retest detection
            breakout_analysis = self._analyze_breakout_retest(
                data_frames['15m'], sr_levels, trend_analysis['direction']
            )
            
            # ========== PHASE 3: INDICATOR CONFIRMATION ==========
            indicator_signals = self._analyze_indicators(data_frames['15m'])
            
            # 3.1 RSI analysis
            rsi_signal = self._analyze_rsi(data_frames['15m'])
            
            # 3.2 MACD analysis
            macd_signal = self._analyze_macd(data_frames['15m'])
            
            # 3.3 Bollinger Bands analysis
            bb_signal = self._analyze_bollinger_bands(data_frames['15m'])
            
            # 3.4 EMA analysis
            ema_signal = self._analyze_ema_alignment(data_frames['15m'])
            
            # ========== PHASE 4: VOLUME & MOMENTUM ==========
            volume_analysis = self._analyze_volume(data_frames['15m'])
            
            # 4.1 Volume spikes
            volume_spike = self._detect_volume_spike(data_frames['15m'])
            
            # 4.2 Volume divergence
            volume_divergence = self._detect_volume_divergence(data_frames['15m'])
            
            # ========== PHASE 5: ORDER FLOW ANALYSIS ==========
            # Note: This requires real-time order book data
            order_flow = self._analyze_order_flow(symbol)
            
            # ========== PHASE 6: CONFLUENCE SCORING ==========
            confluence = self._calculate_professional_confluence(
                trend_analysis=trend_analysis,
                price_action=price_action,
                breakout_analysis=breakout_analysis,
                indicator_signals=indicator_signals,
                volume_analysis=volume_analysis,
                order_flow=order_flow,
                market_regime=market_regime
            )
            
            print(f"  Confluence Score: {confluence['total_score']:.1f}/100")
            print(f"  📈 Required: {self.config.MIN_CONFLUENCE_SCORE}/100")
            
            # ========== PHASE 7: ACCURACY VALIDATION ==========
            if confluence['total_score'] < self.config.MIN_CONFLUENCE_SCORE:
                print("  ❌ Confluence score below minimum threshold")
                return None
            
            # Check minimum category scores
            if not self._check_minimum_category_scores(confluence):
                print("  ❌ Minimum category scores not met")
                return None
            
            # ========== PHASE 8: RISK ASSESSMENT ==========
            direction = self._determine_trade_direction(confluence)
            
            if direction == "NEUTRAL":
                print("  ❌ No clear trade direction")
                return None
            
            # Calculate trade levels with professional precision
            trade_levels = self._calculate_professional_levels(
                data_frames['15m'], direction, confluence
            )
            
            # ========== PHASE 9: POSITION SIZING ==========
            position_size = self._calculate_professional_position_size(
                trade_levels['entry_price'],
                trade_levels['stop_loss'],
                confluence['confidence'],
                direction
            )
            
            # ========== PHASE 10: ACCURACY ESTIMATION ==========
            accuracy_estimate = self._estimate_professional_accuracy(confluence)
            
            if accuracy_estimate < getattr(self.config, 'MIN_ACCEPTABLE_ACCURACY', 0.6):  # Use configured minimum
                print(f"  ❌ Accuracy estimate too low: {accuracy_estimate:.1%} < {getattr(self.config, 'MIN_ACCEPTABLE_ACCURACY', 0.6):.1%}")
                return None
            
            # ========== CREATE PROFESSIONAL TRADE SETUP ==========
            setup = TradeSetup(
                symbol=symbol,
                direction=direction,
                entry_price=trade_levels['entry_price'],
                entry_range=trade_levels['entry_range'],
                stop_loss=trade_levels['stop_loss'],
                take_profits=trade_levels['take_profits'],
                confidence=confluence['confidence'] * 100,
                confluence_score=confluence['total_score'],
                accuracy_estimate=accuracy_estimate * 100,  # Convert to percentage
                timestamp=datetime.now(),
                timeframes=self._get_timeframe_summary(data_frames),
                patterns=price_action['patterns'],
                indicators=indicator_signals,
                risk_reward_ratio=trade_levels['risk_reward_ratio'],
                position_size=position_size,
                leverage=self.config.LEVERAGE,
                risk_percentage=self.config.MAX_RISK_PER_TRADE,
                setup_type=breakout_analysis['setup_type'],
                market_regime=market_regime,
                trade_priority=self._calculate_trade_priority(confluence)
            )
            
            # Store for performance tracking
            self.setup_history.append(setup)
            self.performance_stats['total_setups'] += 1
            if confluence['confidence'] > 0.8:
                self.performance_stats['high_confidence_setups'] += 1
            
            print(f"  PROFESSIONAL SETUP GENERATED!")
            print(f"  🎯 Direction: {direction}")
            print(f"  📊 Accuracy Estimate: {accuracy_estimate:.1%}")
            print(f"  ⚖️ Risk/Reward: {trade_levels['risk_reward_ratio']:.2f}")
            print(f"  🔢 Priority: {setup.trade_priority}/5")
            
            return setup
            
        except Exception as e:
            print(f"  ❌ Error in professional analysis: {e}")
            return None
    
    # ========== ANALYSIS METHODS ==========
    
    def _analyze_higher_timeframe_trend(self, data_frames: Dict) -> Dict:
        """Analyze trend on higher timeframes. This wraps the new multi-timeframe
        structure analysis to keep backward compatibility with callers that
        expect a 'trend_clear' and 'direction' key."""
        # Use the multi-timeframe analyzer for a complete view
        mtf = self._analyze_multi_timeframe_structure(data_frames)
        # Map multi-TF results to the old format
        if mtf.get('trend_clear', False):
            return {
                'trend_clear': True,
                'direction': mtf.get('direction', 'NEUTRAL'),
                'alignment': 'STRONG' if mtf.get('all_aligned', False) or mtf.get('weighted_alignment', 0) >= 0.7 else 'WEAK',
                'timeframe_directions': mtf.get('timeframe_directions', {})
            }
        else:
            return {
                'trend_clear': False,
                'direction': 'NEUTRAL',
                'alignment': 'WEAK',
                'timeframe_directions': mtf.get('timeframe_directions', {})
            }
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime using range + ATR thresholds for robustness."""
        if len(df) < 50:
            return "UNKNOWN"

        # Basic measures
        current_price = df['close'].iloc[-1]
        sma_50 = df['close'].rolling(50).mean()
        trend_strength = abs((current_price - sma_50.iloc[-1]) / sma_50.iloc[-1])

        # Range over the provided window
        range_pct = (df['high'].max() - df['low'].min()) / df['close'].mean()

        # ATR percentage
        atr = self._calculate_atr(df, period=getattr(self.config, 'ATR_PERIOD', 14))
        atr_pct = atr / current_price if current_price and atr else 0.0

        # Config thresholds
        range_threshold = getattr(self.config, 'REGIME_RANGE_THRESHOLD', 0.01)
        atr_threshold = getattr(self.config, 'REGIME_ATR_PERCENTAGE', 0.015)

        # Debug print for visibility
        try:
            print(f"  Analyzer regime check -> range_pct={range_pct:.4f}, atr_pct={atr_pct:.4f}, trend_strength={trend_strength:.4f}")
        except Exception:
            pass

        # Consolidation when both range and ATR are low
        if range_pct < range_threshold and atr_pct < atr_threshold:
            return "CONSOLIDATION"

        # High volatility detection
        if atr_pct > atr_threshold * 3:
            return "HIGH_VOLATILITY"

        # Trending vs ranging determination (using trend strength)
        if trend_strength > 0.05:
            return "TRENDING"
        else:
            return "RANGING"
    
    def _analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """Analyze price action patterns"""
        patterns = []
        confidence_scores = []
        
        # Detect patterns
        patterns_detected = self._detect_all_patterns(df)
        
        for pattern in patterns_detected:
            patterns.append(pattern['name'])
            confidence_scores.append(pattern['confidence'])
        
        # Calculate support/resistance quality
        sr_quality = self._calculate_sr_quality(df)
        
        return {
            'patterns': patterns,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'sr_quality': sr_quality,
            'pattern_count': len(patterns)
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """Calculate professional support and resistance levels"""
        if len(df) < lookback:
            return {'support': [], 'resistance': []}
        
        # Use recent data
        recent = df.tail(lookback)
        
        # Find swing highs and lows
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Detect peaks and troughs
        peaks = self._find_peaks(highs, window=5)
        troughs = self._find_troughs(lows, window=5)
        
        # Convert to price levels
        resistance_levels = [highs[i] for i in peaks]
        support_levels = [lows[i] for i in troughs]
        
        # Filter significant levels (clustering)
        resistance = self._cluster_levels(resistance_levels, tolerance=0.005)
        support = self._cluster_levels(support_levels, tolerance=0.005)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Find nearest levels
        nearest_resistance = min(resistance, key=lambda x: abs(x - current_price)) if resistance else current_price
        nearest_support = min(support, key=lambda x: abs(x - current_price)) if support else current_price
        
        return {
            'resistance': resistance[:5],  # Top 5 resistance levels
            'support': support[:5],        # Top 5 support levels
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'distance_to_resistance': abs(current_price - nearest_resistance) / current_price if resistance else 0,
            'distance_to_support': abs(current_price - nearest_support) / current_price if support else 0
        }
    
    def _analyze_breakout_retest(self, df: pd.DataFrame, sr_levels: Dict, trend: str) -> Dict:
        """Analyze breakout and retest patterns"""
        current_price = df['close'].iloc[-1]
        
        # Check for breakout
        breakout = self._check_breakout(df, sr_levels)
        
        # Check for retest
        retest = self._check_retest(df, sr_levels, breakout)
        
        # Determine setup type
        setup_type = self._determine_setup_type(breakout, retest, trend)
        
        return {
            'breakout_detected': breakout['detected'],
            'retest_detected': retest['detected'],
            'setup_type': setup_type,
            'breakout_strength': breakout['strength'],
            'retest_quality': retest['quality']
        }
    
    def _analyze_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyze all technical indicators"""
        indicators = {}
        
        # RSI
        rsi = self._calculate_rsi(df)
        indicators['rsi'] = rsi
        indicators['rsi_signal'] = 'BULLISH' if rsi < self.rsi_bull_threshold else 'BEARISH' if rsi > self.rsi_bear_threshold else 'NEUTRAL'
        
        # MACD
        macd = self._calculate_macd(df)
        indicators['macd'] = macd
        indicators['macd_signal'] = 'BULLISH' if macd['histogram'] > 0 else 'BEARISH'
        
        # Bollinger Bands
        bb = self._calculate_bollinger_bands(df)
        indicators['bb'] = bb
        
        # EMAs
        emas = self._calculate_emas(df)
        indicators['emas'] = emas
        
        # ATR for volatility
        atr = self._calculate_atr(df)
        indicators['atr'] = atr
        
        # Stochastic
        stoch = self._calculate_stochastic(df)
        indicators['stochastic'] = stoch
        
        return indicators
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Dict:
        """Analyze RSI indicator"""
        rsi = self._calculate_rsi(df)
        return {
            'value': rsi,
            'signal': 'BULLISH' if rsi < self.rsi_bull_threshold else 'BEARISH' if rsi > self.rsi_bear_threshold else 'NEUTRAL',
            'overbought': rsi > self.config.RSI_OVERBOUGHT,
            'oversold': rsi < self.config.RSI_OVERSOLD
        }
    
    def _analyze_macd(self, df: pd.DataFrame) -> Dict:
        """Analyze MACD indicator"""
        macd = self._calculate_macd(df)
        return {
            'value': macd['macd'],
            'signal': 'BULLISH' if macd['histogram'] > 0 else 'BEARISH',
            'histogram': macd['histogram']
        }
    
    def _analyze_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        """Analyze Bollinger Bands"""
        bb = self._calculate_bollinger_bands(df)
        return {
            'upper': bb['upper'],
            'middle': bb['middle'],
            'lower': bb['lower'],
            'width': bb['width']
        }
    
    def _analyze_ema_alignment(self, df: pd.DataFrame) -> Dict:
        """Analyze EMA alignment and trend structure"""
        emas = self._calculate_emas(df)
        
        # Get EMA values with None checks
        ema_8 = emas.get('ema_8')
        ema_20 = emas.get('ema_20')
        ema_50 = emas.get('ema_50')
        
        signal = 'NEUTRAL'
        aligned = False
        
        # Only check alignment if all EMAs are not None
        if ema_8 is not None and ema_20 is not None and ema_50 is not None:
            if ema_8 > ema_20 > ema_50:
                signal = 'BULLISH'
                aligned = True
            elif ema_8 < ema_20 < ema_50:
                signal = 'BEARISH'
                aligned = True
        
        return {
            'signal': signal,
            'ema_8': ema_8,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'aligned': aligned
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Complete volume analysis"""
        if len(df) < 20:
            return {'volume_ratio': 1.0, 'volume_trend': 'NEUTRAL', 'divergence': False}
        
        volume = df['volume']
        close = df['close']
        
        # Volume ratio (current vs average)
        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = float(volume.iloc[-1] / volume_ma.iloc[-1]) if volume_ma.iloc[-1] > 0 else 1.0
        
        # Volume trend
        volume_trend = 'BULLISH' if volume.iloc[-1] > volume_ma.iloc[-1] else 'BEARISH'
        
        # Volume price trend
        price_change = close.pct_change()
        volume_trend_value = (volume * price_change).cumsum().iloc[-1]
        
        return {
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'volume_trend_value': volume_trend_value,
            'volume_ma': float(volume_ma.iloc[-1])
        }
    
    def _analyze_order_flow(self, symbol: str) -> Dict:
        """Analyze order book flow (simulated for now)"""
        # In real implementation, this would fetch from Binance API
        return {
            'bid_ask_ratio': 1.2,  # Simulated
            'order_imbalance': 0.3,
            'market_depth': 'GOOD',
            'large_orders': True
        }
    
    def _calculate_professional_confluence(self, **analysis_results) -> Dict:
        """Calculate professional confluence score following 5+ year trader logic"""
        scores = {
            'trend': 0.0,
            'price_action': 0.0,
            'indicators': 0.0,
            'volume': 0.0,
            'order_book': 0.0,
            'multi_timeframe': 0.0,
            'total_score': 0.0,
            'confidence': 0.0
        }
        
        # 1. TREND SCORE (20%)
        trend = analysis_results.get('trend_analysis', {})
        if trend.get('trend_clear', False):
            if trend['alignment'] == 'STRONG':
                scores['trend'] = 20
            else:
                scores['trend'] = 10
        
        # 2. PRICE ACTION SCORE (25%)
        price_action = analysis_results.get('price_action', {})
        breakout = analysis_results.get('breakout_analysis', {})
        
        # Pattern strength
        pattern_score = min(10, len(price_action.get('patterns', [])) * 2)
        
        # Setup type score
        setup_type = breakout.get('setup_type', '')
        if setup_type in ['BREAKOUT_RETEST', 'TREND_CONTINUATION']:
            setup_score = 10
        elif setup_type in ['REVERSAL', 'BREAKOUT']:
            setup_score = 7
        else:
            setup_score = 3
        
        # SR quality
        sr_quality = price_action.get('sr_quality', 0)
        sr_score = min(5, sr_quality * 5)
        
        scores['price_action'] = pattern_score + setup_score + sr_score
        
        # 3. INDICATOR SCORE (20%)
        indicators = analysis_results.get('indicator_signals', {})
        indicator_score = 0
        
        # RSI alignment
        rsi_signal = indicators.get('rsi_signal', 'NEUTRAL')
        trend_direction = trend.get('direction', 'NEUTRAL')
        if rsi_signal == trend_direction:
            indicator_score += 5
        
        # MACD alignment
        macd_signal = indicators.get('macd_signal', 'NEUTRAL')
        if macd_signal == trend_direction:
            indicator_score += 5
        
        # EMA alignment
        emas = indicators.get('emas', {})
        if self._check_ema_alignment(emas, trend_direction):
            indicator_score += 5
        
        # Bollinger Bands position
        bb = indicators.get('bb', {})
        if self._check_bb_position(bb, trend_direction):
            indicator_score += 5
        
        scores['indicators'] = indicator_score
        
        # 4. VOLUME SCORE (15%)
        volume = analysis_results.get('volume_analysis', {})
        volume_score = 0
        
        volume_ratio = volume.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            volume_score += 10
        elif volume_ratio > 1.2:
            volume_score += 7
        elif volume_ratio >= self.config.VOLUME_SURGE_REQUIREMENT:
            volume_score += 3
        
        volume_trend = volume.get('volume_trend', 'NEUTRAL')
        if volume_trend == trend_direction:
            volume_score += 5
        
        scores['volume'] = volume_score
        
        # 5. ORDER BOOK SCORE (10%)
        order_flow = analysis_results.get('order_flow', {})
        order_book_score = 0
        
        bid_ask_ratio = order_flow.get('bid_ask_ratio', 1.0)
        if (trend_direction == 'BULLISH' and bid_ask_ratio > 1.2) or \
           (trend_direction == 'BEARISH' and bid_ask_ratio < 0.8):
            order_book_score += 10
        
        scores['order_book'] = order_book_score
        
        # 6. MULTI-TIMEFRAME SCORE (10%)
        multi_tf_score = 0
        # New multi-timeframe fields
        mtf = trend
        weighted_alignment = mtf.get('weighted_alignment', 0.0)
        all_aligned = mtf.get('all_aligned', False)

        if all_aligned:
            multi_tf_score = 10
        elif weighted_alignment >= 0.7:
            multi_tf_score = 7
        elif weighted_alignment >= 0.4:
            multi_tf_score = 5
        else:
            multi_tf_score = 3

        scores['multi_timeframe'] = multi_tf_score
        
        # Calculate total score using weights
        # Each score is 0-25, weights sum to 1.0
        # Normalize each score to 0-100 first by dividing by max and multiplying by 100
        normalized_scores = {
            'trend': min(20, scores['trend']) / 20 * 100,
            'price_action': min(25, scores['price_action']) / 25 * 100,
            'indicators': min(20, scores['indicators']) / 20 * 100,
            'volume': min(15, scores['volume']) / 15 * 100,
            'order_book': min(10, scores['order_book']) / 10 * 100,
            'multi_timeframe': min(10, scores['multi_timeframe']) / 10 * 100,
        }
        
        total_score = (
            normalized_scores['trend'] * self.config.CONFLUENCE_WEIGHTS['trend'] +
            normalized_scores['price_action'] * self.config.CONFLUENCE_WEIGHTS['price_action'] +
            normalized_scores['indicators'] * self.config.CONFLUENCE_WEIGHTS['indicators'] +
            normalized_scores['volume'] * self.config.CONFLUENCE_WEIGHTS['volume'] +
            normalized_scores['order_book'] * self.config.CONFLUENCE_WEIGHTS['order_book'] +
            normalized_scores['multi_timeframe'] * self.config.CONFLUENCE_WEIGHTS['multi_timeframe']
        )
        
        scores['total_score'] = total_score
        scores['confidence'] = total_score / 100
        
        return scores
    
    def _check_minimum_category_scores(self, confluence: Dict) -> bool:
        """Check if minimum category scores are met"""
        # Normalize raw scores to 0-100 scale
        max_scores = {
            'trend': 20,
            'price_action': 25,
            'indicators': 20,
            'volume': 15,
            'order_book': 10,
            'multi_timeframe': 10
        }
        
        for category, min_score_pct in self.config.MIN_CATEGORY_SCORES.items():
            raw_score = confluence.get(category, 0)
            max_score = max_scores.get(category, 100)
            normalized_score = (raw_score / max_score * 100) if max_score > 0 else 0
            min_threshold = min_score_pct * 100  # Convert percentage to 0-100 scale
            
            if normalized_score < min_threshold:
                print(f"    ⚠️  {category}: {normalized_score:.1f}/100 < {min_threshold:.1f} (required)")
                return False
        return True
    
    def _determine_trade_direction(self, confluence: Dict) -> str:
        """Determine trade direction based on confluence"""
        # This would be more complex in reality
        # For simplicity, we'll use trend direction
        return 'LONG'  # Simplified
    
    def _calculate_professional_levels(self, df: pd.DataFrame, direction: str, confluence: Dict) -> Dict:
        """Calculate professional trade levels with dynamic SL and minimum buffer validation"""
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR for stop loss with dynamic multiplier
        atr = self._calculate_atr(df)
        
        # Ensure ATR doesn't fall below minimum percentage threshold
        min_atr_distance = current_price * self.config.MIN_ATR_PERCENTAGE
        safe_atr = max(atr, min_atr_distance)
        
        # Dynamic SL calculation using new multiplier and minimum percentage
        sl_distance = safe_atr * self.config.SL_ATR_MULTIPLIER
        min_sl_distance = current_price * self.config.MIN_SL_PERCENTAGE
        # SL distance must be at least the minimum
        sl_distance = max(sl_distance, min_sl_distance)
        
        # Entry price (with slippage consideration)
        if direction == 'LONG':
            entry_price = current_price * 1.001  # Slightly above
            stop_loss = current_price - sl_distance
            entry_range = (current_price * 0.999, current_price * 1.002)
        else:
            entry_price = current_price * 0.999  # Slightly below
            stop_loss = current_price + sl_distance
            entry_range = (current_price * 0.998, current_price * 1.001)
        
        # Calculate stop loss percentage for diagnostics
        stop_loss_pct = sl_distance / current_price
        
        # Calculate take profits
        risk_amount = abs(entry_price - stop_loss)
        take_profits = []
        
        # Multiple take profit levels
        tp_levels = [1.0, 2.0, 3.0]  # Risk multiples
        tp_percentages = [50, 30, 20]  # Position percentages
        
        for i, (multiple, percentage) in enumerate(zip(tp_levels, tp_percentages)):
            if direction == 'LONG':
                tp_price = entry_price + (risk_amount * multiple)
            else:
                tp_price = entry_price - (risk_amount * multiple)
            
            take_profits.append((tp_price, percentage, multiple))
        
        # Calculate risk/reward ratio
        avg_tp = sum(tp[0] * (tp[1]/100) for tp in take_profits) / 100
        risk_reward_ratio = abs(avg_tp - entry_price) / risk_amount
        
        print(f"  🔒 Dynamic SL: entry={entry_price:.6f}, sl={stop_loss:.6f}, sl_pct={stop_loss_pct:.4f} (min={self.config.MIN_SL_PERCENTAGE:.4f})")
        
        return {
            'entry_price': entry_price,
            'entry_range': entry_range,
            'stop_loss': stop_loss,
            'stop_loss_pct': stop_loss_pct,
            'take_profits': take_profits,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _calculate_professional_position_size(self, entry: float, stop_loss: float, 
                                            confidence: float, direction: str) -> float:
        """Calculate professional position size using Kelly Criterion or fixed risk"""
        stop_distance = abs(entry - stop_loss)
        
        # Initialize with default values
        risk_amount_kelly = 0.0
        risk_amount_fixed = 0.0
        
        if self.config.POSITION_SIZING == "kelly":
            # Kelly Criterion: f* = (bp - q) / b
            # where b = risk/reward ratio, p = win probability, q = loss probability
            win_prob = confidence
            loss_prob = 1 - win_prob
            
            # Assume 1:3 risk/reward for Kelly
            risk_reward = 3.0
            kelly_fraction = (win_prob * risk_reward - loss_prob) / risk_reward
            
            # Cap at 25% of bankroll for safety
            kelly_fraction = min(kelly_fraction, 0.25)
            
            risk_amount_kelly = self.config.BANKROLL * kelly_fraction
            
        else:  # fixed_risk
            risk_amount_fixed = self.config.BANKROLL * self.config.MAX_RISK_PER_TRADE
        
        risk_amount = risk_amount_kelly if self.config.POSITION_SIZING == "kelly" else risk_amount_fixed
        
        # Position size = risk amount / stop distance
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        # Adjust for confidence
        position_size *= confidence
        
        return position_size
    
    def _estimate_professional_accuracy(self, confluence: Dict, timeframe: Optional[str] = None) -> float:
        """Estimate accuracy based on confluence score and optionally adjust for selected timeframe"""
        score = confluence.get('total_score', 0)

        # Base accuracy mapping
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

        # If no timeframe provided, return base accuracy
        if timeframe is None or not timeframe:
            return base

        # Timeframe adjustment factors
        tf_factors = {
            '1-3': 1.00,
            '3-5': 0.985,
            '5-15': 0.95,
            '15-30': 0.92,
            '1m': 1.00,
            '3m': 0.985,
            '5m': 0.985,
            '15m': 0.95,
            '30m': 0.92,
            '1h': 0.90,
            '4h': 0.88,
            '1d': 0.85,
            '1w': 0.82,
            '1M': 0.80
        }

        # Normalize timeframe string
        tf_key = str(timeframe).lower()
        # Special case for '1M' (monthly)
        if tf_key == '1m' and len(timeframe) == 2 and timeframe[1] == 'M':
            tf_key = '1M'
        
        factor = tf_factors.get(tf_key)
        
        if factor is not None:
            return round(base * factor, 4)
        return base
    
    def _calculate_trade_priority(self, confluence: Dict) -> int:
        """Calculate trade priority (1-5)"""
        score = confluence['total_score']
        
        if score >= 95:
            return 5  # Highest priority
        elif score >= 90:
            return 4
        elif score >= 85:
            return 3
        elif score >= 80:
            return 2
        else:
            return 1
    
    # ========== HELPER METHODS ==========
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine trend direction"""
        if len(df) < 50:
            return 'NEUTRAL'
        
        # Calculate EMAs
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        ema_200 = df['close'].ewm(span=200, adjust=False).mean()
        
        current_price = df['close'].iloc[-1]
        
        if current_price > ema_50.iloc[-1] > ema_200.iloc[-1]:
            return 'BULLISH'
        elif current_price < ema_50.iloc[-1] < ema_200.iloc[-1]:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _determine_structure(self, df: pd.DataFrame) -> str:
        """Determine structure for a timeframe: HH+HL, LH+LL, or NEUTRAL.
        This uses recent swing highs and lows to determine the internal structure."""
        lookback = min(len(df), 200)
        if lookback < 20:
            return 'NEUTRAL'
        recent = df.tail(lookback)
        highs = recent['high'].values
        lows = recent['low'].values
        peaks = self._find_peaks(highs, window=5)
        troughs = self._find_troughs(lows, window=5)

        # We need at least two peaks and two troughs to infer HH/HL or LH/LL
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Convert to absolute prices
            last_peak = highs[peaks[-1]]
            prev_peak = highs[peaks[-2]]
            last_trough = lows[troughs[-1]]
            prev_trough = lows[troughs[-2]]

            if last_peak > prev_peak and last_trough > prev_trough:
                return 'HH_HL'  # Bullish structure
            elif last_peak < prev_peak and last_trough < prev_trough:
                return 'LH_LL'  # Bearish structure
            else:
                return 'NEUTRAL'
        else:
            return 'NEUTRAL'

    def _analyze_multi_timeframe_structure(self, data_frames: Dict) -> Dict:
        """Analyze multiple timeframes, compute direction/structure per TF,
        compute weighted alignment and enforce 'ALL timeframes must align' rule
        if configured."""
        weights = getattr(self.config, 'MULTI_TF_WEIGHTS', {})
        timeframe_directions = {}
        timeframe_structures = {}
        total_weight = sum(weights.values()) if isinstance(weights, dict) else 0.0
        matched_weight = 0.0
        counts = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}

        for tf, w in weights.items():
            if tf in data_frames and len(data_frames[tf]) >= 30:
                df = data_frames[tf]
                direction = self._determine_trend_direction(df)
                timeframe_directions[tf] = direction
                structure = self._determine_structure(df)
                timeframe_structures[tf] = structure
                counts[direction] = counts.get(direction, 0) + 1

        # Pick majority non-neutral direction
        majority = None
        non_neutral_counts = {k: v for k, v in counts.items() if k != 'NEUTRAL'}
        if non_neutral_counts:
            majority = max(non_neutral_counts.items(), key=lambda x: x[1])[0]

        # Calculate weighted alignment
        if total_weight > 0 and majority is not None:
            for tf, direction in timeframe_directions.items():
                if direction == majority:
                    matched_weight += weights.get(tf, 0)
            weighted_alignment = matched_weight / total_weight
        else:
            weighted_alignment = 0.0

        all_aligned = False
        if timeframe_directions:
            all_aligned = all(d == majority for d in timeframe_directions.values())

        # Enforce strict alignment if configured
        if getattr(self.config, 'REQUIRE_ALL_TF_ALIGNMENT', False) and not all_aligned:
            trend_clear = False
            direction = 'NEUTRAL'
        else:
            # Accept trend if weighted alignment >= 0.6
            trend_clear = majority is not None and weighted_alignment >= 0.6
            direction = majority if trend_clear else 'NEUTRAL'

        return {
            'trend_clear': trend_clear,
            'direction': direction,
            'timeframe_directions': timeframe_directions,
            'timeframe_structures': timeframe_structures,
            'weighted_alignment': weighted_alignment,
            'all_aligned': all_aligned
        }
    
    def _detect_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect all candlestick patterns"""
        patterns = []
        
        # This would implement pattern detection logic
        # For now, return empty list
        return patterns
    
    def _calculate_sr_quality(self, df: pd.DataFrame) -> float:
        """Calculate support/resistance quality score"""
        # Simplified implementation
        return 0.8
    
    def _find_peaks(self, data, window: int = 5) -> List[int]:
        """Find peaks in data - accepts both numpy arrays and pandas Series"""
        if hasattr(data, 'values'):
            data = data.values  # Convert pandas Series to numpy array
        peaks = []
        for i in range(window, len(data) - window):
            if data[i] == max(data[i-window:i+window+1]):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data, window: int = 5) -> List[int]:
        """Find troughs in data - accepts both numpy arrays and pandas Series"""
        if hasattr(data, 'values'):
            data = data.values  # Convert pandas Series to numpy array
        troughs = []
        for i in range(window, len(data) - window):
            if data[i] == min(data[i-window:i+window+1]):
                troughs.append(i)
        return troughs
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.005) -> List[float]:
        """Cluster similar price levels"""
        if not levels:
            return []
        
        levels.sort()
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def _check_breakout(self, df: pd.DataFrame, sr_levels: Dict) -> Dict:
        """Check for breakout"""
        return {'detected': False, 'strength': 0}
    
    def _check_retest(self, df: pd.DataFrame, sr_levels: Dict, breakout: Dict) -> Dict:
        """Check for retest"""
        return {'detected': False, 'quality': 0}
    
    def _determine_setup_type(self, breakout: Dict, retest: Dict, trend: str) -> str:
        """Determine setup type"""
        return 'BREAKOUT_RETEST'
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        close = df['close']
        if len(close) < period:
            return 50.0
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD"""
        close = df['close']
        
        if len(close) < 26:
            return {'histogram': 0.0, 'signal': 0.0, 'macd': 0.0}
        
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'histogram': float(histogram.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'macd': float(macd_line.iloc[-1])
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        """Calculate Bollinger Bands"""
        close = df['close']
        
        if len(close) < 20:
            current = float(close.iloc[-1])
            return {'upper': current, 'middle': current, 'lower': current, 'width': 0.0}
        
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        current_price = float(close.iloc[-1])
        bb_position = (current_price - float(lower_band.iloc[-1])) / \
                     (float(upper_band.iloc[-1]) - float(lower_band.iloc[-1]))
        
        return {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower_band.iloc[-1]),
            'width': float((upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1]),
            'position': bb_position
        }
    
    def _calculate_emas(self, df: pd.DataFrame) -> Dict:
        """Calculate EMAs"""
        close = df['close']
        emas = {}
        
        for period in self.config.EMA_PERIODS:
            if len(close) >= period:
                ema = close.ewm(span=period, adjust=False).mean()
                emas[f'ema_{period}'] = float(ema.iloc[-1])
            else:
                emas[f'ema_{period}'] = float(close.iloc[-1])
        
        return emas
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return float(df['close'].iloc[-1] * 0.02)
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not atr.empty else float(close.iloc[-1] * 0.02)
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        if len(close) < k_period + d_period:
            return {'k': 50.0, 'd': 50.0}
        
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        
        k = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
        d = k.rolling(window=d_period).mean()
        
        return {
            'k': float(k.iloc[-1]),
            'd': float(d.iloc[-1])
        }
    
    def _check_ema_alignment(self, emas: Dict, trend: str) -> bool:
        """Check EMA alignment"""
        if trend == 'BULLISH':
            return emas.get('ema_9', 0) > emas.get('ema_21', 0) > emas.get('ema_50', 0)
        elif trend == 'BEARISH':
            return emas.get('ema_9', 0) < emas.get('ema_21', 0) < emas.get('ema_50', 0)
        return False
    
    def _check_bb_position(self, bb: Dict, trend: str) -> bool:
        """Check Bollinger Bands position"""
        position = bb.get('position', 0.5)
        
        if trend == 'BULLISH':
            return position < 0.3  # Near lower band
        elif trend == 'BEARISH':
            return position > 0.7  # Near upper band
        return False
    
    def _detect_volume_spike(self, df: pd.DataFrame) -> bool:
        """Detect volume spike"""
        if len(df) < 20:
            return False
        
        volume = df['volume']
        volume_ma = volume.rolling(window=20).mean()
        
        return float(volume.iloc[-1]) > float(volume_ma.iloc[-1]) * 1.5
    
    def _detect_volume_divergence(self, df: pd.DataFrame) -> bool:
        """Detect volume divergence"""
        return False  # Simplified
    
    def _get_timeframe_summary(self, data_frames: Dict) -> Dict:
        """Get timeframe summary including direction and structure for each TF"""
        mtf = self._analyze_multi_timeframe_structure(data_frames)
        return {
            'directions': mtf.get('timeframe_directions', {}),
            'structures': mtf.get('timeframe_structures', {}),
            'weighted_alignment': mtf.get('weighted_alignment', 0.0),
            'all_aligned': mtf.get('all_aligned', False)
        }