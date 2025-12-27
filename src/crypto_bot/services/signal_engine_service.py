#!/usr/bin/env python3
"""
Signal Engine Service
====================
Decision engine pipeline:
1. Professional Analyzer (best - 5+ year trader logic)
2. Enhanced Dashboard Fallback (if pro fails)
3. Minimal Fallback (last resort with low confidence)
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from crypto_bot.domain.signal_models import SignalModel, TakeProfit
from crypto_bot.services.market_data_service import MarketDataService
from crypto_bot.services.market_history_service import get_market_history_service

# Load feature flags from settings
try:
    from config.settings import USE_PRO_ANALYZER, SIGNAL_VALID_MINUTES, MIN_CONFIDENCE, MIN_ACCURACY
except ImportError:
    # Fallback defaults
    USE_PRO_ANALYZER = True
    SIGNAL_VALID_MINUTES = 240
    MIN_CONFIDENCE = 15
    MIN_ACCURACY = 0

logger = logging.getLogger(__name__)


class SignalEngineService:
    """
    Decision Engine with 3-step pipeline:
    1. Professional Analyzer (best quality)
    2. Enhanced Dashboard Fallback
    3. Minimal Fallback (last resort)
    """
    
    def __init__(self, market_data: MarketDataService, enhanced_dashboard=None, professional_analyzer=None):
        """
        Args:
            market_data: MarketDataService for live prices
            enhanced_dashboard: EnhancedScalpingDashboard instance (fallback)
            professional_analyzer: ProfessionalAnalyzer instance (primary)
        """
        self.market_data = market_data
        self.dashboard = enhanced_dashboard
        self.pro_analyzer = professional_analyzer
    
    def normalize_analysis(self, analysis: Optional[Dict], symbol: str) -> Optional[Dict]:
        """
        Normalize analyzer output to standard format.
        Handles various return formats from different analyzers.
        
        Returns normalized dict with keys:
        - direction, entry, sl, take_profits, confidence_score, accuracy_percent, source
        """
        if not analysis:
            return None
        
        # If analysis has nested 'setup' key, use that
        if isinstance(analysis, dict) and 'setup' in analysis:
            setup = analysis['setup']
        else:
            setup = analysis
        
        if not isinstance(setup, dict):
            logger.warning(f"[{symbol}] Unexpected analysis structure (not dict): {type(setup)}")
            return None
        
        logger.info(f"[{symbol}] Normalizing analysis: keys={list(setup.keys())[:5]}...")
        
        # Normalize entry price (support entry_price, entry_range, entry)
        entry = None
        if 'entry_price' in setup:
            entry = setup['entry_price']
        elif 'entry_range' in setup:
            # Handle entry_range as [low, high] tuple/list
            er = setup['entry_range']
            if isinstance(er, (list, tuple)) and len(er) >= 2:
                entry = (er[0] + er[1]) / 2  # Use midpoint
            elif isinstance(er, dict):
                entry = er.get('mid') or er.get('entry') or (er.get('low', 0) + er.get('high', 0)) / 2
        elif 'entry' in setup:
            entry = setup['entry']
        
        # Normalize stop loss (support stop_loss, sl)
        sl = None
        if 'stop_loss' in setup:
            sl = setup['stop_loss']
        elif 'sl' in setup:
            sl = setup['sl']
        
        # Normalize take profits (support list of tuples, dicts, floats)
        tp_list = []
        if 'take_profits' in setup:
            tp_raw = setup['take_profits']
            if isinstance(tp_raw, (list, tuple)):
                for i, tp in enumerate(tp_raw):
                    if isinstance(tp, dict):
                        tp_list.append(tp.get('price') or tp.get('value') or 0)
                    elif isinstance(tp, (list, tuple)):
                        tp_list.append(tp[0] if len(tp) > 0 else 0)  # First element is price
                    else:
                        tp_list.append(float(tp) if tp else 0)
        else:
            # Fallback: check for individual tp fields (take_profit_1, take_profit_2, take_profit_3)
            for i in [1, 2, 3]:
                tp_key = f'take_profit_{i}'
                if tp_key in setup and setup[tp_key]:
                    tp_list.append(float(setup[tp_key]))
        
        # Normalize confidence (support confidence_score, confidence, confluence_score)
        confidence = 50
        if 'confidence_score' in setup:
            confidence = setup['confidence_score']
        elif 'confidence' in setup:
            confidence = setup['confidence']
        elif 'confluence_score' in setup:
            confidence = setup['confluence_score']
        
        # Normalize accuracy (support accuracy_percent, accuracy, accuracy_estimate)
        accuracy = 50.0
        if 'accuracy_percent' in setup:
            accuracy = setup['accuracy_percent']
        elif 'accuracy' in setup:
            accuracy = setup['accuracy']
        elif 'accuracy_estimate' in setup:
            accuracy = setup['accuracy_estimate']
        
        # Check if essential fields are present
        if entry is None or sl is None or not tp_list:
            logger.warning(f"[{symbol}] Missing essential fields after normalization: entry={entry}, sl={sl}, tp_count={len(tp_list)}")
            return None
        
        # Build normalized result
        normalized = {
            'direction': setup.get('direction', 'LONG'),
            'entry': entry,
            'sl': sl,
            'take_profits': tp_list,
            'confidence_score': int(confidence),
            'accuracy_percent': float(accuracy),
            'leverage': setup.get('leverage', 5),
            'reasons': setup.get('reasons', []),
            'patterns': setup.get('patterns', []),
            'market_context': setup.get('market_context', ''),
            'source': setup.get('source', 'UNKNOWN')
        }
        
        logger.info(f"[{symbol}] ✓ Analysis normalized: entry={entry:.2f}, sl={sl:.2f}, conf={confidence}, acc={accuracy:.1f}")
        return normalized
    
    
    def _step1_professional_analyzer(self, symbol: str) -> Optional[Dict]:
        """
        STEP 1 (Best): Use ProfessionalAnalyzer with MarketHistoryService
        Fetches 6 timeframes from Binance, analyzes with 5+ year trader logic
        """
        if not self.pro_analyzer:
            logger.info(f"[{symbol}] STEP 1 SKIPPED: Professional analyzer not available")
            return None
        
        logger.info(f"[{symbol}] STEP 1 STARTING: Professional Analyzer")
        
        try:
            # Get historical dataframes from MarketHistoryService
            hist_service = get_market_history_service(cache_ttl=45)
            dfs = hist_service.get_dataframes(symbol)
            
            if not dfs or all(df.empty for df in dfs.values()):
                logger.info(f"[{symbol}] STEP 1 FAILED: No historical data available")
                return None
            
            logger.info(f"[{symbol}] STEP 1: Historical data loaded ({len([d for d in dfs.values() if not d.empty])} timeframes)")
            
            # Run ProfessionalAnalyzer
            setup = self.pro_analyzer.analyze_complete_setup(symbol, dfs)
            
            if not setup:
                logger.info(f"[{symbol}] STEP 1 FAILED: analyze_complete_setup returned None")
                return None
            
            logger.info(f"[{symbol}] STEP 1: Raw setup received (type={type(setup).__name__})")
            
            # Normalize the output
            analysis = self.normalize_analysis(setup, symbol)
            
            if not analysis:
                logger.info(f"[{symbol}] STEP 1 FAILED: Normalization failed or missing required fields")
                return None
            
            # Mark source
            analysis['source'] = 'PRO'
            logger.info(f"✅ [${symbol}] STEP 1 SUCCESS: Professional Analyzer")
            return analysis
            
        except Exception as e:
            logger.info(f"[{symbol}] STEP 1 EXCEPTION: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def _step2_enhanced_dashboard(self, symbol: str) -> Optional[Dict]:
        """
        STEP 2: Fallback to EnhancedDashboard/StreamingProcessor
        Uses existing dashboard analyzer if professional analyzer fails
        """
        if not self.dashboard:
            logger.info(f"[{symbol}] STEP 2 SKIPPED: Dashboard not available")
            return None
        
        logger.info(f"[{symbol}] STEP 2 STARTING: Enhanced Dashboard Analyzer")
        
        try:
            # Try signal_analyzer.analyze_symbol
            if hasattr(self.dashboard, 'signal_analyzer'):
                logger.info(f"[{symbol}] STEP 2a: Checking signal_analyzer...")
                
                if hasattr(self.dashboard.signal_analyzer, 'analyze_symbol'):
                    try:
                        result = self.dashboard.signal_analyzer.analyze_symbol(symbol)
                        logger.info(f"[{symbol}] STEP 2a: analyze_symbol returned (type={type(result).__name__})")
                        
                        if result:
                            analysis = self.normalize_analysis(result, symbol)
                            if analysis:
                                analysis['source'] = 'DASHBOARD'
                                logger.info(f"✅ [{symbol}] STEP 2a SUCCESS: signal_analyzer")
                                return analysis
                            else:
                                logger.info(f"[{symbol}] STEP 2a: Normalization failed")
                        else:
                            logger.info(f"[{symbol}] STEP 2a: analyze_symbol returned None/empty")
                    except Exception as e:
                        logger.info(f"[{symbol}] STEP 2a EXCEPTION: {type(e).__name__}: {str(e)[:80]}")
                else:
                    logger.info(f"[{symbol}] STEP 2a: signal_analyzer.analyze_symbol not available")
            else:
                logger.info(f"[{symbol}] STEP 2a: dashboard.signal_analyzer does not exist")
            
            # Try streaming_processor.process_symbols_batch
            if hasattr(self.dashboard, 'streaming_processor'):
                logger.info(f"[{symbol}] STEP 2b: Checking streaming_processor...")
                
                if hasattr(self.dashboard.streaming_processor, 'process_symbols_batch'):
                    try:
                        # Call process_symbols_batch() - returns List[EnhancedSignal]
                        batch_result = self.dashboard.streaming_processor.process_symbols_batch()
                        logger.info(f"[{symbol}] STEP 2b: process_symbols_batch returned (type={type(batch_result).__name__}, len={len(batch_result) if isinstance(batch_result, list) else 'N/A'})")
                        
                        if batch_result:
                            # Handle List[EnhancedSignal] format
                            if isinstance(batch_result, list):
                                # Find signal for current symbol
                                symbol_signal = None
                                for sig in batch_result:
                                    sig_symbol = getattr(sig, 'symbol', None)
                                    if sig_symbol == symbol:
                                        symbol_signal = sig
                                        break
                                
                                if symbol_signal:
                                    logger.info(f"[{symbol}] STEP 2b: Found signal in batch list")
                                    
                                    # Convert EnhancedSignal to dict for normalization
                                    if hasattr(symbol_signal, '__dict__'):
                                        signal_dict = symbol_signal.__dict__.copy()
                                    elif isinstance(symbol_signal, dict):
                                        signal_dict = symbol_signal.copy()
                                    else:
                                        logger.info(f"[{symbol}] STEP 2b: Unable to convert signal to dict")
                                        signal_dict = {}
                                    
                                    # Build take_profits list from separate TP fields if they exist
                                    if 'take_profit_1' in signal_dict or 'take_profit_2' in signal_dict or 'take_profit_3' in signal_dict:
                                        tps = []
                                        if 'take_profit_1' in signal_dict and signal_dict['take_profit_1']:
                                            tps.append(signal_dict['take_profit_1'])
                                        if 'take_profit_2' in signal_dict and signal_dict['take_profit_2']:
                                            tps.append(signal_dict['take_profit_2'])
                                        if 'take_profit_3' in signal_dict and signal_dict['take_profit_3']:
                                            tps.append(signal_dict['take_profit_3'])
                                        if tps:
                                            signal_dict['take_profits'] = tps
                                    
                                    analysis = self.normalize_analysis(signal_dict, symbol)
                                    
                                    if analysis:
                                        analysis['source'] = 'DASHBOARD'
                                        logger.info(f"✅ [{symbol}] STEP 2b SUCCESS: streaming_processor (from list)")
                                        return analysis
                                    else:
                                        logger.info(f"[{symbol}] STEP 2b: Normalization failed")
                                else:
                                    logger.info(f"[{symbol}] STEP 2b: Symbol not found in batch list. Available symbols: {[getattr(s, 'symbol', '?') for s in batch_result[:5]]}")
                            
                            # Handle dict format (backward compatibility)
                            elif isinstance(batch_result, dict):
                                if symbol in batch_result:
                                    result = batch_result[symbol]
                                    logger.info(f"[{symbol}] STEP 2b: Found symbol in batch dict (type={type(result).__name__})")
                                    analysis = self.normalize_analysis(result, symbol)
                                    
                                    if analysis:
                                        analysis['source'] = 'DASHBOARD'
                                        logger.info(f"✅ [{symbol}] STEP 2b SUCCESS: streaming_processor (from dict)")
                                        return analysis
                                    else:
                                        logger.info(f"[{symbol}] STEP 2b: Normalization failed")
                                else:
                                    logger.info(f"[{symbol}] STEP 2b: Symbol not found in batch dict. Keys: {list(batch_result.keys())[:5]}")
                            else:
                                logger.info(f"[{symbol}] STEP 2b: Unexpected batch result type: {type(batch_result).__name__}")
                        else:
                            logger.info(f"[{symbol}] STEP 2b: process_symbols_batch returned empty list/dict")
                    except Exception as e:
                        logger.info(f"[{symbol}] STEP 2b EXCEPTION: {type(e).__name__}: {str(e)[:80]}")
                else:
                    logger.info(f"[{symbol}] STEP 2b: streaming_processor.process_symbols_batch not available")
            else:
                logger.info(f"[{symbol}] STEP 2b: dashboard.streaming_processor does not exist")
            
            logger.info(f"[{symbol}] STEP 2 FAILED: Both signal_analyzer and streaming_processor failed/unavailable")
            return None
            
        except Exception as e:
            logger.info(f"[{symbol}] STEP 2 EXCEPTION: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def _step3_minimal_fallback(self, symbol: str, current_price: float) -> Dict:
        """
        STEP 3 (Last resort): Minimal signal with low confidence
        Used only when both pro analyzer and dashboard fail
        """
        logger.info(f"⚠️  [{symbol}] STEP 3 FALLBACK: Both Step 1 and Step 2 failed. Using minimal signal at price={current_price:.2f}")
        
        analysis = {
            'direction': 'LONG',  # Neutral fallback
            'entry': current_price,
            'sl': current_price * 0.98,
            'take_profits': [
                current_price * 1.01,
                current_price * 1.02,
                current_price * 1.03,
            ],
            'confidence_score': 15,  # Low confidence
            'accuracy_percent': 0.0,
            'leverage': 1,  # Minimal leverage
            'reasons': ['No analyzer available - minimal signal'],
            'patterns': [],
            'market_context': 'Fallback mode - waiting for data',
            'source': 'FALLBACK'  # Mark source
        }
        
        return analysis
    
    def generate_for_symbol(self, symbol: str, timeframe: str = "15m") -> Optional[SignalModel]:
        """
        Generate signal for ONE symbol using 3-step pipeline:
        1. Professional Analyzer (best)
        2. Enhanced Dashboard (fallback)
        3. Minimal Signal (last resort)
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{symbol}] GENERATING SIGNAL")
        logger.info(f"{'='*60}")
        
        # Get current price (needed for all steps)
        current_price = self.market_data.get_price(symbol)
        if not current_price:
            logger.warning(f"[{symbol}] ⚠️  FATAL: No price available - cannot generate signal")
            return None
        
        logger.info(f"[{symbol}] Current price: ${current_price:.2f}")
        
        # STEP 1: Try Professional Analyzer (if enabled by feature flag)
        analysis = None
        if USE_PRO_ANALYZER:
            analysis = self._step1_professional_analyzer(symbol)
        else:
            logger.info(f"[{symbol}] STEP 1 SKIPPED: Feature flag USE_PRO_ANALYZER=false")
        
        # STEP 2: Fallback to Enhanced Dashboard
        if analysis is None:
            analysis = self._step2_enhanced_dashboard(symbol)
        
        # STEP 3: Last resort minimal fallback
        if analysis is None:
            analysis = self._step3_minimal_fallback(symbol, current_price)
        
        if not analysis:
            logger.warning(f"[{symbol}] ⚠️  FATAL: No analysis generated in any step")
            return None
        
        # Extract signal data (already normalized)
        direction = analysis.get('direction', 'LONG')
        entry_price = float(analysis.get('entry', current_price))
        stop_loss = float(analysis.get('sl', entry_price * 0.95))
        confidence = int(analysis.get('confidence_score', 50))
        accuracy = float(analysis.get('accuracy_percent', 50.0))
        leverage = int(analysis.get('leverage', 10))
        source = analysis.get('source', 'UNKNOWN')
        
        logger.info(f"[{symbol}] Analysis extracted: dir={direction}, entry=${entry_price:.2f}, sl=${stop_loss:.2f}, conf={confidence}%, acc={accuracy:.1f}%, src={source}")
        
        # Apply minimum thresholds from feature flags
        if confidence < MIN_CONFIDENCE:
            logger.info(f"[{symbol}] ❌ FILTERED: confidence {confidence}% < MIN_CONFIDENCE {MIN_CONFIDENCE}%")
            return None
        
        if accuracy < MIN_ACCURACY:
            logger.info(f"[{symbol}] ❌ FILTERED: accuracy {accuracy:.1f}% < MIN_ACCURACY {MIN_ACCURACY}%")
            return None
        
        # Get or estimate take profits (already processed by analyzers)
        tp_data = analysis.get('take_profits', [])
        if not tp_data or len(tp_data) < 3:
            logger.info(f"[{symbol}] Generating synthetic TPs (received {len(tp_data) if tp_data else 0})")
            # Generate default TPs if not provided
            distance = abs(entry_price - stop_loss)
            tp_data = [
                entry_price + (distance * 0.5),   # TP1: 50% of risk
                entry_price + (distance * 1.0),   # TP2: 100% of risk
                entry_price + (distance * 1.5),   # TP3: 150% of risk
            ]
        
        # Create TakeProfit objects with ETAs based on SIGNAL_VALID_MINUTES
        now = datetime.utcnow()
        take_profits = [
            TakeProfit(
                level=1,
                price=float(tp_data[0]),
                eta=now + timedelta(minutes=min(5, SIGNAL_VALID_MINUTES // 4))
            ),
            TakeProfit(
                level=2,
                price=float(tp_data[1]),
                eta=now + timedelta(minutes=min(15, SIGNAL_VALID_MINUTES // 2))
            ),
            TakeProfit(
                level=3,
                price=float(tp_data[2]),
                eta=now + timedelta(minutes=SIGNAL_VALID_MINUTES)
            ),
        ]
        
        # Create signal model
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
            reasons=analysis.get('reasons', []),
            patterns=analysis.get('patterns', []),
            market_context=analysis.get('market_context', ''),
            source=source  # Add source (PRO, DASHBOARD, or FALLBACK)
        )
        
        logger.info(f"✅ [{symbol}] SIGNAL GENERATED: {direction} @ ${entry_price:.2f} (conf={confidence}%, src={source})")
        logger.info(f"{'='*60}\n")
        
        return signal
    
    def generate_for_all(self, symbols: List[str], timeframe: str = "15m") -> Dict[str, SignalModel]:
        """Generate signals for all symbols"""
        
        signals = {}
        
        for symbol in symbols:
            signal = self.generate_for_symbol(symbol, timeframe)
            if signal:
                signals[symbol] = signal
        
        return signals
