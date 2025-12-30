#!/usr/bin/env python3
"""
Signal Orchestrator
==================
Background scheduler for signal refresh and broadcasting
"""

import threading
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class SignalOrchestrator:
    """Background service that refreshes signals and broadcasts updates"""
    
    def __init__(self, 
                 signal_engine,
                 signal_repo,
                 socketio,
                 refresh_interval_secs: int = 30):
        """
        Args:
            signal_engine: SignalEngineService
            signal_repo: SignalRepository
            socketio: Flask-SocketIO instance
            refresh_interval_secs: Refresh interval in seconds
        """
        self.engine = signal_engine
        self.repo = signal_repo
        self.socketio = socketio
        self.refresh_interval = refresh_interval_secs
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self, symbols: List[str]):
        """Start background refresh loop"""
        
        logger.info(f"🚀 Starting Signal Orchestrator (interval: {self.refresh_interval}s)...")
        self.running = True
        
        def scheduler_loop():
            """Main refresh loop"""
            while self.running:
                try:
                    # Generate signals for all symbols
                    signals = self.engine.generate_for_all(symbols, timeframe="15m")
                    
                    logger.info(f"📊 SIGNAL GENERATION: Generated {len(signals)} signals")
                    
                    # Store in repository
                    for symbol, signal in signals.items():
                        logger.info(f"💾 SAVING [{symbol}]: {signal.direction} @ ${signal.entry_price:.2f} (conf={signal.confidence_score}%)")
                        
                        try:
                            result = self.repo.upsert_latest(signal)
                            logger.info(f"✅ SAVED [{symbol}]: Result={result}")
                        except Exception as save_err:
                            logger.error(f"❌ FAILED to save [{symbol}]: {type(save_err).__name__}: {str(save_err)[:100]}")
                    
                    # Broadcast to all connected clients via SocketIO
                    if signals:
                        signal_dicts = {
                            symbol: signal.to_dict()
                            for symbol, signal in signals.items()
                        }
                        
                        try:
                            self.socketio.emit(
                                'prediction:update',
                                {
                                    'timestamp': time.time(),
                                    'predictions': signal_dicts
                                }
                            )
                            logger.debug(f"📡 Broadcast {len(signals)} signals")
                        
                        except Exception as e:
                            logger.warning(f"⚠️  SocketIO emit error: {e}")
                    
                    # Sleep before next refresh
                    time.sleep(self.refresh_interval)
                
                except Exception as e:
                    logger.error(f"❌ Scheduler error: {e}")
                    time.sleep(self.refresh_interval)
        
        # Start in background thread
        self.thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.thread.start()
        logger.info("✅ Signal Orchestrator started")
    
    def stop(self):
        """Stop background refresh loop gracefully"""
        logger.info("🛑 Stopping Signal Orchestrator...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("✅ Signal Orchestrator stopped")
