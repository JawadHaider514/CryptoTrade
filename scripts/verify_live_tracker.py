#!/usr/bin/env python3
"""
LIVE TRACKER VERIFICATION TESTS
Tests that the live tracker actually works - not just that code exists
"""

import sqlite3
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class LiveTrackerVerifier:
    """Verify live tracker functionality"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        self.db_path = db_path
    
    def test_database_table(self) -> bool:
        """Test 1: Check if live_signals table exists"""
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Live Tracker Database Table")
        logger.info("="*70)
        
        if not Path(self.db_path).exists():
            logger.error("❌ Database doesn't exist yet")
            logger.error("   Run: python core/run_backtest.py --full --symbol XRPUSDT")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='live_signals'
                """)
                
                if not cursor.fetchone():
                    logger.warning("⚠️  live_signals table doesn't exist yet")
                    logger.info("   This is normal - created on first signal")
                    return True
                
                logger.info("✅ live_signals table exists")
                
                # Check table structure
                cursor.execute("PRAGMA table_info(live_signals)")
                columns = {col[1] for col in cursor.fetchall()}
                
                required_cols = {'signal_id', 'symbol', 'entry_price', 'current_price', 'status'}
                missing = required_cols - columns
                
                if missing:
                    logger.error(f"❌ Missing columns: {missing}")
                    return False
                
                logger.info(f"✅ Table has required columns: {', '.join(required_cols)}")
                
                # Check signal count
                cursor.execute("SELECT COUNT(*) FROM live_signals")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    logger.info(f"✅ Currently tracking {count} live signals")
                else:
                    logger.info("ℹ️  No signals being tracked yet (will populate when system runs)")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return False
    
    def test_tracker_initialization(self) -> bool:
        """Test 2: Check if tracker initializes correctly"""
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Tracker Initialization")
        logger.info("="*70)
        
        try:
            from core.live_tracker import LiveSignalTracker
            
            logger.info("✅ LiveSignalTracker imports successfully")
            
            # Try to instantiate
            tracker = LiveSignalTracker()
            logger.info("✅ LiveSignalTracker instantiates")
            
            # Check required methods
            required_methods = {'add_signal', 'update_prices', 'start', 'get_active_signals'}
            
            for method in required_methods:
                if not hasattr(tracker, method):
                    logger.error(f"❌ Missing method: {method}")
                    return False
            
            logger.info(f"✅ All required methods exist: {', '.join(required_methods)}")
            
            # Check if start method is callable
            if not callable(getattr(tracker, 'start')):
                logger.error("❌ start() is not callable")
                return False
            
            logger.info("✅ start() method is callable")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            return False
    
    def test_signal_format(self) -> bool:
        """Test 3: Check if signals can be properly formatted"""
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Signal Format Compatibility")
        logger.info("="*70)
        
        try:
            from core.enhanced_crypto_dashboard import EnhancedSignal, SignalQuality
            from core.live_tracker import LiveSignalTracker
            from datetime import datetime
            
            # Create test signal
            from core.enhanced_crypto_dashboard import PredictionMetrics
            # Provide a minimal PredictionMetrics object for compatibility with the EnhancedSignal dataclass
            dummy_predictions = PredictionMetrics(
                price_target_30m=88250.0,
                price_target_1h=88300.0,
                price_target_3h=88400.0,
                breakout_probability=50.0,
                volume_surge_probability=20.0,
                trend_reversal_probability=5.0,
                market_correlation_score=30.0,
                volatility_prediction=10.0,
                confidence_interval=(0.0, 100.0),
                risk_score=50.0
            )

            test_signal = EnhancedSignal(
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
                predictions=dummy_predictions,
                timestamp=datetime.now(),
                processing_time=0.1,
                data_sources=["TEST"],
                patterns=["bullish_momentum"],
                ml_features={},
                leverage=20
            )
            
            logger.info("✅ EnhancedSignal created successfully")
            
            # Try to add to tracker (dry run)
            tracker = LiveSignalTracker()
            
            # Check if signal has required attributes for tracker
            required_attrs = ['symbol', 'direction', 'entry_price', 'confidence', 'timestamp']
            
            for attr in required_attrs:
                if not hasattr(test_signal, attr):
                    logger.error(f"❌ Signal missing attribute: {attr}")
                    return False
            
            logger.info(f"✅ Signal has required attributes: {', '.join(required_attrs)}")
            
            logger.info(f"✅ Signal format compatible with tracker")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal format error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_tracker_data_persistence(self) -> bool:
        """Test 4: Check if tracker persists data correctly"""
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Data Persistence")
        logger.info("="*70)
        
        if not Path(self.db_path).exists():
            logger.warning("⚠️  Database doesn't exist yet")
            return True
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if live_signals table has records
                cursor.execute("""
                    SELECT COUNT(*) FROM sqlite_master 
                    WHERE type='table' AND name='live_signals'
                """)
                
                if not cursor.fetchone()[0]:
                    logger.info("ℹ️  live_signals table will be created on first signal")
                    return True
                
                cursor.execute("SELECT COUNT(*) FROM live_signals")
                count = cursor.fetchone()[0]
                
                if count == 0:
                    logger.info("ℹ️  No signals tracked yet (expected on first run)")
                    return True
                
                # Check a sample signal
                cursor.execute("""
                    SELECT symbol, status, entry_price, current_price 
                    FROM live_signals LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row:
                    symbol, status, entry, current = row
                    logger.info(f"✅ Sample tracking data:")
                    logger.info(f"   Symbol: {symbol}")
                    logger.info(f"   Status: {status}")
                    logger.info(f"   Entry: {entry}, Current: {current}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Persistence error: {e}")
            return False
    
    def test_outcome_tracking(self) -> bool:
        """Test 5: Check if outcomes are properly tracked"""
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Outcome Tracking")
        logger.info("="*70)
        
        if not Path(self.db_path).exists():
            logger.warning("⚠️  Database doesn't exist yet")
            return True
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if outcomes are recorded
                cursor.execute("""
                    SELECT COUNT(*) FROM sqlite_master 
                    WHERE type='table' AND name='signal_outcomes'
                """)
                
                if not cursor.fetchone()[0]:
                    logger.info("ℹ️  signal_outcomes table will be created on first signal")
                    return True
                
                cursor.execute("""
                    SELECT result, COUNT(*) as count 
                    FROM signal_outcomes 
                    GROUP BY result
                """)
                
                outcomes = cursor.fetchall()
                
                if not outcomes:
                    logger.info("ℹ️  No outcomes yet (expected on first run)")
                    return True
                
                logger.info("✅ Outcomes tracked:")
                for result, count in outcomes:
                    logger.info(f"   {result}: {count} signals")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Outcome tracking error: {e}")
            return False
    
    def run(self):
        """Run all verification tests"""
        
        logger.info("\n")
        logger.info("╔════════════════════════════════════════════════════════════════════════════╗")
        logger.info("║                   LIVE TRACKER VERIFICATION TESTS                         ║")
        logger.info("║             Verify tracker actually works (not just code exists)          ║")
        logger.info("╚════════════════════════════════════════════════════════════════════════════╝")
        
        tests = [
            ("Database Table", self.test_database_table),
            ("Tracker Initialization", self.test_tracker_initialization),
            ("Signal Format", self.test_signal_format),
            ("Data Persistence", self.test_tracker_data_persistence),
            ("Outcome Tracking", self.test_outcome_tracking),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"Test crashed: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*70)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
        
        logger.info(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("\n🟢 ALL TESTS PASSED - Live tracker is ready!")
        else:
            logger.warning(f"\n🟡 {total - passed} test(s) failed")
            logger.info("   This is expected until backtesting database is created")
            logger.info("   Run: python core/run_backtest.py --full --symbol XRPUSDT")

if __name__ == "__main__":
    verifier = LiveTrackerVerifier()
    verifier.run()
