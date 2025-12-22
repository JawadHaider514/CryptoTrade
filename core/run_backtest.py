#!/usr/bin/env python3
"""
COMPLETE BACKTESTING SYSTEM RUNNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrates all backtesting components:
1. Download historical data
2. Generate signals on historical data
3. Track actual outcomes
4. Calculate statistics
5. Generate reports
"""

import sys
import os
from datetime import datetime, timedelta
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backtest_system import HistoricalDataCollector
from signal_generator import HistoricalSignalGenerator
from outcome_tracker import OutcomeTracker
from statistics_calculator import BacktestStatisticsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteBacktestingSystem:
    """Complete backtesting system orchestrator"""
    
    def __init__(self, symbol: str = "XRPUSDT", db_path: str = "data/backtest.db"):
        """Initialize backtesting system"""
        self.symbol = symbol
        self.db_path = db_path
        
        self.collector = HistoricalDataCollector(db_path)
        self.signal_generator = HistoricalSignalGenerator(db_path)
        self.outcome_tracker = OutcomeTracker(db_path)
        self.stats_calculator = BacktestStatisticsCalculator(db_path)
    
    def step_1_download_data(self, days: int = 30):
        """STEP 1: Download historical data"""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: DOWNLOAD HISTORICAL DATA")
        logger.info("="*70)
        
        # Check if we already have data
        stats = self.collector.get_data_stats(self.symbol)
        
        if stats and 'days_of_data' in stats:
            logger.info(f"✅ Data already available: {stats['days_of_data']:.1f} days")
            return True
        
        logger.info(f"📥 Downloading {days} days of {self.symbol} data...")
        
        try:
            self.collector.download_30_days_of_data(self.symbol)
            
            # Show stats
            stats = self.collector.get_data_stats(self.symbol)
            logger.info("\n📊 Data Statistics:")
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download data: {e}")
            return False
    
    def step_2_generate_signals(self, lookback_days: int = 30, interval_minutes: int = 5):
        """STEP 2: Generate signals on historical data"""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: GENERATE SIGNALS ON HISTORICAL DATA")
        logger.info("="*70)
        
        # Get data period
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)
        
        logger.info(f"Period: {start_time.date()} to {end_time.date()}")
        logger.info(f"Interval: {interval_minutes} minutes")
        
        try:
            signal_count = self.signal_generator.generate_signals_for_period(
                self.symbol,
                start_time,
                end_time,
                interval_minutes=interval_minutes
            )
            
            logger.info(f"✅ Generated {signal_count} signals")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate signals: {e}")
            return False
    
    def step_3_track_outcomes(self):
        """STEP 3: Track what actually happened"""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: TRACK SIGNAL OUTCOMES")
        logger.info("="*70)
        
        logger.info("Checking what ACTUALLY happened with each signal...")
        
        try:
            tracked_count = self.outcome_tracker.track_all_signals(self.symbol)
            logger.info(f"✅ Tracked {tracked_count} signal outcomes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to track outcomes: {e}")
            return False
    
    def step_4_calculate_statistics(self):
        """STEP 4: Calculate real performance metrics"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: CALCULATE STATISTICS & METRICS")
        logger.info("="*70)
        
        try:
            # Generate comprehensive report
            report = self.stats_calculator.generate_comprehensive_report(self.symbol)
            logger.info(report)
            
            # Save to file
            self.stats_calculator.save_report_to_file(self.symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate statistics: {e}")
            return False
    
    def run_complete_backtest(self):
        """Run complete backtesting workflow"""
        logger.info("\n" + "╔" + "="*68 + "╗")
        logger.info("║" + " "*68 + "║")
        logger.info("║  🎯 COMPLETE BACKTESTING SYSTEM" + " "*35 + "║")
        logger.info("║" + " "*68 + "║")
        logger.info("╚" + "="*68 + "╝")
        
        # Step 1: Download data
        if not self.step_1_download_data():
            logger.error("❌ Failed at data download step")
            return False
        
        # Step 2: Generate signals
        if not self.step_2_generate_signals():
            logger.error("❌ Failed at signal generation step")
            return False
        
        # Step 3: Track outcomes
        if not self.step_3_track_outcomes():
            logger.error("❌ Failed at outcome tracking step")
            return False
        
        # Step 4: Calculate statistics
        if not self.step_4_calculate_statistics():
            logger.error("❌ Failed at statistics calculation step")
            return False
        
        logger.info("\n" + "╔" + "="*68 + "╗")
        logger.info("║" + " "*68 + "║")
        logger.info("║  ✅ BACKTESTING COMPLETE!" + " "*40 + "║")
        logger.info("║" + " "*68 + "║")
        logger.info("╚" + "="*68 + "╝")
        
        return True
    
    def show_results_summary(self):
        """Show quick summary of results"""
        logger.info("\n" + "="*70)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("="*70)
        
        stats = self.stats_calculator.calculate_overall_stats(self.symbol)
        
        if 'error' in stats:
            logger.warning(f"⚠️  No results available yet")
            return
        
        logger.info(f"\nSymbol: {self.symbol}")
        logger.info(f"Total Signals Tested: {stats['total_signals']}")
        logger.info(f"\n✅ Wins: {stats['wins']} ({stats['win_rate']:.2f}%)")
        logger.info(f"❌ Losses: {stats['losses']} ({stats['loss_rate']:.2f}%)")
        logger.info(f"⏱️  Timeouts: {stats['timeouts']} ({stats['timeout_rate']:.2f}%)")
        logger.info(f"\n💰 Total Profit: ${stats['total_profit']:.2f}")
        logger.info(f"💸 Total Loss: ${stats['total_loss']:.2f}")
        logger.info(f"📊 Net Profit: ${stats['net_profit']:.2f}")
        logger.info(f"📈 Profit Factor: {stats['profit_factor']:.2f}x")
        logger.info(f"🎯 Expectancy: {stats['expectancy']:.4f}%")
        
        # Show accuracy by score
        by_score = self.stats_calculator.calculate_accuracy_by_confluence_score(self.symbol)
        if by_score:
            logger.info(f"\n📊 ACCURACY BY CONFLUENCE SCORE:")
            for score_range, data in by_score.items():
                logger.info(f"   {score_range}: {data['win_rate']:.2f}% ({data['signals']} signals)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Backtesting System')
    parser.add_argument('--symbol', default='XRPUSDT', help='Trading symbol')
    parser.add_argument('--db', default='data/backtest.db', help='Database path')
    parser.add_argument('--full', action='store_true', help='Run full backtest (download + signals + outcomes + stats)')
    parser.add_argument('--data-only', action='store_true', help='Download data only')
    parser.add_argument('--signals-only', action='store_true', help='Generate signals only')
    parser.add_argument('--outcomes-only', action='store_true', help='Track outcomes only')
    parser.add_argument('--stats-only', action='store_true', help='Calculate stats only')
    parser.add_argument('--summary', action='store_true', help='Show results summary')
    
    args = parser.parse_args()
    
    system = CompleteBacktestingSystem(args.symbol, args.db)
    
    if args.full:
        system.run_complete_backtest()
    elif args.data_only:
        system.step_1_download_data()
    elif args.signals_only:
        system.step_2_generate_signals()
    elif args.outcomes_only:
        system.step_3_track_outcomes()
    elif args.stats_only:
        system.step_4_calculate_statistics()
    elif args.summary:
        system.show_results_summary()
    else:
        # Default: show summary
        system.show_results_summary()
