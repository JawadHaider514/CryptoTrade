#!/usr/bin/env python3
"""
EXTRACT ML FEATURES FROM BACKTESTING DATABASE
Exports backtest signal data to CSV format for ML model training
"""

import sqlite3
import csv
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MLFeatureExtractor:
    """Extract ML features from backtesting database"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        self.db_path = db_path
        self.output_file = "data/ml_training_data.csv"
    
    def check_database(self) -> bool:
        """Verify database and tables exist"""
        if not Path(self.db_path).exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            logger.error("   Run: python core/run_backtest.py --full --symbol XRPUSDT")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {t[0] for t in cursor.fetchall()}
                
                required = {'backtest_signals', 'signal_outcomes'}
                missing = required - tables
                
                if missing:
                    logger.error(f"❌ Missing tables: {missing}")
                    return False
                
                # Check data
                cursor.execute("SELECT COUNT(*) FROM backtest_signals")
                signal_count = cursor.fetchone()[0]
                
                if signal_count == 0:
                    logger.error("❌ backtest_signals table is empty")
                    return False
                
                logger.info(f"✅ Database valid: {signal_count} signals found")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return False
    
    def extract_features(self) -> int:
        """Extract features and write to CSV"""
        
        logger.info(f"\n📊 Extracting ML training data...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Query all signals with outcomes
                cursor.execute("""
                    SELECT 
                        bs.id,
                        bs.symbol,
                        bs.confluence_score,
                        bs.direction,
                        bs.entry_price,
                        bs.stop_loss,
                        bs.take_profit_1,
                        bs.take_profit_2,
                        bs.take_profit_3,
                        bs.timeframe,
                        bs.patterns,
                        bs.rsi_value,
                        bs.macd_value,
                        bs.volume_ratio,
                        bs.trend_strength,
                        so.result,
                        so.exit_price,
                        so.profit_loss,
                        so.pnl_percentage
                    FROM backtest_signals bs
                    LEFT JOIN signal_outcomes so ON bs.id = so.signal_id
                    ORDER BY bs.id
                """)
                
                rows = cursor.fetchall()
                logger.info(f"✅ Found {len(rows)} signals with data")
                
                # Write to CSV
                with open(self.output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Headers
                    headers = [
                        'signal_id', 'symbol', 'confluence_score', 'direction',
                        'entry_price', 'stop_loss', 'tp1', 'tp2', 'tp3',
                        'timeframe', 'patterns', 'rsi', 'macd', 'volume_ratio',
                        'trend_strength', 'result', 'exit_price', 'pnl',
                        'pnl_percentage'
                    ]
                    writer.writerow(headers)
                    
                    # Data rows
                    for row in rows:
                        writer.writerow([
                            row['id'], row['symbol'], row['confluence_score'],
                            row['direction'], row['entry_price'], row['stop_loss'],
                            row['take_profit_1'], row['take_profit_2'],
                            row['take_profit_3'], row['timeframe'], row['patterns'],
                            row['rsi_value'], row['macd_value'], row['volume_ratio'],
                            row['trend_strength'], row['result'], row['exit_price'],
                            row['profit_loss'], row['pnl_percentage']
                        ])
                
                logger.info(f"✅ Exported {len(rows)} signals to {self.output_file}")
                return len(rows)
                
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return 0
    
    def generate_statistics(self):
        """Generate basic statistics from extracted data"""
        
        logger.info(f"\n📈 Feature Statistics:")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Win rate by score range
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN confluence_score >= 85 THEN '85+'
                            WHEN confluence_score >= 75 THEN '75-84'
                            WHEN confluence_score >= 65 THEN '65-74'
                            ELSE '<65'
                        END as score_range,
                        COUNT(*) as total,
                        SUM(CASE WHEN so.result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        ROUND(100.0 * SUM(CASE WHEN so.result = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
                    FROM backtest_signals bs
                    LEFT JOIN signal_outcomes so ON bs.id = so.signal_id
                    GROUP BY score_range
                    ORDER BY confluence_score DESC
                """)
                
                results = cursor.fetchall()
                
                for score_range, total, wins, win_rate in results:
                    wins_val = wins if wins else 0
                    rate = win_rate if win_rate else 0
                    logger.info(f"   {score_range}: {wins_val}/{total} = {rate:.1f}%")
                
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
    
    def run(self):
        """Run full extraction"""
        
        logger.info("\n" + "="*70)
        logger.info("ML FEATURE EXTRACTION")
        logger.info("="*70)
        
        if not self.check_database():
            logger.error("\n❌ Cannot extract features - database issues")
            return False
        
        count = self.extract_features()
        
        if count == 0:
            logger.error("\n❌ No data extracted")
            return False
        
        self.generate_statistics()
        
        logger.info(f"\n✅ EXTRACTION COMPLETE")
        logger.info(f"   Input: {self.db_path} ({count} signals)")
        logger.info(f"   Output: {self.output_file}")
        logger.info(f"\n   Next step: python core/train_ml_model.py")
        
        return True

if __name__ == "__main__":
    extractor = MLFeatureExtractor()
    success = extractor.run()
    exit(0 if success else 1)
