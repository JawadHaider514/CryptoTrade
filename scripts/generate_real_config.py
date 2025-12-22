#!/usr/bin/env python3
"""
GENERATE REAL CONFIG FROM BACKTEST DATA
This script extracts actual backtesting results and creates optimized_config.json
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealConfigGenerator:
    """Generate config from actual backtest results"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        self.db_path = db_path
        self.config: Dict[str, Any] = {}
    
    def check_backtest_data_exists(self) -> bool:
        """Verify backtest database has data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if tables exist and have data
                cursor.execute("SELECT COUNT(*) FROM backtest_signals")
                signal_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM signal_outcomes")
                outcome_count = cursor.fetchone()[0]
                
                logger.info(f"✅ Database has {signal_count} signals and {outcome_count} outcomes")
                
                return signal_count > 0 and outcome_count > 0
        except Exception as e:
            logger.error(f"❌ Cannot access backtest database: {e}")
            return False
    
    def get_accuracy_by_score(self) -> Dict[str, Dict]:
        """Extract REAL accuracy from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get signals with their outcomes
                query = """
                    SELECT 
                        bs.confluence_score,
                        so.result
                    FROM backtest_signals bs
                    JOIN signal_outcomes so ON bs.id = so.signal_id
                """
                
                rows = conn.execute(query).fetchall()
                
                if not rows:
                    logger.warning("❌ No signal outcomes found!")
                    return {}
                
                # Group by score ranges
                results = {}
                ranges = {
                    '85+': (85, 999),
                    '75-84': (75, 84),
                    '65-74': (65, 74),
                    '<65': (0, 64)
                }
                
                for range_name, (min_score, max_score) in ranges.items():
                    signals_in_range = [
                        r for r in rows 
                        if min_score <= r['confluence_score'] <= max_score
                    ]
                    
                    if not signals_in_range:
                        continue
                    
                    wins = sum(1 for s in signals_in_range if s['result'] == 'WIN')
                    total = len(signals_in_range)
                    win_rate = (wins / total * 100) if total > 0 else 0
                    
                    results[range_name] = {
                        'win_rate': round(win_rate, 1),
                        'signals': total,
                        'wins': wins
                    }
                    
                    logger.info(f"  {range_name}: {win_rate:.1f}% win rate ({total} signals)")
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Failed to get accuracy by score: {e}")
            return {}
    
    def get_pattern_performance(self) -> Dict[str, Dict]:
        """Extract REAL pattern performance from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = """
                    SELECT 
                        bs.patterns,
                        so.result
                    FROM backtest_signals bs
                    JOIN signal_outcomes so ON bs.id = so.signal_id
                    WHERE bs.patterns IS NOT NULL
                """
                
                rows = conn.execute(query).fetchall()
                
                if not rows:
                    logger.warning("⚠️  No pattern data found")
                    return {}
                
                import json
                pattern_stats = {}
                
                for row in rows:
                    try:
                        patterns = json.loads(row['patterns'])
                        if isinstance(patterns, list):
                            for pattern in patterns:
                                pattern_name = pattern.lower()
                                if pattern_name not in pattern_stats:
                                    pattern_stats[pattern_name] = {'wins': 0, 'total': 0}
                                
                                pattern_stats[pattern_name]['total'] += 1
                                if row['result'] == 'WIN':
                                    pattern_stats[pattern_name]['wins'] += 1
                    except:
                        continue
                
                # Calculate win rates and convert to scores
                results = {}
                for pattern, stats in pattern_stats.items():
                    win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    score = int(win_rate / 5)  # Convert 70% → 14 points (max 20)
                    
                    results[pattern] = {
                        'win_rate': round(win_rate, 1),
                        'score': score,
                        'signals': stats['total'],
                        'wins': stats['wins']
                    }
                    
                    logger.info(f"  {pattern}: {win_rate:.1f}% win rate ({score} points)")
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Failed to get pattern performance: {e}")
            return {}
    
    def optimize_confluence_threshold(self) -> int:
        """Find optimal minimum confluence score"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = """
                    SELECT 
                        bs.confluence_score,
                        so.result
                    FROM backtest_signals bs
                    JOIN signal_outcomes so ON bs.id = so.signal_id
                """
                
                rows = conn.execute(query).fetchall()
                
                best_threshold = 50
                best_score = 0
                
                # Test thresholds from 50 to 85
                for threshold in range(50, 86, 5):
                    signals_above = [r for r in rows if r['confluence_score'] >= threshold]
                    
                    if not signals_above:
                        continue
                    
                    wins = sum(1 for s in signals_above if s['result'] == 'WIN')
                    win_rate = wins / len(signals_above) * 100
                    
                    # Prefer higher win rate, but with minimum signal count
                    if len(signals_above) >= 20:
                        score = win_rate * 0.7 + (len(signals_above) / len(rows)) * 30
                        
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
                        
                        logger.info(f"  Threshold {threshold}: {win_rate:.1f}% ({len(signals_above)} signals)")
                
                logger.info(f"  ✅ Optimal threshold: {best_threshold}")
                return best_threshold
                
        except Exception as e:
            logger.error(f"❌ Failed to optimize threshold: {e}")
            return 72  # Default
    
    def get_total_signals(self) -> int:
        """Get total number of signals in backtest"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM signal_outcomes")
                return cursor.fetchone()[0]
        except:
            return 0
    
    def generate(self) -> bool:
        """Generate config from real data"""
        logger.info("=" * 70)
        logger.info("GENERATING CONFIG FROM REAL BACKTEST DATA")
        logger.info("=" * 70)
        
        # Step 1: Verify data exists
        if not self.check_backtest_data_exists():
            logger.error("\n❌ No backtest data found!")
            logger.error("   Run: python core/run_backtest.py --full")
            return False
        
        # Step 2: Get accuracy by score
        logger.info("\n📊 Extracting accuracy by confluence score...")
        accuracy_by_score = self.get_accuracy_by_score()
        
        if not accuracy_by_score:
            logger.error("❌ Failed to get accuracy data")
            return False
        
        # Step 3: Get pattern performance
        logger.info("\n📊 Extracting pattern performance...")
        pattern_perf = self.get_pattern_performance()
        
        # Step 4: Optimize threshold
        logger.info("\n📊 Finding optimal confluence threshold...")
        optimal_threshold = self.optimize_confluence_threshold()
        
        # Step 5: Build config
        self.config = {
            "metadata": {
                "version": "2.0-REAL-DATA",
                "generated": datetime.now().isoformat(),
                "note": "This config contains REAL data from backtesting, NOT guesses"
            },
            "based_on": {
                "total_signals": self.get_total_signals(),
                "backtest_period": "30 days",
                "database": "data/backtest.db"
            },
            "confluence_threshold": {
                "optimal_minimum": optimal_threshold,
                "note": f"Score must be >= {optimal_threshold} to generate signal"
            },
            "accuracy_by_score": {
                range_name: data['win_rate']
                for range_name, data in accuracy_by_score.items()
            },
            "accuracy_details": accuracy_by_score,
            "pattern_scores": {
                pattern: data['score']
                for pattern, data in pattern_perf.items()
            },
            "pattern_details": {
                pattern: {
                    'win_rate': data['win_rate'],
                    'score': data['score'],
                    'sample_size': data['signals']
                }
                for pattern, data in pattern_perf.items()
            }
        }
        
        # Step 6: Save config
        config_path = Path("config/optimized_config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ CONFIG GENERATED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"\n📝 Config saved to: {config_path}")
        logger.info(f"📊 Based on {self.config['based_on']['total_signals']} tested signals")
        logger.info(f"🎯 Optimal threshold: {optimal_threshold}")
        logger.info(f"📈 Accuracy ranges: {', '.join(accuracy_by_score.keys())}")
        logger.info(f"🎨 Patterns analyzed: {len(pattern_perf)}")
        logger.info("\n" + "=" * 70)
        
        return True

if __name__ == "__main__":
    generator = RealConfigGenerator()
    
    if generator.generate():
        logger.info("\n✅ Next steps:")
        logger.info("   1. Verify config was created: cat config/optimized_config.json")
        logger.info("   2. Restart the application to use new config")
        logger.info("   3. Run tests to verify accuracy values are real")
    else:
        logger.error("\n❌ Config generation failed!")
        exit(1)
