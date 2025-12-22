#!/usr/bin/env python3
"""
OPTIMIZE PATTERN SCORES
Calculate actual win rates for each candlestick pattern from
backtesting data and assign point values based on real performance
"""

import sqlite3
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PatternOptimizer:
    """Calculate optimal pattern scores from backtesting"""
    
    def __init__(self, db_path: str = "data/backtest.db", config_path: str = "config/optimized_config.json"):
        self.db_path = db_path
        self.config_path = config_path
    
    def analyze_patterns(self) -> dict:
        """Analyze win rates for each pattern"""
        
        logger.info(f"\n📊 Analyzing pattern performance...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all unique patterns
                cursor.execute("SELECT DISTINCT patterns FROM backtest_signals WHERE patterns IS NOT NULL")
                patterns_raw = cursor.fetchall()
                
                all_patterns = set()
                for pattern_str in patterns_raw:
                    if pattern_str[0]:
                        # Pattern might be comma-separated
                        for p in pattern_str[0].split(','):
                            all_patterns.add(p.strip())
                
                pattern_stats = {}
                
                for pattern in sorted(all_patterns):
                    # Count signals with this pattern
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN so.result = 'WIN' THEN 1 ELSE 0 END) as wins,
                            SUM(CASE WHEN so.result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                            AVG(so.profit_loss) as avg_pnl
                        FROM backtest_signals bs
                        LEFT JOIN signal_outcomes so ON bs.id = so.signal_id
                        WHERE patterns LIKE ?
                    """, (f'%{pattern}%',))
                    
                    row = cursor.fetchone()
                    if not row or row[0] < 5:  # Need minimum 5 signals for statistical significance
                        continue
                    
                    total, wins, losses, avg_pnl = row
                    wins = wins or 0
                    losses = losses or 0
                    avg_pnl = avg_pnl or 0.0
                    
                    win_rate = (wins / total * 100) if total > 0 else 0
                    
                    # Calculate score: base 10 points + performance bonus
                    # High win rate (70%+) = 18 points
                    # Medium win rate (60-70%) = 14 points
                    # Low win rate (50-60%) = 10 points
                    # Very low (<50%) = 5 points
                    
                    if win_rate >= 70:
                        score = 18
                    elif win_rate >= 60:
                        score = 14
                    elif win_rate >= 50:
                        score = 10
                    else:
                        score = 5
                    
                    pattern_stats[pattern] = {
                        'total_signals': total,
                        'wins': wins,
                        'losses': losses,
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'calculated_score': score
                    }
                
                return pattern_stats
                
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {}
    
    def update_config(self, pattern_stats: dict) -> bool:
        """Update config file with new pattern scores"""
        
        if not Path(self.config_path).exists():
            logger.error(f"❌ Config not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Update pattern scores
            if 'pattern_scores' not in config:
                config['pattern_scores'] = {'patterns': {}}
            
            patterns_section = config['pattern_scores'].get('patterns', {})
            
            for pattern, stats in pattern_stats.items():
                patterns_section[pattern] = {
                    'points': stats['calculated_score'],
                    'win_rate': stats['win_rate'],
                    'signals': stats['total_signals'],
                    'avg_pnl': round(stats['avg_pnl'], 2),
                    'description': f"Based on {stats['total_signals']} test signals"
                }
            
            config['pattern_scores']['patterns'] = patterns_section
            config['pattern_scores']['last_updated'] = Path(self.config_path).stat().st_mtime
            
            # Write back
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Config updated: {len(pattern_stats)} patterns")
            return True
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
    
    def run(self):
        """Run pattern optimization"""
        
        logger.info("\n" + "="*70)
        logger.info("PATTERN SCORE OPTIMIZATION")
        logger.info("="*70)
        
        # Check database
        if not Path(self.db_path).exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            logger.error("   Run: python core/run_backtest.py --full --symbol XRPUSDT")
            return False
        
        # Analyze patterns
        pattern_stats = self.analyze_patterns()
        
        if not pattern_stats:
            logger.warning("⚠️  No patterns found with sufficient signals")
            return False
        
        # Show results
        logger.info(f"\n📊 PATTERN ANALYSIS RESULTS:")
        logger.info(f"   Pattern          | Signals | Win Rate | Score")
        logger.info(f"   ─────────────────┼─────────┼──────────┼──────")
        
        for pattern, stats in sorted(pattern_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True):
            print(f"   {pattern:16} | {stats['total_signals']:7} | {stats['win_rate']:7.1f}% | {stats['calculated_score']:5}")
        
        # Update config
        if self.update_config(pattern_stats):
            logger.info(f"\n✅ OPTIMIZATION COMPLETE")
            logger.info(f"   Analyzed: {len(pattern_stats)} patterns")
            logger.info(f"   Updated: {self.config_path}")
        else:
            logger.error(f"❌ Failed to update config")
            return False
        
        return True

if __name__ == "__main__":
    optimizer = PatternOptimizer()
    success = optimizer.run()
    exit(0 if success else 1)
