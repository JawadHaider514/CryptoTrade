#!/usr/bin/env python3
"""
OPTIMIZE CONFLUENCE SCORE THRESHOLD
Tests different score thresholds against backtesting database to find
the optimal value that maximizes profit per signal
"""

import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ThresholdResult:
    """Result of testing a single threshold"""
    threshold: int
    total_signals: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_signal: float
    profit_factor: float

class ThresholdOptimizer:
    """Find optimal confluence threshold"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        self.db_path = db_path
    
    def test_threshold(self, threshold: int) -> ThresholdResult | None:
        """Test a single threshold value"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get signals above threshold
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN so.result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN so.result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN so.result = 'TIMEOUT' THEN 1 ELSE 0 END) as timeouts,
                        SUM(so.profit_loss) as total_pnl,
                        AVG(so.profit_loss) as avg_pnl
                    FROM backtest_signals bs
                    LEFT JOIN signal_outcomes so ON bs.id = so.signal_id
                    WHERE bs.confluence_score >= ?
                """, (threshold,))
                
                row = cursor.fetchone()
                if not row or row[0] == 0:
                    return None
                
                total, wins, losses, timeouts, total_pnl, avg_pnl = row
                wins = wins or 0
                losses = losses or 0
                timeouts = timeouts or 0
                total_pnl = total_pnl or 0.0
                avg_pnl = avg_pnl or 0.0
                
                win_rate = (wins / total * 100) if total > 0 else 0
                
                # Calculate profit factor (wins / losses)
                profit_factor = 1.0
                if losses > 0:
                    winning_pnl = cursor.execute(
                        "SELECT SUM(so.profit_loss) FROM backtest_signals bs "
                        "LEFT JOIN signal_outcomes so ON bs.id = so.signal_id "
                        "WHERE bs.confluence_score >= ? AND so.result = 'WIN'",
                        (threshold,)
                    ).fetchone()[0] or 0
                    
                    losing_pnl = abs(cursor.execute(
                        "SELECT SUM(so.profit_loss) FROM backtest_signals bs "
                        "LEFT JOIN signal_outcomes so ON bs.id = so.signal_id "
                        "WHERE bs.confluence_score >= ? AND so.result = 'LOSS'",
                        (threshold,)
                    ).fetchone()[0] or 0)
                    
                    if losing_pnl > 0:
                        profit_factor = winning_pnl / losing_pnl
                    else:
                        profit_factor = float('inf')
                
                return ThresholdResult(
                    threshold=threshold,
                    total_signals=total,
                    wins=wins,
                    losses=losses,
                    timeouts=timeouts,
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    avg_pnl_per_signal=avg_pnl,
                    profit_factor=profit_factor
                )
                
        except Exception as e:
            logger.error(f"Error testing threshold {threshold}: {e}")
            return None
    
    def run(self):
        """Test all thresholds and find optimal"""
        
        logger.info("\n" + "="*70)
        logger.info("CONFLUENCE THRESHOLD OPTIMIZATION")
        logger.info("="*70)
        
        # Check database
        if not Path(self.db_path).exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            logger.error("   Run: python core/run_backtest.py --full --symbol XRPUSDT")
            return False
        
        # Test range of thresholds
        logger.info(f"\n📊 Testing thresholds from 50 to 85...")
        
        results = []
        for threshold in range(50, 90, 5):
            result = self.test_threshold(threshold)
            if result:
                results.append(result)
                logger.info(
                    f"   Threshold {threshold:2d}: "
                    f"{result.total_signals:3d} signals, "
                    f"{result.win_rate:5.1f}% win, "
                    f"${result.avg_pnl_per_signal:7.2f}/signal"
                )
        
        if not results:
            logger.error("❌ No valid results - database might be empty")
            return False
        
        # Find optimal by different criteria
        logger.info(f"\n🎯 OPTIMIZATION RESULTS:")
        
        # Highest win rate
        best_wr = max(results, key=lambda r: r.win_rate)
        logger.info(f"   Best win rate: {best_wr.threshold} → {best_wr.win_rate:.1f}%")
        
        # Best profit per signal
        best_pnl = max(results, key=lambda r: r.avg_pnl_per_signal)
        logger.info(f"   Best PnL/signal: {best_pnl.threshold} → ${best_pnl.avg_pnl_per_signal:.2f}")
        
        # Best profit factor
        best_pf = max(results, key=lambda r: r.profit_factor)
        logger.info(f"   Best profit factor: {best_pf.threshold} → {best_pf.profit_factor:.2f}")
        
        # Balanced: win rate × signals (more signals = better diversification)
        best_balanced = max(results, key=lambda r: r.win_rate * (r.total_signals / max(r.total_signals for r in results)))
        logger.info(f"   Balanced (win_rate × signal_count): {best_balanced.threshold}")
        
        # Recommendation
        logger.info(f"\n✅ RECOMMENDATION:")
        logger.info(f"   Update confluence threshold to: {best_balanced.threshold}")
        logger.info(f"   Expected: {best_balanced.win_rate:.1f}% win rate on {best_balanced.total_signals} signals")
        logger.info(f"   Expected: ${best_balanced.avg_pnl_per_signal:.2f} profit per signal")
        
        logger.info(f"\n📝 Update config/optimized_config.json:")
        logger.info(f'   "optimal_minimum": {best_balanced.threshold}')
        
        return True

if __name__ == "__main__":
    optimizer = ThresholdOptimizer()
    success = optimizer.run()
    exit(0 if success else 1)
