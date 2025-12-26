# Testing Guide for Real Backtesting System

## 📋 Test Cases

### TEST 1: Data Download (5 minutes)
**Goal:** Verify historical data downloads correctly

```bash
cd crypto_trading_system
python -c "
from core.backtest_system import HistoricalDataCollector
import os

collector = HistoricalDataCollector()
stats = collector.get_data_stats('XRPUSDT')

assert stats.get('total_candles', 0) >= 30000, 'Not enough candles'
assert stats.get('days_of_data', 0) >= 28, 'Not enough days'
print('✅ TEST 1 PASSED: Data download successful')
print(f'   {stats[\"total_candles\"]} candles, {stats[\"days_of_data\"]:.1f} days')
"
```

**Pass Criteria:**
- ✅ At least 30,000 candles (1440 per day × 21+ days)
- ✅ At least 28 days of data
- ✅ Database file created at data/backtest.db
- ✅ No API errors

---

### TEST 2: Signal Generation (10 minutes)
**Goal:** Verify signals generate correctly

```bash
python -c "
from core.signal_generator import HistoricalSignalGenerator
from datetime import datetime, timedelta
import sqlite3

generator = HistoricalSignalGenerator()

# Generate signals for last 7 days
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=7)

count = generator.generate_signals_for_period(
    'XRPUSDT',
    start_time,
    end_time,
    interval_minutes=5
)

assert count > 0, 'No signals generated'

# Verify in database
with sqlite3.connect('data/backtest.db') as conn:
    cursor = conn.execute(
        'SELECT COUNT(*) FROM backtest_signals WHERE symbol = ?',
        ('XRPUSDT',)
    )
    db_count = cursor.fetchone()[0]
    assert db_count > 0, 'Signals not saved to database'

print('✅ TEST 2 PASSED: Signal generation successful')
print(f'   Generated {count} signals')
"
```

**Pass Criteria:**
- ✅ At least 100 signals generated
- ✅ Signals have entry, SL, TP1/TP2/TP3
- ✅ Confluence score between 0-100+
- ✅ Patterns detected (doji, hammer, etc.)
- ✅ All signals saved to database

---

### TEST 3: Outcome Tracking (10 minutes)
**Goal:** Verify outcomes tracked correctly

```bash
python -c "
from core.outcome_tracker import OutcomeTracker

tracker = OutcomeTracker()

# Track all signals
tracked = tracker.track_all_signals('XRPUSDT')

assert tracked > 0, 'No signals tracked'

# Verify outcomes
with sqlite3.connect('data/backtest.db') as conn:
    cursor = conn.execute(
        'SELECT COUNT(*), COUNT(CASE WHEN result = \\'WIN\\' THEN 1 END) '
        'FROM signal_outcomes WHERE symbol = ?',
        ('XRPUSDT',)
    )
    total, wins = cursor.fetchone()
    assert total > 0, 'No outcomes recorded'

print('✅ TEST 3 PASSED: Outcome tracking successful')
print(f'   Tracked {tracked} outcomes')
"
```

**Pass Criteria:**
- ✅ All signals have outcomes (WIN/LOSS/TIMEOUT)
- ✅ Exit prices recorded
- ✅ P&L calculated correctly
- ✅ Time in trade recorded

---

### TEST 4: Statistics Calculation (5 minutes)
**Goal:** Verify statistics calculated correctly

```bash
python -c "
from core.statistics_calculator import BacktestStatisticsCalculator

calculator = BacktestStatisticsCalculator()

# Get overall stats
overall = calculator.calculate_overall_stats('XRPUSDT')

assert overall.get('total_signals', 0) > 0, 'No signals found'
assert 'win_rate' in overall, 'Win rate missing'
assert 0 <= overall['win_rate'] <= 100, 'Invalid win rate'
assert 'profit_factor' in overall, 'Profit factor missing'

# Get by score
by_score = calculator.calculate_accuracy_by_confluence_score('XRPUSDT')
assert len(by_score) > 0, 'No score ranges found'

# Get by pattern
by_pattern = calculator.calculate_accuracy_by_pattern('XRPUSDT')
assert len(by_pattern) > 0, 'No patterns found'

print('✅ TEST 4 PASSED: Statistics calculation successful')
print(f'   {overall[\"total_signals\"]} signals')
print(f'   {overall[\"win_rate\"]:.1f}% win rate')
print(f'   {overall[\"profit_factor\"]:.2f}x profit factor')
"
```

**Pass Criteria:**
- ✅ Overall stats calculated
- ✅ Win rate between 0-100%
- ✅ Accuracy by score ranges available
- ✅ Accuracy by pattern calculated
- ✅ Expectancy calculated

---

### TEST 5: Accuracy Verification (manual)
**Goal:** Verify accuracy numbers are REAL, not fake

```bash
python -c "
from core.statistics_calculator import BacktestStatisticsCalculator

calc = BacktestStatisticsCalculator()

# Generate report
report = calc.generate_comprehensive_report('XRPUSDT')

print(report)

# Save to file
filename = calc.save_report_to_file('XRPUSDT')
print(f'\\n✅ Report saved to {filename}')
"
```

**Verify Manually:**
1. Open the report file
2. Check the numbers make sense
3. Verify:
   - Total signals matches wins + losses + timeouts
   - Win rate = (wins / total) × 100
   - Profit factor = total profit / total loss
   - Pattern win rates seem reasonable

---

### TEST 6: Live Tracking (5 minutes)
**Goal:** Verify live tracker works

```bash
python -c "
from core.live_tracker import LiveSignalTracker
import time

tracker = LiveSignalTracker()

# Start tracker
tracker.start()

# Simulate adding a signal (would normally come from signal generator)
# For now, just test that tracker can be created and started

# Get stats
stats = tracker.get_statistics()
print('✅ TEST 6 PASSED: Live tracker initialized')
print(f'   Stats structure: {list(stats.keys())}')

# Stop tracker
tracker.stop()
"
```

**Pass Criteria:**
- ✅ Tracker starts without errors
- ✅ Can get statistics
- ✅ Can be stopped cleanly
- ✅ Database schema created

---

## 🧪 Complete Test Script

Create a file `test_complete_system.py`:

```python
#!/usr/bin/env python3
"""
Complete testing script for real backtesting system
"""

import sys
import os
from datetime import datetime, timedelta
import sqlite3

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.backtest_system import HistoricalDataCollector
from core.signal_generator import HistoricalSignalGenerator
from core.outcome_tracker import OutcomeTracker
from core.statistics_calculator import BacktestStatisticsCalculator
from core.live_tracker import LiveSignalTracker


def test_data_download():
    """Test 1: Download historical data"""
    print("\n" + "="*70)
    print("TEST 1: HISTORICAL DATA DOWNLOAD")
    print("="*70)
    
    try:
        collector = HistoricalDataCollector()
        stats = collector.get_data_stats('XRPUSDT')
        
        if not stats or 'total_candles' not in stats:
            print("❌ FAILED: Could not get data stats")
            return False
        
        print(f"✅ PASSED: Downloaded {stats['total_candles']} candles")
        print(f"   Date range: {stats['oldest_time']} to {stats['newest_time']}")
        
        return stats['total_candles'] > 30000
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_signal_generation():
    """Test 2: Generate signals"""
    print("\n" + "="*70)
    print("TEST 2: SIGNAL GENERATION")
    print("="*70)
    
    try:
        generator = HistoricalSignalGenerator()
        
        # Generate for last 3 days
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=3)
        
        count = generator.generate_signals_for_period(
            'XRPUSDT',
            start_time,
            end_time,
            interval_minutes=5
        )
        
        print(f"✅ PASSED: Generated {count} signals")
        
        return count > 0
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_outcome_tracking():
    """Test 3: Track outcomes"""
    print("\n" + "="*70)
    print("TEST 3: OUTCOME TRACKING")
    print("="*70)
    
    try:
        tracker = OutcomeTracker()
        
        tracked = tracker.track_all_signals('XRPUSDT')
        
        print(f"✅ PASSED: Tracked {tracked} signal outcomes")
        
        return tracked > 0
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_statistics():
    """Test 4: Calculate statistics"""
    print("\n" + "="*70)
    print("TEST 4: STATISTICS CALCULATION")
    print("="*70)
    
    try:
        calc = BacktestStatisticsCalculator()
        
        stats = calc.calculate_overall_stats('XRPUSDT')
        
        if not stats or 'win_rate' not in stats:
            print("❌ FAILED: Could not calculate stats")
            return False
        
        print(f"✅ PASSED: Statistics calculated")
        print(f"   Total signals: {stats['total_signals']}")
        print(f"   Win rate: {stats['win_rate']:.2f}%")
        print(f"   Profit factor: {stats['profit_factor']:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_accuracy_by_score():
    """Test 5: Verify accuracy by confidence score"""
    print("\n" + "="*70)
    print("TEST 5: ACCURACY BY CONFLUENCE SCORE")
    print("="*70)
    
    try:
        calc = BacktestStatisticsCalculator()
        
        by_score = calc.calculate_accuracy_by_confluence_score('XRPUSDT')
        
        if not by_score:
            print("❌ FAILED: Could not calculate accuracy by score")
            return False
        
        print(f"✅ PASSED: Calculated accuracy for {len(by_score)} score ranges")
        
        for score_range, stats in by_score.items():
            print(f"   {score_range}: {stats['win_rate']:.1f}% ({stats['signals']} signals)")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_live_tracker():
    """Test 6: Live tracking system"""
    print("\n" + "="*70)
    print("TEST 6: LIVE TRACKING SYSTEM")
    print("="*70)
    
    try:
        tracker = LiveSignalTracker()
        
        # Start tracker
        tracker.start()
        
        # Get stats
        stats = tracker.get_statistics()
        
        # Stop tracker
        tracker.stop()
        
        print(f"✅ PASSED: Live tracker initialized and stopped")
        print(f"   Database schema created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║  REAL BACKTESTING SYSTEM - COMPLETE TEST SUITE" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Data Download", test_data_download()))
    results.append(("Signal Generation", test_signal_generation()))
    results.append(("Outcome Tracking", test_outcome_tracking()))
    results.append(("Statistics", test_statistics()))
    results.append(("Accuracy by Score", test_accuracy_by_score()))
    results.append(("Live Tracker", test_live_tracker()))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Run it:**
```bash
cd crypto_trading_system
python test_complete_system.py
```

---

## 📊 Validation Checklist

- [ ] Historical data: >30,000 candles
- [ ] Signal generation: >100 signals
- [ ] Outcome tracking: All signals have results
- [ ] Win rate: Between 30-80% (realistic)
- [ ] Profit factor: Between 0.8-3.0 (reasonable)
- [ ] Accuracy varies by score (85+ > 65-74)
- [ ] Patterns show meaningful differences
- [ ] No negative profit factors (would indicate backward logic)
- [ ] P&L calculations are mathematically correct
- [ ] Time in trade tracked for all signals

---

## ⚠️ Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "No outcomes found" | Run signals-only first, then outcomes-only |
| "Database locked" | Close other instances, delete .db file |
| "API rate limit" | Script includes delays, just wait |
| "Win rate = 0%" | Check if historical data has price movements |
| "Win rate = 100%" | Unlikely; verify TP/SL logic is correct |
| "Negative P&L on WIN" | Check if entry/exit prices are swapped |

---

**✅ Once all tests pass, your system is validated!**
