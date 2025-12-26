#!/usr/bin/env python3
"""
VERIFY REAL CONFIG DATA
Determines if config values are from REAL backtesting or manual guesses
"""

import json
from pathlib import Path

def check_config_source():
    """Check if config came from real backtesting"""
    
    config_path = Path("config/optimized_config.json")
    
    print("\n" + "="*70)
    print("CONFIG DATA VERIFICATION")
    print("="*70)
    
    if not config_path.exists():
        print("❌ No config file found")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Check 1: Does config have metadata about its source?
    print("\n1. SOURCE METADATA")
    print("-" * 70)
    
    metadata = config.get("backtesting_metadata", {})
    if metadata:
        print(f"   Description: {metadata.get('description', 'N/A')}")
        print(f"   Created: {metadata.get('created', 'N/A')}")
    else:
        print("   ⚠️  No metadata field - suspicious")
    
    # Check 2: Do the numbers add up mathematically?
    print("\n2. DATA INTEGRITY")
    print("-" * 70)
    
    ranges = config.get("confluence_thresholds", {}).get("ranges", {})
    
    total_signals = 0
    total_wins = 0
    total_losses = 0
    
    for range_name, data in ranges.items():
        signals = data.get("signals_count", 0)
        win_rate = data.get("win_rate", 0)
        loss_rate = data.get("loss_rate", 0)
        
        total_signals += signals
        total_wins += signals * (win_rate / 100)
        total_losses += signals * (loss_rate / 100)
        
        # Check consistency
        if abs(win_rate + loss_rate - 100.0) > 0.1:
            print(f"   ❌ {range_name}: win_rate + loss_rate ≠ 100% ({win_rate}% + {loss_rate}%)")
        else:
            print(f"   ✅ {range_name}: rates sum to 100% ({win_rate}% + {loss_rate}%)")
        
        # Check if numbers are suspiciously round (sign of manual entry)
        if win_rate % 1 == 0.5 or win_rate % 1 == 0.0:
            print(f"      🔍 {win_rate}% is suspiciously round/simple")
    
    # Check 3: Does database exist with matching data?
    print("\n3. DATABASE VERIFICATION")
    print("-" * 70)
    
    db_path = Path("data/backtest.db")
    if db_path.exists():
        print("   ✅ Database exists")
        try:
            import sqlite3
            with sqlite3.connect("data/backtest.db") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM backtest_signals")
                db_signals = cursor.fetchone()[0]
                
                if db_signals > 0:
                    print(f"   ✅ Database has {db_signals} signals")
                    print(f"   ✅ Config claims {total_signals} total signals")
                    
                    if db_signals == total_signals:
                        print(f"   ✅ NUMBERS MATCH - Config is from REAL backtesting!")
                    else:
                        print(f"   ❌ Mismatch! DB has {db_signals} but config sums to {total_signals}")
                else:
                    print("   ❌ Database is empty - no real backtesting data")
        except Exception as e:
            print(f"   ❌ Cannot read database: {e}")
    else:
        print("   ❌ NO DATABASE FOUND")
        print(f"      Config claims {total_signals} signals exist")
        print(f"      But there's NO data/backtest.db file!")
        print("      This likely means config values are MANUAL GUESSES, not real data")
    
    # Check 4: Comparison to baseline
    print("\n4. PLAUSIBILITY CHECK")
    print("-" * 70)
    
    accuracy_estimates = config.get("accuracy_estimates", {})
    by_score = accuracy_estimates.get("by_score", {})
    
    if by_score:
        print("   Accuracy by score:")
        for score_key, accuracy in by_score.items():
            print(f"      {score_key}: {accuracy}%")
        
        # Sanity check
        if by_score.get("85_plus", 0) < by_score.get("below_65", 0):
            print("   ⚠️  ERROR: Higher scores have LOWER accuracy - data is backwards!")
        else:
            print("   ✅ Accuracy improves with higher confidence scores (expected)")
    
    # Summary
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if not db_path.exists():
        print("\n⚠️  CRITICAL FINDING:")
        print("    The config file has numbers in it, but...")
        print("    NO data/backtest.db file exists")
        print("    This means the values are likely MANUAL GUESSES, not from real backtesting")
        print("\n✅ TO FIX THIS:")
        print("    1. Run backtesting: python core/run_backtest.py --full")
        print("    2. This will create data/backtest.db with real signal results")
        print("    3. Then regenerate config: python scripts/generate_real_config.py")
        print("\n    OR verify that backtest.db should exist elsewhere")
    else:
        print("\n✅ Database exists - config may be from real backtesting")
        print("   (Verify by running: python scripts/validate_integration.py)")

if __name__ == "__main__":
    check_config_source()
