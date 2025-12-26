#!/usr/bin/env python3
"""
COMPLETE SYSTEM STATUS REPORT
Diagnoses exactly what's working, what's not, and why
"""

import json
import sqlite3
from pathlib import Path
import sys

class SystemDiagnostic:
    """Complete system diagnostic"""
    
    def __init__(self):
        self.results = {}
    
    def check_config_file(self):
        """Check config file status"""
        print("\n" + "="*70)
        print("CHECK 1: CONFIG FILE")
        print("="*70)
        
        config_path = Path("config/optimized_config.json")
        
        if not config_path.exists():
            print("❌ Config file not found")
            self.results['config_exists'] = False
            return
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            print("✅ Config file exists and is valid JSON")
            
            # Check content
            if 'confluence_thresholds' in config:
                ranges = config['confluence_thresholds'].get('ranges', {})
                signal_count = sum(r.get('signals_count', 0) for r in ranges.values())
                print(f"✅ Config claims {signal_count} total signals from backtesting")
                self.results['config_signal_count'] = signal_count
            
            if 'backtesting_metadata' in config:
                metadata = config['backtesting_metadata']
                print(f"✅ Metadata present: {metadata.get('created', 'N/A')}")
                print(f"   Description: {metadata.get('description', 'N/A')}")
            
            self.results['config_exists'] = True
            
        except Exception as e:
            print(f"❌ Config file invalid: {e}")
            self.results['config_exists'] = False
    
    def check_database(self):
        """Check backtest database"""
        print("\n" + "="*70)
        print("CHECK 2: BACKTEST DATABASE")
        print("="*70)
        
        db_path = Path("data/backtest.db")
        
        if not db_path.exists():
            print("❌ Database does not exist: data/backtest.db")
            print("   This is the ROOT CAUSE of the problem!")
            print("   Run: python core/run_backtest.py --full --symbol XRPUSDT")
            self.results['db_exists'] = False
            return
        
        print("✅ Database exists")
        
        try:
            with sqlite3.connect("data/backtest.db") as conn:
                cursor = conn.cursor()
                
                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [t[0] for t in cursor.fetchall()]
                print(f"✅ Database has {len(tables)} tables: {', '.join(tables)}")
                
                # Check signal data
                cursor.execute("SELECT COUNT(*) FROM backtest_signals")
                signal_count = cursor.fetchone()[0]
                print(f"✅ backtest_signals table: {signal_count} signals")
                
                # Check outcomes
                cursor.execute("SELECT COUNT(*) FROM signal_outcomes")
                outcome_count = cursor.fetchone()[0]
                print(f"✅ signal_outcomes table: {outcome_count} outcomes")
                
                if signal_count == 0 or outcome_count == 0:
                    print("❌ Database is empty! Run backtesting first.")
                    self.results['db_has_data'] = False
                    return
                
                # Show sample accuracy data
                cursor.execute("""
                    SELECT bs.confluence_score, so.result, COUNT(*) as count
                    FROM backtest_signals bs
                    JOIN signal_outcomes so ON bs.id = so.signal_id
                    GROUP BY bs.confluence_score, so.result
                    ORDER BY bs.confluence_score DESC
                    LIMIT 10
                """)
                
                print(f"✅ Sample signal outcomes:")
                for score, result, count in cursor.fetchall():
                    print(f"   Score {score:5.1f} → {result:7} ({count:3} signals)")
                
                self.results['db_exists'] = True
                self.results['db_has_data'] = True
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            self.results['db_exists'] = False
    
    def check_code(self):
        """Check if code is correctly set up"""
        print("\n" + "="*70)
        print("CHECK 3: CODE IMPLEMENTATION")
        print("="*70)
        
        code_file = Path("core/enhanced_crypto_dashboard.py")
        
        try:
            with open(code_file, encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check 1: Crash on missing data
            if 'Cannot load accuracy data' in content and 'RuntimeError' in content:
                print("✅ Code crashes on missing data (good!)")
                self.results['crashes_on_error'] = True
            else:
                print("❌ Code might have fallbacks to fake numbers")
                self.results['crashes_on_error'] = False
            
            # Check 2: Live tracker starts
            if '.start()' in content and 'live_signal_tracker' in content:
                print("✅ Code calls live_signal_tracker.start()")
                self.results['tracker_starts'] = True
            else:
                print("❌ Live tracker not called")
                self.results['tracker_starts'] = False
            
            # Check 3: No fake timeline generator
            if 'create_realistic_timeline' not in content:
                print("✅ Fake timeline function is removed")
                self.results['no_fake_timeline'] = True
            else:
                print("⚠️  Fake timeline function still exists")
                self.results['no_fake_timeline'] = False
            
            # Check 4: Has estimate accuracy method
            if 'def _estimate_accuracy' in content:
                print("✅ _estimate_accuracy() method exists")
                self.results['has_accuracy_method'] = True
            else:
                print("❌ _estimate_accuracy() missing")
                self.results['has_accuracy_method'] = False
            
        except Exception as e:
            print(f"❌ Error reading code: {e}")
    
    def check_config_vs_database(self):
        """Verify config values match database"""
        print("\n" + "="*70)
        print("CHECK 4: CONFIG vs DATABASE CONSISTENCY")
        print("="*70)
        
        if not Path("config/optimized_config.json").exists():
            print("❌ Config file not found")
            return
        
        if not Path("data/backtest.db").exists():
            print("❌ Database not found - cannot compare")
            return
        
        try:
            with open("config/optimized_config.json") as f:
                config = json.load(f)
            
            config_signals = sum(
                r.get('signals_count', 0) 
                for r in config.get('confluence_thresholds', {}).get('ranges', {}).values()
            )
            
            with sqlite3.connect("data/backtest.db") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM signal_outcomes")
                db_signals = cursor.fetchone()[0]
            
            print(f"Config claims: {config_signals} signals")
            print(f"Database has: {db_signals} signals")
            
            if config_signals == db_signals:
                print("✅ MATCH - Config was generated from this database")
                self.results['config_from_db'] = True
            elif db_signals == 0:
                print("❌ Database is empty - config is unverified")
                self.results['config_from_db'] = False
            elif config_signals == 0:
                print("❌ Config has no data")
                self.results['config_from_db'] = False
            else:
                print(f"❌ MISMATCH - Config is NOT from current database")
                print(f"   Config signals: {config_signals}")
                print(f"   DB signals: {db_signals}")
                self.results['config_from_db'] = False
            
        except Exception as e:
            print(f"❌ Error comparing: {e}")
    
    def print_summary(self):
        """Print executive summary"""
        print("\n" + "="*70)
        print("SYSTEM STATUS SUMMARY")
        print("="*70)
        
        if not self.results.get('db_exists'):
            print("\n🔴 CRITICAL ISSUE: NO BACKTESTING DATABASE")
            print("   Status: BROKEN - System cannot work")
            print("   Reason: data/backtest.db doesn't exist")
            print("   Fix: python core/run_backtest.py --full --symbol XRPUSDT")
            return
        
        if not self.results.get('db_has_data'):
            print("\n🔴 CRITICAL ISSUE: DATABASE IS EMPTY")
            print("   Status: BROKEN - Backtesting didn't complete")
            print("   Reason: signal_outcomes table has no data")
            print("   Fix: python core/run_backtest.py --full --symbol XRPUSDT")
            return
        
        if not self.results.get('config_from_db'):
            print("\n🟡 WARNING: CONFIG NOT FROM DATABASE")
            print("   Status: WORKING BUT UNVERIFIED")
            print("   Reason: Config values don't match database")
            print("   Fix: python scripts/generate_real_config.py")
        elif self.results.get('crashes_on_error') and self.results.get('tracker_starts'):
            print("\n🟢 GOOD STATUS: System appears correct")
            print("   ✅ Database exists with real backtesting data")
            print("   ✅ Config generated from database")
            print("   ✅ Code configured for error handling")
            print("   ✅ Live tracker initialization looks good")
        
        # What's working
        print("\nFEATURES:")
        print(f"   {'✅' if self.results.get('crashes_on_error') else '❌'} Crashes on missing data")
        print(f"   {'✅' if self.results.get('tracker_starts') else '❌'} Live tracker initialization")
        print(f"   {'✅' if self.results.get('no_fake_timeline') else '❌'} Removed fake timeline")
        print(f"   {'✅' if self.results.get('has_accuracy_method') else '❌'} Accuracy method exists")
    
    def run(self):
        """Run all diagnostics"""
        print("\n")
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                      SYSTEM DIAGNOSTIC REPORT                             ║")
        print("║                   Checking integration implementation                      ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        
        self.check_config_file()
        self.check_database()
        self.check_code()
        self.check_config_vs_database()
        self.print_summary()
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        
        if not self.results.get('db_exists'):
            print("\nRun backtesting (takes 20-45 minutes):")
            print("  python core/run_backtest.py --full --symbol XRPUSDT")
        elif not self.results.get('config_from_db'):
            print("\nGenerate config from database:")
            print("  python scripts/generate_real_config.py")
        else:
            print("\nSystem looks good! Run application:")
            print("  python run.py")
        
        print()

if __name__ == "__main__":
    diagnostic = SystemDiagnostic()
    diagnostic.run()
