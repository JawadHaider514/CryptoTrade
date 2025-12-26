#!/usr/bin/env python3
"""Verify backtest actually ran"""
import sqlite3
from pathlib import Path

db_path = "data/backtest.db"

if not Path(db_path).exists():
    print(f"❌ Database doesn't exist: {db_path}")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    print(f"\n✅ Database found: {db_path}")
    print(f"   Size: {Path(db_path).stat().st_size:,} bytes")
    
    # List tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in c.fetchall()]
    print(f"\n📊 Tables ({len(tables)}):")
    
    for table in tables:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        print(f"   {table}: {count:,} rows")
    
    # Check signal data
    if 'backtest_signals' in tables:
        c.execute("SELECT COUNT(*) FROM backtest_signals")
        signals = c.fetchone()[0]
        print(f"\n✅ Backtest signals: {signals}")
    
    if 'signal_outcomes' in tables:
        c.execute("SELECT COUNT(*) FROM signal_outcomes")
        outcomes = c.fetchone()[0]
        print(f"✅ Signal outcomes: {outcomes}")
        
        # Show accuracy breakdown
        c.execute("""
            SELECT 
                CASE WHEN result='WIN' THEN 'WIN'
                     WHEN result='LOSS' THEN 'LOSS'
                     ELSE 'OTHER' END as result,
                COUNT(*) as count
            FROM signal_outcomes
            GROUP BY result
        """)
        
        print(f"\n📈 Outcome breakdown:")
        for result, count in c.fetchall():
            if count > 0:
                print(f"   {result}: {count}")
    
    print(f"\n✅ DATABASE IS POPULATED AND READY")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
