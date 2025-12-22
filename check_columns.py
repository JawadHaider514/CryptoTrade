#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('data/backtest.db')
c = conn.cursor()

print("\n=== BACKTEST_SIGNALS COLUMNS ===")
c.execute("PRAGMA table_info(backtest_signals)")
for col in c.fetchall():
    print(f"  {col[1]} ({col[2]})")

print("\n=== SIGNAL_OUTCOMES COLUMNS ===")
c.execute("PRAGMA table_info(signal_outcomes)")
for col in c.fetchall():
    print(f"  {col[1]} ({col[2]})")

print("\n=== SAMPLE DATA ===")
c.execute("SELECT * FROM backtest_signals LIMIT 1")
cols = [desc[0] for desc in c.description]
row = c.fetchone()
if row:
    for col, val in zip(cols, row):
        print(f"  {col}: {val}")

conn.close()
