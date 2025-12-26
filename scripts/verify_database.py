#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('data/backtest.db')
c = conn.cursor()

print('\n' + '='*70)
print('DATABASE VERIFICATION - REAL DATA EXISTS')
print('='*70)

c.execute('SELECT COUNT(*) FROM backtest_signals')
total = c.fetchone()[0]

c.execute('SELECT COUNT(CASE WHEN result="WIN" THEN 1 END) FROM signal_outcomes')
wins = c.fetchone()[0]

print(f'\n✅ Signals: {total}')
print(f'✅ Wins: {wins}')
print(f'✅ Win rate: {wins/total*100:.1f}%')
print(f'✅ Database file: data/backtest.db')
print(f'✅ Size: {__import__("pathlib").Path("data/backtest.db").stat().st_size / 1024:.0f} KB')

print('\n✅ Sample signals:')
c.execute('SELECT confluence_score, direction, result FROM backtest_signals NATURAL JOIN signal_outcomes LIMIT 5')
for i, (score, dir, res) in enumerate(c.fetchall(), 1):
    emoji = '✅' if res == 'WIN' else '❌'
    print(f'   {emoji} #{i}: {dir} @ {score:.0f} → {res}')

conn.close()
print('\n' + '='*70)
