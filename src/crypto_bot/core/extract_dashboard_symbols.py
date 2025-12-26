#!/usr/bin/env python3
"""Extract symbols configured for the dashboard.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'crypto_data.db')


def get_dashboard_symbols():
    """
    Dashboard configuration se saare symbols nikalo.
    First try: pull from `data/crypto_data.db` klines table.
    Fallback: return a small default list.
    """
    db_path = os.path.abspath(DB_PATH)
    symbols = []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM klines")
        symbols = [row[0] for row in cur.fetchall()]
        conn.close()
    except Exception:
        # Fallback hardcoded list
        symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']

    return symbols


if __name__ == '__main__':
    syms = get_dashboard_symbols()
    print(f"Total coins to download: {len(syms)}")
    for s in syms[:50]:
        print(s)
