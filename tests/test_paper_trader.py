import os
import sqlite3
from crypto_bot.core.paper_trader import PaperTrader

DB = 'data/test_paper.db'


def setup_module():
    try:
        os.remove(DB)
    except Exception:
        pass


def test_place_order_and_persistence():
    t = PaperTrader(db_path=DB)
    o = t.place_order('TEST/USDT', 'BUY', 1.0, price=1.0)
    assert o['status'] == 'FILLED'

    # Check DB entry
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM paper_trades')
    cnt = cur.fetchone()[0]
    conn.close()
    assert cnt >= 1
