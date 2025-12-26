#!/usr/bin/env python3
"""
Paper trading simulator: used to test live strategies without real funds.
Stores simulated trades in `paper_trades` table for auditing.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

DB_PATH = Path(__file__).parent.parent / 'data' / 'crypto_historical.db'


class PaperTrader:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DB_PATH)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id TEXT PRIMARY KEY,
                timestamp INTEGER,
                symbol TEXT,
                side TEXT,
                qty REAL,
                price REAL,
                filled_qty REAL,
                status TEXT,
                meta TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def place_order(self, symbol: str, side: str, qty: float, price: Optional[float] = None,
                    order_type: str = 'market', meta: Optional[Dict[str, Any]] = None) -> Dict:
        """Place a simulated order and return order details (immediately filled for simplicity)."""
        order_id = uuid.uuid4().hex
        ts = self._now_ms()

        # Simulate immediate fill
        filled_qty = qty
        avg_price = float(price) if price is not None else 0.0
        status = 'FILLED'

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO paper_trades
            (id, timestamp, symbol, side, qty, price, filled_qty, status, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (order_id, ts, symbol, side, qty, avg_price, filled_qty, status, str(meta or {})))
        conn.commit()
        conn.close()

        return {
            'id': order_id,
            'timestamp': ts,
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': avg_price,
            'filled_qty': filled_qty,
            'status': status
        }

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if symbol:
            cur.execute("SELECT id, timestamp, symbol, side, qty, price, filled_qty, status, meta FROM paper_trades WHERE symbol=? ORDER BY timestamp DESC LIMIT ?", (symbol, limit))
        else:
            cur.execute("SELECT id, timestamp, symbol, side, qty, price, filled_qty, status, meta FROM paper_trades ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        trades = []
        for r in rows:
            trades.append({
                'id': r[0], 'timestamp': r[1], 'symbol': r[2], 'side': r[3], 'qty': r[4], 'price': r[5], 'filled_qty': r[6], 'status': r[7], 'meta': r[8]
            })
        return trades


if __name__ == '__main__':
    t = PaperTrader()
    print('Placing demo paper order...')
    o = t.place_order('BTC/USDT', 'BUY', 0.001, price=50000.0)
    print('Order placed:', o)
    print('Recent trades:', t.get_trades('BTC/USDT', limit=5))
