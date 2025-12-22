#!/usr/bin/env python3
"""
Demo script to place a paper trade using the PaperTrader and RiskManager.
"""
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from core.paper_trader import PaperTrader
from core.risk_manager import RiskManager

if __name__ == '__main__':
    pt = PaperTrader()
    rm = RiskManager()

    balance = 1000.0
    stop_distance = 50.0  # dollars
    size = rm.position_size(balance, None, stop_distance)

    print(f"Computed position size: {size:.6f} units")

    if rm.check_trade_allowed(balance, 0.02, 0.01) and size > 0:
        order = pt.place_order('BTC/USDT', 'BUY', round(size, 6), price=50000.0)
        print('Placed demo paper order:', order)
    else:
        print('Trade not allowed by risk manager')
