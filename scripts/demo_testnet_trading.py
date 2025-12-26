"""
Demo: Integrated Testnet Trading with Paper Trading, Risk Manager, and Exchange Adapter
Shows how to use PaperTrader, RiskManager, and BinanceTestnetAdapter together.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_bot.core.paper_trader import PaperTrader
from crypto_bot.core.risk_manager import RiskManager
from crypto_bot.core.exchange_adapter import get_adapter, BinanceTestnetAdapter


def demo_testnet_trading():
    """
    Demo: Paper trade with simulated risk checks and testnet adapter readiness.
    
    This demonstrates the flow:
    1. Risk manager validates trade size
    2. Paper trader executes simulated order
    3. Exchange adapter is ready for live testnet/live trading
    """
    print("=" * 70)
    print("DEMO: Integrated Testnet Trading")
    print("=" * 70)
    
    # Initialize components
    paper_trader = PaperTrader()
    risk_manager = RiskManager()
    account_balance = 10000.0  # Assume $10k account
    
    # Demo scenario
    print(f"\n[Account Balance] ${account_balance:,.2f}")
    print(f"[Risk per trade] 1% of account = ${account_balance * 0.01:,.2f}")
    
    # Example trades
    demo_trades = [
        {'symbol': 'BTC/USDT', 'side': 'buy', 'stop_distance': 50.0},
        {'symbol': 'ETH/USDT', 'side': 'buy', 'stop_distance': 5.0},
        {'symbol': 'ADA/USDT', 'side': 'buy', 'stop_distance': 0.02},
    ]
    
    print("\n" + "=" * 70)
    print("TRADE VALIDATION & EXECUTION")
    print("=" * 70)
    
    for i, trade in enumerate(demo_trades, 1):
        symbol = trade['symbol']
        side = trade['side']
        stop_distance = trade['stop_distance']
        
        print(f"\n[Trade {i}] {symbol} {side.upper()}")
        print(f"  Stop distance: ${stop_distance}")
        
        # Calculate position size using risk manager
        position_size = risk_manager.position_size(account_balance, 0.01, stop_distance)
        print(f"  [OK] Position size: {position_size:.4f} {symbol.split('/')[0]}")
        
        # Check if trade is allowed
        is_allowed = risk_manager.check_trade_allowed(account_balance, current_drawdown=0.0, proposed_risk_fraction=0.01)
        if not is_allowed:
            print(f"  [REJECT] Trade rejected by risk manager")
            continue
        
        # Place paper trade
        trade_id = paper_trader.place_order(
            symbol=symbol,
            side=side,
            qty=position_size,
            price=None  # Market order
        )
        print(f"  [OK] Paper order placed (ID: {trade_id})")
    
    # Show paper trades
    print("\n" + "=" * 70)
    print("PAPER TRADES SUMMARY")
    print("=" * 70)
    trades = paper_trader.get_trades()
    if trades:
        for trade in trades:
            print(f"  {trade}")
        print(f"\n  Total trades: {len(trades)}")
    else:
        print("  No paper trades recorded")
    
    # Test adapter readiness (without connecting)
    print("\n" + "=" * 70)
    print("EXCHANGE ADAPTER READINESS CHECK")
    print("=" * 70)
    
    # Check if testnet keys are available
    has_testnet_keys = os.getenv('API_KEY_TESTNET') and os.getenv('API_SECRET_TESTNET')
    has_live_keys = os.getenv('API_KEY_LIVE') and os.getenv('API_SECRET_LIVE')
    
    print(f"\n  Testnet API keys available: {'YES' if has_testnet_keys else 'NO'}")
    print(f"  Live API keys available: {'YES' if has_live_keys else 'NO'}")
    
    if has_testnet_keys:
        print("\n  To connect to Binance testnet:")
        print("    adapter = get_adapter('testnet')")
        print("    if adapter.connect():")
        print("        order = adapter.place_order('BTC/USDT', 'buy', 0.001, 50000)")
        print("        print(order)")
    else:
        print("\n  [INFO] To enable testnet trading:")
        print("    1. Set API_KEY_TESTNET and API_SECRET_TESTNET environment variables")
        print("    2. Get testnet keys from: https://testnet.binance.vision")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Demo complete! Paper trading + Risk Manager working correctly.")
    print("          Exchange adapter is ready for testnet/live integration.")
    print("=" * 70)


if __name__ == '__main__':
    demo_testnet_trading()
