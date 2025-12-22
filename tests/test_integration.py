"""
Integration test: Multi-symbol paper trading dry-run with audit logging
Tests the full flow: Risk Manager -> Paper Trader -> Audit Logger
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.paper_trader import PaperTrader
from core.risk_manager import RiskManager
from core.audit_logger import AuditLogger


def integration_test_paper_trading():
    """Full integration test of paper trading flow."""
    
    print("=" * 70)
    print("INTEGRATION TEST: Multi-Symbol Paper Trading with Audit Logging")
    print("=" * 70)
    
    # Initialize components
    paper_trader = PaperTrader()
    risk_manager = RiskManager()
    audit_logger = AuditLogger()
    
    account_balance = 50000.0  # Larger account for integration test
    max_drawdown = 0.0
    
    print(f"\n[SETUP] Account: ${account_balance:,.2f}")
    audit_logger.log_event(
        'TEST_START',
        f'Integration test started with ${account_balance:,.2f} account',
        context={'account_balance': account_balance}
    )
    
    # Multi-symbol test scenario
    test_signals = [
        {'symbol': 'BTC/USDT', 'signal': 'buy', 'confidence': 0.92, 'stop': 200.0},
        {'symbol': 'ETH/USDT', 'signal': 'buy', 'confidence': 0.88, 'stop': 20.0},
        {'symbol': 'BNB/USDT', 'signal': 'buy', 'confidence': 0.85, 'stop': 10.0},
        {'symbol': 'XRP/USDT', 'signal': 'buy', 'confidence': 0.78, 'stop': 0.05},
        {'symbol': 'ADA/USDT', 'signal': 'buy', 'confidence': 0.80, 'stop': 0.02},
    ]
    
    successful_trades = 0
    rejected_trades = 0
    
    print(f"\n[TRADING] Processing {len(test_signals)} signals...\n")
    
    for sig in test_signals:
        symbol = sig['symbol']
        confidence = sig['confidence']
        stop = sig['stop']
        
        print(f"[SIGNAL] {symbol} - Confidence: {confidence:.0%}")
        
        # Step 1: Risk check
        can_trade = risk_manager.check_trade_allowed(
            account_balance, 
            current_drawdown=max_drawdown,
            proposed_risk_fraction=0.01
        )
        
        if not can_trade:
            print(f"  [REJECT] Risk manager blocked trade")
            audit_logger.log_trade_decision(
                symbol=symbol,
                action='REJECTED',
                signal_strength=confidence,
                ml_confidence=confidence,
                risk_check_passed=False,
                position_size=0.0,
                reason='Risk manager check failed'
            )
            rejected_trades += 1
            continue
        
        # Step 2: Calculate position size
        position_size = risk_manager.position_size(account_balance, 0.01, stop)
        print(f"  [SIZE] Position: {position_size:.4f} units")
        
        # Step 3: Place paper trade
        order_result = paper_trader.place_order(
            symbol=symbol,
            side=sig['signal'],
            qty=position_size,
            price=None
        )
        
        print(f"  [ORDER] Placed: ID {order_result['id'][:8]}... Status: {order_result['status']}")
        
        # Step 4: Log trade decision
        audit_logger.log_trade_decision(
            symbol=symbol,
            action=sig['signal'].upper(),
            signal_strength=confidence,
            ml_confidence=confidence,
            risk_check_passed=True,
            position_size=position_size,
            reason=f"ML signal: {confidence:.0%} confidence",
            context={'order_id': order_result['id']}
        )
        
        successful_trades += 1
        print(f"  [SUCCESS] Trade logged to audit trail\n")
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Successful trades: {successful_trades}/{len(test_signals)}")
    print(f"Rejected trades: {rejected_trades}/{len(test_signals)}")
    
    # Verify paper trades
    trades = paper_trader.get_trades()
    print(f"Paper trades in DB: {len(trades)}")
    
    # Log test completion
    audit_logger.log_event(
        'TEST_COMPLETE',
        f'Integration test completed: {successful_trades} trades executed',
        context={'successful': successful_trades, 'rejected': rejected_trades}
    )
    
    print("\n[OK] Integration test complete!")
    print("=" * 70)
    
    return successful_trades == len(test_signals)


if __name__ == '__main__':
    success = integration_test_paper_trading()
    sys.exit(0 if success else 1)
