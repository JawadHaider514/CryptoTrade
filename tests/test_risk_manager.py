from crypto_bot.core.risk_manager import RiskManager


def test_position_size():
    rm = RiskManager()
    size = rm.position_size(1000.0, 0.01, 10.0)
    # risk_amount = 1000 * 0.01 = 10; size = 10/10 = 1
    assert abs(size - 1.0) < 1e-6


def test_check_trade_allowed():
    rm = RiskManager()
    assert rm.check_trade_allowed(1000.0, 0.0, 0.01) is True
    assert rm.check_trade_allowed(5.0, 0.0, 0.01) is False  # below min_balance
    assert rm.check_trade_allowed(1000.0, 0.25, 0.01) is False  # drawdown too high
    assert rm.check_trade_allowed(1000.0, 0.0, 0.5) is False  # risk too high
