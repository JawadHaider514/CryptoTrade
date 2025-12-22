#!/usr/bin/env python3
"""
Basic Risk Manager for position sizing and simple risk checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01  # fraction of account
    max_drawdown: float = 0.20  # fraction of account
    min_balance: float = 10.0


class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def position_size(self, account_balance: float, risk_fraction: Optional[float], stop_distance: float) -> float:
        """Compute position size (units) given balance, risk fraction (or use default), and stop distance (price units).
        Simple formula: risk_amount = balance * risk_fraction; size = risk_amount / stop_distance
        Note: stop_distance must be > 0 (price units)."""
        if account_balance < self.config.min_balance:
            return 0.0
        risk_fraction = risk_fraction if risk_fraction is not None else self.config.max_risk_per_trade
        if stop_distance <= 0:
            raise ValueError('stop_distance must be > 0')
        risk_amount = account_balance * risk_fraction
        size = risk_amount / stop_distance
        return max(0.0, float(size))

    def check_trade_allowed(self, account_balance: float, current_drawdown: float, proposed_risk_fraction: float) -> bool:
        """Basic checks: account above min, drawdown below max, proposed risk not exceeding max."""
        if account_balance < self.config.min_balance:
            return False
        if current_drawdown >= self.config.max_drawdown:
            return False
        if proposed_risk_fraction > self.config.max_risk_per_trade:
            return False
        return True


if __name__ == '__main__':
    rm = RiskManager()
    sz = rm.position_size(1000.0, None, 50.0)
    print('Size for $1,000 with default risk and $50 stop:', sz)
    allowed = rm.check_trade_allowed(1000.0, 0.05, 0.01)
    print('Trade allowed?', allowed)
