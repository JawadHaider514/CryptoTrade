Trading & Paper Mode
=====================

Purpose
-------
Documentation for the safe paper trading mode and risk fundamentals included in the repository.

Paper Trading
-------------
- Use `scripts/demo_paper_trade.py` to run a one-off simulated trade.
- Use `PaperTrader` (`core/paper_trader.py`) to place simulated orders; trades are saved to the `paper_trades` table in the existing DB.
- Paper trades are **simulated immediate fills** for deterministic testing; this is safe for development.

Risk Manager
------------
- `core/risk_manager.py` provides simple position sizing and trade-allowance checks (max risk per trade, min balance, max drawdown).
- Use `RiskManager.position_size()` to compute trade sizes given account balances and stop distances.

Running a demo
--------------
- From project root:
  python run.py --paper-demo

Notes & Next Steps
------------------
- The paper trader is intentionally simple (immediate fills) to facilitate tests. Next steps: support partial fills, slippage, and time-based fills to simulate realistic conditions.
- The risk manager is minimal; production requires more advanced risk controls (per-symbol limits, portfolio constraints, margin/leverage calculations).
