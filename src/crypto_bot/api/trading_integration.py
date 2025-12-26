"""
INTEGRATION MODULE
Connect TradeTracker with existing trading system
"""

from trade_tracker import TradeTracker
from datetime import datetime
from typing import Dict, Any, Optional

class TradingSystemIntegration:
    """
    Integration layer between trading system and tracker
    Use this in your main trading bot
    """
    
    def __init__(self, data_dir: str = "./trade_data"):
        self.tracker = TradeTracker(data_dir=data_dir)
        self.active_trades = {}  # Store active trade IDs
        
        print("🔗 Trading System Integration initialized")
    
    def on_signal_generated(self, signal: Dict[str, Any]) -> str:
        """
        Call this when a trading signal is generated
        
        Args:
            signal: Signal dictionary from your analyzer
        
        Returns:
            trade_id: Generated trade ID
        """
        # Extract coin name from symbol
        coin_name = self._get_coin_name(signal.get('symbol', 'UNKNOWN'))
        
        # Prepare trade data
        trade_data = {
            'pair': signal.get('symbol', 'N/A'),
            'coin_name': coin_name,
            'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'direction': signal.get('direction', 'N/A'),
            'entry_price': signal.get('entry_price', 0.0),
            'stop_loss': signal.get('stop_loss', 0.0),
            'take_profit': self._format_take_profits(signal.get('take_profits', [])),
            'tp_timeframe': signal.get('timeframe', 'N/A'),
            'predicted_accuracy': signal.get('accuracy_estimate', 0.0),
            'status': 'OPEN',
            'timeframe': signal.get('timeframe', '5m'),
            'strategy': 'Professional Analyzer'
        }
        
        # Log trade
        trade_id = self.tracker.log_trade(trade_data)
        
        # Store in active trades
        self.active_trades[signal['symbol']] = trade_id
        
        print(f"📝 Trade logged: {trade_id} ({signal['symbol']} {signal['direction']})")
        
        return trade_id
    
    def on_trade_exit(self, symbol: str, exit_data: Dict[str, Any]):
        """
        Call this when a trade exits (TP hit or SL hit)
        
        Args:
            symbol: Trading pair symbol
            exit_data: Exit information
        """
        if symbol not in self.active_trades:
            print(f"⚠️  No active trade found for {symbol}")
            return
        
        trade_id = self.active_trades[symbol]
        
        # Calculate profit/loss
        profit_usdt = self._calculate_profit(exit_data)
        
        # Determine status
        status = "WIN" if profit_usdt > 0 else "LOSS"
        
        # Prepare updates
        updates = {
            'exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'exit_price': exit_data.get('exit_price', 0.0),
            'exit_timeframe': exit_data.get('timeframe', 'N/A'),
            'actual_result': exit_data.get('profit_percentage', 0.0),
            'status': status,
            'profit_usdt': profit_usdt
        }
        
        # Update trade
        self.tracker.update_trade(trade_id, updates)
        
        # Remove from active trades
        del self.active_trades[symbol]
        
        print(f"✅ Trade closed: {trade_id} ({status}) - ${profit_usdt:.2f}")
    
    def get_active_trade_id(self, symbol: str) -> Optional[str]:
        """Get active trade ID for a symbol"""
        return self.active_trades.get(symbol)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current trading statistics"""
        return self.tracker.get_trade_statistics()
    
    def _get_coin_name(self, symbol: str) -> str:
        """Extract coin name from symbol"""
        coin_map = {
            'BTCUSDT': 'Bitcoin',
            'ETHUSDT': 'Ethereum',
            'XRPUSDT': 'Ripple',
            'BNBUSDT': 'Binance Coin',
            'ADAUSDT': 'Cardano',
            'SOLUSDT': 'Solana',
            'DOGEUSDT': 'Dogecoin',
            'DOTUSDT': 'Polkadot'
        }
        
        return coin_map.get(symbol, symbol.replace('USDT', ''))
    
    def _format_take_profits(self, take_profits: list) -> str:
        """Format take profit levels into string"""
        if not take_profits:
            return "N/A"
        
        # Take first TP level
        if isinstance(take_profits[0], (list, tuple)):
            return f"{take_profits[0][0]:.2f}"
        else:
            return f"{take_profits[0]:.2f}"
    
    def _calculate_profit(self, exit_data: Dict[str, Any]) -> float:
        """Calculate profit in USDT"""
        # This is simplified - you should use actual position size
        profit_pct = exit_data.get('profit_percentage', 0.0)
        position_size = exit_data.get('position_size', 100.0)  # Default $100
        
        profit_usdt = position_size * (profit_pct / 100)
        
        return round(profit_usdt, 2)


# ============================================================================
# USAGE EXAMPLE WITH YOUR TRADING BOT
# ============================================================================

def example_integration():
    """
    Example of how to integrate with your existing trading bot
    """
    
    # Initialize integration
    integration = TradingSystemIntegration()
    
    print("\n" + "="*60)
    print("INTEGRATION EXAMPLE")
    print("="*60)
    
    # Example 1: When signal is generated
    print("\n1️⃣ Signal Generated:")
    
    signal = {
        'symbol': 'BTCUSDT',
        'direction': 'LONG',
        'entry_price': 50234.56,
        'stop_loss': 49800.00,
        'take_profits': [(51200.00, 50), (51800.00, 30), (52500.00, 20)],
        'timeframe': '5m',
        'accuracy_estimate': 92.5,
        'confluence_score': 88
    }
    
    trade_id = integration.on_signal_generated(signal)
    print(f"   Trade ID: {trade_id}")
    
    # Example 2: When trade exits (after some time)
    print("\n2️⃣ Trade Exit:")
    
    import time
    time.sleep(1)  # Simulate time passing
    
    exit_data = {
        'exit_price': 51150.00,
        'timeframe': '5m',
        'profit_percentage': 1.82,  # 1.82% profit
        'position_size': 5000.00    # $5000 position
    }
    
    integration.on_trade_exit('BTCUSDT', exit_data)
    
    # Example 3: Get statistics
    print("\n3️⃣ Trading Statistics:")
    
    stats = integration.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Integration example complete!")


# ============================================================================
# ADD TO YOUR MAIN TRADING BOT
# ============================================================================

"""
HOW TO USE IN YOUR TRADING BOT:
================================

# In your main trading file (e.g., crypto_dashboard.py):

from trading_integration import TradingSystemIntegration

# Initialize at startup
tracker = TradingSystemIntegration()

# When signal is generated:
def analyze_and_trade(symbol):
    signal = analyzer.analyze_symbol(symbol)
    
    if signal:
        # Log the trade
        trade_id = tracker.on_signal_generated(signal)
        
        # Execute trade...
        # ...
        
        return trade_id

# When trade exits:
def on_trade_close(symbol, exit_price, profit_pct):
    exit_data = {
        'exit_price': exit_price,
        'timeframe': '5m',
        'profit_percentage': profit_pct,
        'position_size': 1000.0
    }
    
    tracker.on_trade_exit(symbol, exit_data)

# View statistics anytime:
def show_stats():
    stats = tracker.get_statistics()
    print(stats)
"""


if __name__ == "__main__":
    example_integration()