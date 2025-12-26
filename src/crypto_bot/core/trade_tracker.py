"""
📊 TRADE TRACKER
================
Track all trades, calculate statistics, and export to CSV.

This is a placeholder/stub - you may have a separate trade_tracker.py file
that should be placed here.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class TradeTracker:
    """
    Trade tracking and statistics system.
    Logs all trades and provides analytics.
    """
    
    def __init__(self, data_dir: str = "./trade_data"):
        """
        Initialize the trade tracker.
        
        Args:
            data_dir: Directory to store trade data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades_file = self.data_dir / "trades.json"
        self.trades: List[Dict[str, Any]] = []
        self.trade_counter = 0
        
        # Load existing trades
        self._load_trades()
        
        print(f"📊 TradeTracker initialized with {len(self.trades)} existing trades")
    
    def _load_trades(self) -> None:
        """Load trades from JSON file"""
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    self.trades = json.load(f)
                    self.trade_counter = len(self.trades)
            except (json.JSONDecodeError, IOError):
                self.trades = []
    
    def _save_trades(self) -> None:
        """Save trades to JSON file"""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2, default=str)
        except IOError as e:
            print(f"❌ Error saving trades: {e}")
    
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Log a new trade.
        
        Args:
            trade_data: Dictionary containing trade information
        
        Returns:
            trade_id: Generated trade ID
        """
        self.trade_counter += 1
        trade_id = f"TRADE_{datetime.now().strftime('%Y%m%d')}_{self.trade_counter:04d}"
        
        trade = {
            'trade_id': trade_id,
            'logged_at': datetime.now().isoformat(),
            **trade_data
        }
        
        self.trades.append(trade)
        self._save_trades()
        
        return trade_id
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing trade.
        
        Args:
            trade_id: ID of trade to update
            updates: Dictionary of fields to update
        
        Returns:
            True if trade was found and updated
        """
        for trade in self.trades:
            if trade.get('trade_id') == trade_id:
                trade.update(updates)
                trade['updated_at'] = datetime.now().isoformat()
                self._save_trades()
                return True
        return False
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific trade by ID"""
        for trade in self.trades:
            if trade.get('trade_id') == trade_id:
                return trade
        return None
    
    def get_all_trades(self) -> List[Dict[str, Any]]:
        """Get all trades"""
        return self.trades.copy()
    
    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades"""
        return [t for t in self.trades if t.get('status') == 'OPEN']
    
    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Get all closed trades"""
        return [t for t in self.trades if t.get('status') in ['WIN', 'LOSS', 'CLOSED']]
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate and return trade statistics.
        
        Returns:
            Dictionary with trade statistics
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return {
                'total_trades': len(self.trades),
                'open_trades': len(self.get_open_trades()),
                'closed_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'avg_profit': 0.0,
            }
        
        wins = [t for t in closed_trades if t.get('status') == 'WIN']
        losses = [t for t in closed_trades if t.get('status') == 'LOSS']
        
        total_profit = sum(t.get('profit_usdt', 0) for t in closed_trades)
        
        return {
            'total_trades': len(self.trades),
            'open_trades': len(self.get_open_trades()),
            'closed_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0,
            'total_profit': round(total_profit, 2),
            'avg_profit': round(total_profit / len(closed_trades), 2) if closed_trades else 0.0,
        }
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export all trades to CSV file.
        
        Args:
            filename: Optional custom filename
        
        Returns:
            Path to exported CSV file
        """
        if filename is None:
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.data_dir / filename
        
        if not self.trades:
            print("⚠️ No trades to export")
            return str(filepath)
        
        # Get all unique keys from trades
        fieldnames = set()
        for trade in self.trades:
            fieldnames.update(trade.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trades)
        
        print(f"✅ Exported {len(self.trades)} trades to {filepath}")
        return str(filepath)
    
    def clear_all_trades(self) -> None:
        """Clear all trades (use with caution!)"""
        self.trades = []
        self.trade_counter = 0
        self._save_trades()
        print("🗑️ All trades cleared")


# For backwards compatibility
def create_tracker(data_dir: str = "./trade_data") -> TradeTracker:
    """Factory function to create a TradeTracker instance"""
    return TradeTracker(data_dir=data_dir)
