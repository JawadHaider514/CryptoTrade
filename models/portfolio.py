"""
💼 PORTFOLIO MODELS
===================
Data models for portfolio management and trade tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class TradeStatus(Enum):
    """Trade execution status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class TradeResult(Enum):
    """Trade outcome"""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"


class ExitReason(Enum):
    """Reason for trade exit"""
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    TP3_HIT = "TP3_HIT"
    SL_HIT = "SL_HIT"
    MANUAL = "MANUAL"
    TIMEOUT = "TIMEOUT"
    TRAILING_STOP = "TRAILING_STOP"


@dataclass
class TradePosition:
    """
    Represents an active or historical trade position.
    """
    # Identification
    trade_id: str
    symbol: str
    
    # Trade parameters
    direction: str  # LONG or SHORT
    entry_price: float
    current_price: float
    quantity: float
    leverage: int
    
    # Risk management
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Timing
    entry_time: datetime
    exit_time: Optional[datetime] = None
    
    # Status
    status: TradeStatus = TradeStatus.OPEN
    result: TradeResult = TradeResult.OPEN
    exit_reason: Optional[ExitReason] = None
    
    # PnL tracking
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    fees: float = 0.0
    
    # Partial close tracking
    closed_quantity: float = 0.0
    realized_pnl: float = 0.0
    
    def calculate_pnl(self, current_price: Optional[float] = None) -> float:
        """
        Calculate current PnL for the position.
        
        Args:
            current_price: Current market price (uses self.current_price if not provided)
        
        Returns:
            PnL in USDT
        """
        price = current_price if current_price is not None else self.current_price
        remaining_qty = self.quantity - self.closed_quantity
        
        if self.direction == "LONG":
            unrealized = (price - self.entry_price) * remaining_qty * self.leverage
        else:  # SHORT
            unrealized = (self.entry_price - price) * remaining_qty * self.leverage
        
        return self.realized_pnl + unrealized - self.fees
    
    def calculate_pnl_percentage(self, current_price: Optional[float] = None) -> float:
        """Calculate PnL as percentage of entry"""
        price = current_price if current_price is not None else self.current_price
        
        if self.direction == "LONG":
            return ((price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            return ((self.entry_price - price) / self.entry_price) * 100 * self.leverage
    
    def update_price(self, new_price: float) -> None:
        """Update current price and recalculate PnL"""
        self.current_price = new_price
        self.pnl = self.calculate_pnl()
        self.pnl_percentage = self.calculate_pnl_percentage()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'quantity': self.quantity,
            'leverage': self.leverage,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'status': self.status.value,
            'result': self.result.value,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'fees': self.fees,
            'closed_quantity': self.closed_quantity,
            'realized_pnl': self.realized_pnl,
        }


@dataclass
class Portfolio:
    """
    Manages portfolio state and statistics.
    """
    # Balance
    initial_balance: float
    balance: float
    equity: float
    
    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    
    # Active positions
    active_trades: List[TradePosition] = field(default_factory=list)
    completed_trades: List[TradePosition] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize peak equity"""
        if self.peak_equity == 0.0:
            self.peak_equity = self.initial_balance
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def average_win(self) -> float:
        """Calculate average winning trade"""
        wins = [t for t in self.completed_trades if t.result == TradeResult.WIN]
        if not wins:
            return 0.0
        return sum(t.pnl for t in wins) / len(wins)
    
    @property
    def average_loss(self) -> float:
        """Calculate average losing trade"""
        losses = [t for t in self.completed_trades if t.result == TradeResult.LOSS]
        if not losses:
            return 0.0
        return sum(t.pnl for t in losses) / len(losses)
    
    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.pnl for t in self.completed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.completed_trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    @property
    def return_percentage(self) -> float:
        """Calculate total return percentage"""
        return ((self.equity - self.initial_balance) / self.initial_balance) * 100
    
    def update_equity(self) -> None:
        """Update equity based on balance and open positions"""
        unrealized_pnl = sum(t.pnl for t in self.active_trades)
        self.equity = self.balance + unrealized_pnl
        
        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        current_drawdown = ((self.peak_equity - self.equity) / self.peak_equity) * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def add_trade(self, trade: TradePosition) -> None:
        """Add a new trade to active positions"""
        self.active_trades.append(trade)
        self.update_equity()
    
    def close_trade(self, trade: TradePosition, exit_price: float, reason: ExitReason) -> None:
        """Close a trade and move to completed"""
        trade.exit_time = datetime.now()
        trade.current_price = exit_price
        trade.status = TradeStatus.CLOSED
        trade.exit_reason = reason
        trade.pnl = trade.calculate_pnl(exit_price)
        trade.pnl_percentage = trade.calculate_pnl_percentage(exit_price)
        
        # Determine result
        if trade.pnl > 0:
            trade.result = TradeResult.WIN
            self.winning_trades += 1
        elif trade.pnl < 0:
            trade.result = TradeResult.LOSS
            self.losing_trades += 1
        else:
            trade.result = TradeResult.BREAKEVEN
        
        # Update portfolio
        self.balance += trade.pnl
        self.total_pnl += trade.pnl
        self.total_trades += 1
        
        # Move to completed
        if trade in self.active_trades:
            self.active_trades.remove(trade)
        self.completed_trades.append(trade)
        
        self.update_equity()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get portfolio statistics"""
        return {
            'initial_balance': self.initial_balance,
            'balance': self.balance,
            'equity': self.equity,
            'total_pnl': self.total_pnl,
            'return_percentage': self.return_percentage,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'active_positions': len(self.active_trades),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary"""
        return {
            **self.get_statistics(),
            'active_trades': [t.to_dict() for t in self.active_trades],
            'completed_trades': [t.to_dict() for t in self.completed_trades[-50:]],  # Last 50
        }


@dataclass
class TradeStatistics:
    """
    Aggregated trade statistics for reporting.
    """
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    avg_holding_time: float = 0.0  # in minutes
    
    @classmethod
    def from_trades(cls, trades: List[TradePosition]) -> 'TradeStatistics':
        """Calculate statistics from a list of trades"""
        stats = cls()
        
        if not trades:
            return stats
        
        stats.total_trades = len(trades)
        
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        
        stats.wins = len(wins)
        stats.losses = len(losses)
        stats.breakeven = stats.total_trades - stats.wins - stats.losses
        
        stats.gross_profit = sum(t.pnl for t in wins)
        stats.gross_loss = abs(sum(t.pnl for t in losses))
        stats.total_pnl = stats.gross_profit - stats.gross_loss
        
        if wins:
            stats.largest_win = max(t.pnl for t in wins)
            stats.avg_win = stats.gross_profit / len(wins)
        
        if losses:
            stats.largest_loss = min(t.pnl for t in losses)
            stats.avg_loss = stats.gross_loss / len(losses)
        
        stats.avg_trade = stats.total_pnl / stats.total_trades
        stats.win_rate = (stats.wins / stats.total_trades) * 100
        
        if stats.gross_loss > 0:
            stats.profit_factor = stats.gross_profit / stats.gross_loss
        
        # Calculate average holding time
        holding_times = []
        for t in trades:
            if t.exit_time and t.entry_time:
                delta = (t.exit_time - t.entry_time).total_seconds() / 60
                holding_times.append(delta)
        
        if holding_times:
            stats.avg_holding_time = sum(holding_times) / len(holding_times)
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'breakeven': self.breakeven,
            'total_pnl': round(self.total_pnl, 2),
            'gross_profit': round(self.gross_profit, 2),
            'gross_loss': round(self.gross_loss, 2),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2),
            'avg_trade': round(self.avg_trade, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'win_rate': round(self.win_rate, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_holding_time': round(self.avg_holding_time, 2),
        }
