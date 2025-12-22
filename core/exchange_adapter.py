"""
Exchange Adapter Interface and Binance Testnet Implementation

Provides a unified interface for placing orders, checking balances, and querying market data
across different exchanges. Supports both paper trading and live/testnet trading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import ccxt
import os


@dataclass
class OrderResult:
    """Result of order placement."""
    success: bool
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None  # 'buy' or 'sell'
    amount: Optional[float] = None
    price: Optional[float] = None
    status: Optional[str] = None  # 'open', 'closed', 'canceled'
    error_message: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class BalanceInfo:
    """Account balance information."""
    free: float
    used: float
    total: float


class ExchangeAdapterBase(ABC):
    """Abstract base class for exchange adapters."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Initialize adapter.
        
        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            testnet: If True, use testnet/sandbox mode
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange = None

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the exchange. Return True if successful."""
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> OrderResult:
        """
        Place an order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit price (None for market order)
        
        Returns:
            OrderResult with status and order_id or error
        """
        pass

    @abstractmethod
    def get_balance(self, symbol: str = None) -> Dict[str, BalanceInfo]:
        """Get account balance. If symbol provided, return specific currency."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str, symbol: str) -> Optional[str]:
        """Get order status ('open', 'closed', 'canceled')."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order. Return True if successful."""
        pass


class BinanceTestnetAdapter(ExchangeAdapterBase):
    """Binance Testnet Trading Adapter."""

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Binance testnet adapter.
        
        Args:
            api_key: Binance testnet API key
            api_secret: Binance testnet API secret
        """
        super().__init__(api_key, api_secret, testnet=True)

    def connect(self) -> bool:
        """Connect to Binance testnet."""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'urls': {
                    'api': {
                        'public': 'https://testnet.binance.vision/api',
                        'private': 'https://testnet.binance.vision/api',
                    }
                },
                'options': {
                    'defaultType': 'spot',
                    'warnOnFetchOpenOrdersWithoutSymbol': False,
                }
            })
            # Test connection
            self.exchange.load_markets()
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Binance testnet: {e}")
            return False

    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> OrderResult:
        """Place order on testnet."""
        try:
            if not self.exchange:
                return OrderResult(success=False, error_message="Not connected to exchange")

            order_type = 'limit' if price is not None else 'market'
            
            # For market orders on testnet, use a small amount to avoid issues
            if order_type == 'market':
                # Use approximate current price for market orders
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker['last']  # Use last price as estimate

            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            
            return OrderResult(
                success=True,
                order_id=order.get('id'),
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                status=order.get('status', 'open'),
                timestamp=order.get('timestamp')
            )
        except Exception as e:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                amount=amount,
                error_message=str(e)
            )

    def get_balance(self, symbol: str = None) -> Dict[str, BalanceInfo]:
        """Fetch account balance from testnet."""
        try:
            if not self.exchange:
                return {}
            
            balance = self.exchange.fetch_balance()
            result = {}
            
            for currency, data in balance.items():
                if currency not in ['free', 'used', 'total']:
                    result[currency] = BalanceInfo(
                        free=float(data.get('free', 0)),
                        used=float(data.get('used', 0)),
                        total=float(data.get('total', 0))
                    )
            
            return result
        except Exception as e:
            print(f"❌ Failed to fetch balance: {e}")
            return {}

    def get_order_status(self, order_id: str, symbol: str) -> Optional[str]:
        """Get order status from testnet."""
        try:
            if not self.exchange:
                return None
            
            order = self.exchange.fetch_order(order_id, symbol)
            return order.get('status')
        except Exception as e:
            print(f"❌ Failed to fetch order status: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on testnet."""
        try:
            if not self.exchange:
                return False
            
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"❌ Failed to cancel order: {e}")
            return False


class LiveBinanceAdapter(ExchangeAdapterBase):
    """Binance Live Trading Adapter (REAL MONEY - use with caution)."""

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Binance live adapter.
        
        ⚠️ WARNING: This connects to Binance LIVE API. Use only with real API keys and extreme caution.
        """
        super().__init__(api_key, api_secret, testnet=False)

    def connect(self) -> bool:
        """Connect to Binance live API."""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {
                    'defaultType': 'spot',
                    'warnOnFetchOpenOrdersWithoutSymbol': False,
                }
            })
            self.exchange.load_markets()
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Binance live API: {e}")
            return False

    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> OrderResult:
        """Place order on live Binance (REAL MONEY)."""
        try:
            if not self.exchange:
                return OrderResult(success=False, error_message="Not connected to exchange")

            order_type = 'limit' if price is not None else 'market'
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            
            return OrderResult(
                success=True,
                order_id=order.get('id'),
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                status=order.get('status', 'open'),
                timestamp=order.get('timestamp')
            )
        except Exception as e:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                amount=amount,
                error_message=str(e)
            )

    def get_balance(self, symbol: str = None) -> Dict[str, BalanceInfo]:
        """Fetch account balance from live API."""
        try:
            if not self.exchange:
                return {}
            
            balance = self.exchange.fetch_balance()
            result = {}
            
            for currency, data in balance.items():
                if currency not in ['free', 'used', 'total']:
                    result[currency] = BalanceInfo(
                        free=float(data.get('free', 0)),
                        used=float(data.get('used', 0)),
                        total=float(data.get('total', 0))
                    )
            
            return result
        except Exception as e:
            print(f"❌ Failed to fetch balance: {e}")
            return {}

    def get_order_status(self, order_id: str, symbol: str) -> Optional[str]:
        """Get order status from live API."""
        try:
            if not self.exchange:
                return None
            
            order = self.exchange.fetch_order(order_id, symbol)
            return order.get('status')
        except Exception as e:
            print(f"❌ Failed to fetch order status: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on live API."""
        try:
            if not self.exchange:
                return False
            
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"❌ Failed to cancel order: {e}")
            return False


def get_adapter(mode: str = 'testnet') -> Optional[ExchangeAdapterBase]:
    """
    Factory function to get an exchange adapter based on mode.
    
    Args:
        mode: 'testnet' or 'live'
    
    Returns:
        Configured adapter instance or None if keys not found
    """
    if mode == 'testnet':
        api_key = os.getenv('API_KEY_TESTNET')
        api_secret = os.getenv('API_SECRET_TESTNET')
        if not api_key or not api_secret:
            print("❌ Testnet API keys not found in environment (API_KEY_TESTNET, API_SECRET_TESTNET)")
            return None
        return BinanceTestnetAdapter(api_key, api_secret)
    elif mode == 'live':
        api_key = os.getenv('API_KEY_LIVE')
        api_secret = os.getenv('API_SECRET_LIVE')
        if not api_key or not api_secret:
            print("❌ Live API keys not found in environment (API_KEY_LIVE, API_SECRET_LIVE)")
            return None
        return LiveBinanceAdapter(api_key, api_secret)
    else:
        print(f"❌ Unknown mode: {mode}. Use 'testnet' or 'live'")
        return None
