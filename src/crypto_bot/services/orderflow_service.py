#!/usr/bin/env python3
"""
Order Flow Service (Optional)
=============================
Analyze order book flow and market microstructure.
Used by ProfessionalAnalyzer for order book layer analysis.

Can be disabled via feature flag: ENABLE_ORDERFLOW = False
"""

import requests
import time
from typing import Dict, Optional, Tuple
import threading

# Load feature flags from settings
try:
    from config.settings import ENABLE_ORDERFLOW
except ImportError:
    # Fallback default
    ENABLE_ORDERFLOW = False


class OrderFlowService:
    """
    Analyze order book flow and market depth
    
    Features:
    - Bid/Ask ratio analysis
    - Order imbalance detection
    - Market depth assessment
    - Volume profile analysis
    """
    
    ENABLED = False
    
    def __init__(self, enable: Optional[bool] = None):
        """
        Initialize OrderFlowService
        
        Args:
            enable: Enable orderflow analysis (default: uses ENABLE_ORDERFLOW setting)
        """
        # Use provided parameter, fall back to settings, then to False
        if enable is None:
            enable = ENABLE_ORDERFLOW
        
        self.enabled = enable
        self.api_url = "https://api.binance.com/api/v3"
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 30  # 30 seconds
        self.last_request_time = 0
        self.rate_limit_delay = 0.05  # 50ms between requests
    
    def analyze_order_flow(self, symbol: str, depth_limit: int = 20) -> Dict:
        """
        Analyze order book flow for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            depth_limit: Number of price levels to analyze (max 20)
            
        Returns:
            Dictionary with order flow metrics:
            {
                'symbol': str,
                'bid_ask_ratio': float,
                'order_imbalance': float (-1 to 1),
                'market_depth': str ('SHALLOW', 'MODERATE', 'DEEP', 'VERY_DEEP'),
                'large_orders': bool,
                'timestamp': float,
                'error': str (if error occurred)
            }
        """
        if not self.enabled:
            return self._get_neutral_response(symbol, "orderflow disabled")
        
        try:
            # Check cache
            cache_key = f"{symbol}_flow"
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached
            
            # Fetch order book
            order_book = self._fetch_order_book(symbol, depth_limit)
            if not order_book:
                return self._get_neutral_response(symbol, "api error")
            
            # Analyze order flow
            result = self._analyze_book(symbol, order_book)
            
            # Cache result
            self._set_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            return self._get_neutral_response(symbol, str(e))
    
    def _fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Fetch order book from Binance
        
        Args:
            symbol: Trading pair
            limit: Number of levels (5, 10, 20, 50, 100, 500, 1000)
            
        Returns:
            Order book dict or None on error
        """
        try:
            # Rate limiting
            self._rate_limit()
            
            url = f"{self.api_url}/depth"
            params = {
                "symbol": symbol.upper(),
                "limit": min(limit, 1000)
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code != 200:
                return None
            
            return response.json()
            
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def _analyze_book(self, symbol: str, book: Dict) -> Dict:
        """
        Analyze order book structure
        
        Args:
            symbol: Trading pair
            book: Order book data from Binance
            
        Returns:
            Analysis results dictionary
        """
        try:
            bids = [[float(p), float(v)] for p, v in book.get('bids', [])]
            asks = [[float(p), float(v)] for p, v in book.get('asks', [])]
            
            if not bids or not asks:
                return self._get_neutral_response(symbol, "empty book")
            
            # Calculate metrics
            bid_volume = sum(v for _, v in bids[:10])
            ask_volume = sum(v for _, v in asks[:10])
            
            bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            # Order imbalance (-1 to 1)
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                order_imbalance = (bid_volume - ask_volume) / total_volume
            else:
                order_imbalance = 0.0
            
            # Market depth assessment
            bid_depth = sum(v for _, v in bids)
            ask_depth = sum(v for _, v in asks)
            avg_depth = (bid_depth + ask_depth) / 2
            
            if avg_depth > 1000000:
                market_depth = 'VERY_DEEP'
            elif avg_depth > 500000:
                market_depth = 'DEEP'
            elif avg_depth > 100000:
                market_depth = 'MODERATE'
            else:
                market_depth = 'SHALLOW'
            
            # Detect large orders (>5% of top 10)
            large_orders = any(v > bid_volume * 0.05 for _, v in bids[:10]) or \
                          any(v > ask_volume * 0.05 for _, v in asks[:10])
            
            return {
                'symbol': symbol,
                'bid_ask_ratio': round(bid_ask_ratio, 4),
                'order_imbalance': round(order_imbalance, 4),
                'market_depth': market_depth,
                'large_orders': large_orders,
                'bid_volume': round(bid_volume, 2),
                'ask_volume': round(ask_volume, 2),
                'timestamp': time.time(),
                'error': None
            }
            
        except Exception as e:
            return self._get_neutral_response(symbol, str(e))
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        
        self.last_request_time = time.time()
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get value from cache if not expired"""
        with self.cache_lock:
            if key in self.cache:
                cached = self.cache[key]
                age = time.time() - cached['timestamp']
                
                if age < self.cache_ttl:
                    return cached['data']
                else:
                    del self.cache[key]
        
        return None
    
    def _set_in_cache(self, key: str, value: Dict):
        """Set value in cache"""
        with self.cache_lock:
            self.cache[key] = {
                'data': value,
                'timestamp': time.time()
            }
    
    def _get_neutral_response(self, symbol: str, error: str) -> Dict:
        """
        Return neutral/default response
        
        Args:
            symbol: Trading pair
            error: Error message
            
        Returns:
            Neutral response (no signal)
        """
        return {
            'symbol': symbol,
            'bid_ask_ratio': 1.0,  # Neutral
            'order_imbalance': 0.0,  # Neutral
            'market_depth': 'MODERATE',  # Neutral
            'large_orders': False,
            'timestamp': time.time(),
            'error': error
        }
    
    @staticmethod
    def enable_feature():
        """Enable orderflow analysis feature"""
        OrderFlowService.ENABLED = True
    
    @staticmethod
    def disable_feature():
        """Disable orderflow analysis feature"""
        OrderFlowService.ENABLED = False
    
    @staticmethod
    def is_enabled() -> bool:
        """Check if orderflow analysis is enabled"""
        return OrderFlowService.ENABLED


# Singleton instance
_orderflow_service: Optional[OrderFlowService] = None

def get_orderflow_service(enable: bool = False) -> OrderFlowService:
    """
    Get or create singleton OrderFlowService instance
    
    Args:
        enable: Enable orderflow analysis
        
    Returns:
        OrderFlowService instance
    """
    global _orderflow_service
    
    if _orderflow_service is None:
        _orderflow_service = OrderFlowService(enable=enable)
    
    return _orderflow_service
