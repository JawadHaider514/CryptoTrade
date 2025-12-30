#!/usr/bin/env python3
"""
Market Data Service
===================
Live price tracking with Binance WebSocket
"""

import requests
import threading
import json
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
import websocket

logger = logging.getLogger(__name__)


class MarketDataService:
    """Manages live price data for all symbols"""
    
    def __init__(self, symbols: List[str], use_websocket: bool = True):
        self.symbols = symbols
        self.prices: Dict[str, float] = {}
        self.timestamps: Dict[str, datetime] = {}
        self.lock = threading.Lock()
        self.running = False
        self.ws = None
        self.use_websocket = use_websocket
        
        # Initialize with fallback prices immediately
        for symbol in symbols:
            self.prices[symbol] = 100.0
            self.timestamps[symbol] = datetime.utcnow()
        
        # Bootstrap prices in background (non-blocking)
        bootstrap_thread = threading.Thread(target=self._bootstrap_prices, daemon=True)
        bootstrap_thread.start()
    
    def _bootstrap_prices(self):
        """Get initial prices from Binance REST API"""
        logger.info(f"🔄 Bootstrapping prices for {len(self.symbols)} symbols...")
        
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_price(symbol):
            try:
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                response = requests.get(url, timeout=10)  # increased from 2 to 10 seconds
                if response.status_code == 200:
                    data = response.json()
                    price = float(data['price'])
                    
                    with self.lock:
                        self.prices[symbol] = price
                        self.timestamps[symbol] = datetime.utcnow()
                    
                    logger.debug(f"✅ {symbol}: ${price}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to get price for {symbol}: {e}")
                # Use default price if fetch fails
                with self.lock:
                    self.prices[symbol] = 100.0  # Default fallback price
                    self.timestamps[symbol] = datetime.utcnow()
        
        # Use ThreadPoolExecutor with max 10 concurrent threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_price, symbol) for symbol in self.symbols]
            # Wait for completion with timeout (60s for all symbols)
            for future in as_completed(futures, timeout=60):
                try:
                    future.result()
                except Exception as e:
                    logger.debug(f"Thread error during bootstrap: {e}")
        
        logger.info(f"✅ Bootstrapped {len(self.prices)} symbols")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        with self.lock:
            return self.prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices"""
        with self.lock:
            return dict(self.prices)
    
    def get_price_with_timestamp(self, symbol: str) -> Optional[Dict]:
        """Get price with timestamp"""
        with self.lock:
            if symbol in self.prices:
                return {
                    'price': self.prices[symbol],
                    'timestamp': self.timestamps[symbol]
                }
        return None
    
    def _update_price(self, symbol: str, price: float):
        """Internal: update price with timestamp"""
        with self.lock:
            self.prices[symbol] = price
            self.timestamps[symbol] = datetime.utcnow()
    
    def start_websocket(self):
        """Start Binance WebSocket for live prices"""
        if not self.use_websocket:
            logger.warning("WebSocket disabled, using REST polling instead")
            self._start_polling()
            return
        
        logger.info("🔌 Starting Binance WebSocket...")
        self.running = True
        
        try:
            # Use preferred endpoint: !miniTicker@arr for all symbols (most efficient)
            ws_url = "wss://stream.binance.com:9443/ws/!miniTicker@arr"
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    
                    # Handle !miniTicker@arr format (array of all tickers)
                    if isinstance(data, list):
                        for tick in data:
                            symbol_lower = tick.get('s', '').lower()
                            # Find matching symbol (case-insensitive)
                            for sym in self.symbols:
                                if sym.lower() == symbol_lower:
                                    price = float(tick.get('c', 0))
                                    self._update_price(sym, price)
                                    break
                    # Handle fallback format (data wrapper)
                    elif 'data' in data:
                        tick = data['data']
                        symbol_lower = tick.get('s', '').lower()
                        
                        # Find matching symbol (case-insensitive)
                        for sym in self.symbols:
                            if sym.lower() == symbol_lower:
                                price = float(tick.get('c', 0))
                                self._update_price(sym, price)
                                break
                except Exception as e:
                    logger.error(f"WebSocket parse error: {e}")
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
                # Switch to polling on error
                logger.warning("Switching to REST polling due to WebSocket error")
                self.use_websocket = False
                self._start_polling()
            
            def on_close(ws, close_status_code, close_msg):
                logger.warning("WebSocket closed")
                if self.running and self.use_websocket:
                    logger.info("Attempting to reconnect...")
                    time.sleep(5)
                    self.start_websocket()
            
            def on_open(ws):
                logger.info("✅ WebSocket connected to !miniTicker@arr")
            
            websocket.enableTrace(False)
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Run in background thread
            thread = threading.Thread(
                target=self.ws.run_forever,
                daemon=True
            )
            thread.start()
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
            logger.info("Falling back to REST polling")
            self.use_websocket = False
            self._start_polling()
    
    def _start_polling(self):
        """Fallback: poll prices every 1 second"""
        logger.info("📊 Starting price polling (fallback mode)...")
        
        def polling_loop():
            while self.running:
                try:
                    prices = self.get_all_prices()
                    self._bootstrap_prices()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Polling error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=polling_loop, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop WebSocket and polling"""
        logger.info("🛑 Stopping market data service...")
        self.running = False
        
        if self.ws:
            self.ws.close()
