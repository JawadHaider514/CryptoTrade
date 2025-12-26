#!/usr/bin/env python3
"""
🔴 BINANCE WEBSOCKET INTEGRATION
Real-time price and candle streaming from Binance
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
import requests
from collections import deque

logger = logging.getLogger(__name__)

class BinanceWSManager:
    """Manages Binance WebSocket connections for real-time data"""
    
    def __init__(self):
        self.trade_data: Dict[str, float] = {}
        self.kline_data: Dict[str, dict] = {}
        self.ws_connections: Dict[str, Any] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.server_time_offset = 0
        self.lock = threading.Lock()
        self._init_server_time()
    
    def _init_server_time(self):
        """Get Binance server time to sync candle boundaries"""
        try:
            resp = requests.get('https://api.binance.com/api/v3/time', timeout=5)
            if resp.status_code == 200:
                binance_time = resp.json()['serverTime']
                local_time = int(time.time() * 1000)
                self.server_time_offset = binance_time - local_time
                logger.info(f"✅ Binance server time synced (offset: {self.server_time_offset}ms)")
        except Exception as e:
            logger.error(f"⚠️  Failed to sync server time: {e}")
    
    def get_server_time(self) -> int:
        """Get current Binance server time in milliseconds"""
        return int(time.time() * 1000) + self.server_time_offset
    
    def fetch_klines(self, symbol: str, interval: str = '1h', limit: int = 500) -> List[dict]:
        """Fetch historical klines from Binance REST API"""
        try:
            print(f"[BINANCE] Fetching klines: {symbol} {interval}")
            url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': symbol.upper(),
                'interval': interval.lower(),
                'limit': limit
            }
            print(f"[BINANCE] GET {url} params={params}")
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            
            candles = []
            for kline in resp.json():
                candles.append({
                    'time': int(kline[0] / 1000),  # Convert to seconds
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[7]),
                    'trades': int(kline[8])
                })
            
            print(f"[BINANCE] SUCCESS: {len(candles)} candles for {symbol}")
            return candles
        except Exception as e:
            print(f"[BINANCE ERROR] Failed to fetch klines for {symbol}: {e}")
            return []
    
    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch latest price from Binance"""
        try:
            print(f"[BINANCE] Fetching price for {symbol}")
            url = 'https://api.binance.com/api/v3/ticker/price'
            params = {'symbol': symbol.upper()}
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            price = float(resp.json()['price'])
            print(f"[BINANCE] Price for {symbol}: {price}")
            return price
        except Exception as e:
            print(f"[BINANCE ERROR] Failed to fetch price for {symbol}: {e}")
            return None
    
    def fetch_24h_stats(self, symbol: str) -> dict:
        """Fetch 24h statistics from Binance"""
        try:
            print(f"[BINANCE] Fetching 24h stats for {symbol}")
            url = 'https://api.binance.com/api/v3/ticker/24hr'
            params = {'symbol': symbol.upper()}
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            stats = {
                'price': float(data['lastPrice']),
                'high24h': float(data['highPrice']),
                'low24h': float(data['lowPrice']),
                'volume24h': float(data['volume']),
                'quoteVolume24h': float(data['quoteVolume']),
                'change24h': float(data['priceChange']),
                'changePercent24h': float(data['priceChangePercent'])
            }
            print(f"[BINANCE] Stats for {symbol}: {stats}")
            return stats
        except Exception as e:
            print(f"[BINANCE ERROR] Failed to fetch 24h stats for {symbol}: {e}")
            return {}
    
    def get_account_balance(self) -> dict:
        """Get account balance using API keys (authenticated)"""
        try:
            from crypto_bot.config.settings import APP_CONFIG
            import hmac
            import hashlib
            
            api_key = APP_CONFIG.get('BINANCE_API_KEY')
            secret_key = APP_CONFIG.get('BINANCE_SECRET_KEY')
            testnet_url = APP_CONFIG.get('BINANCE_TESTNET_URL', 'https://testnet.binance.vision')
            
            if not api_key or not secret_key:
                print("[BINANCE] API keys not configured")
                return {}
            
            timestamp = int(time.time() * 1000)
            params: Dict[str, Any] = {'timestamp': timestamp}
            
            # Create query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            
            # Create signature
            signature = hmac.new(
                secret_key.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            url = f'{testnet_url}/api/v3/account'
            headers = {'X-MBX-APIKEY': api_key}
            
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            
            data = resp.json()
            print(f"[BINANCE] Account balance fetched successfully")
            return data
        except Exception as e:
            print(f"[BINANCE ERROR] Failed to fetch account balance: {e}")
            return {}
    
    def place_test_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """Place test order (testnet only - no real money)"""
        try:
            from crypto_bot.config.settings import APP_CONFIG
            import hmac
            import hashlib
            
            api_key = APP_CONFIG.get('BINANCE_API_KEY')
            secret_key = APP_CONFIG.get('BINANCE_SECRET_KEY')
            testnet_url = APP_CONFIG.get('BINANCE_TESTNET_URL', 'https://testnet.binance.vision')
            
            if not api_key or not secret_key:
                print("[BINANCE] API keys not configured")
                return {}
            
            timestamp = int(time.time() * 1000)
            params: Dict[str, Any] = {
                'symbol': symbol.upper(),
                'side': side.upper(),
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp
            }
            
            # Create query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            
            # Create signature
            signature = hmac.new(
                secret_key.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            url = f'{testnet_url}/api/v3/order/test'
            headers = {'X-MBX-APIKEY': api_key}
            
            resp = requests.post(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            
            print(f"[BINANCE] Test order placed: {symbol} {side} {quantity} @ {price}")
            return resp.json()
        except Exception as e:
            print(f"[BINANCE ERROR] Failed to place test order: {e}")
            return {}

# Global instance
binance_ws = BinanceWSManager()

def get_binance_manager() -> BinanceWSManager:
    """Get global Binance WS manager instance"""
    return binance_ws
