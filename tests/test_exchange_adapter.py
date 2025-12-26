"""
Unit tests for ExchangeAdapter
"""

import pytest
import os
from crypto_bot.core.exchange_adapter import (
    ExchangeAdapterBase, 
    BinanceTestnetAdapter, 
    LiveBinanceAdapter,
    OrderResult,
    BalanceInfo,
    get_adapter
)


class TestOrderResult:
    def test_success_order(self):
        order = OrderResult(
            success=True,
            order_id='12345',
            symbol='BTC/USDT',
            side='buy',
            amount=0.1,
            price=50000.0,
            status='open'
        )
        assert order.success is True
        assert order.order_id == '12345'
        assert order.symbol == 'BTC/USDT'

    def test_failed_order(self):
        order = OrderResult(
            success=False,
            error_message='Insufficient balance'
        )
        assert order.success is False
        assert order.error_message == 'Insufficient balance'


class TestBalanceInfo:
    def test_balance_creation(self):
        balance = BalanceInfo(free=1000.0, used=500.0, total=1500.0)
        assert balance.free == 1000.0
        assert balance.used == 500.0
        assert balance.total == 1500.0


class TestBinanceTestnetAdapter:
    def test_adapter_init(self):
        adapter = BinanceTestnetAdapter('test_key', 'test_secret')
        assert adapter.api_key == 'test_key'
        assert adapter.api_secret == 'test_secret'
        assert adapter.testnet is True
        assert adapter.exchange is None

    def test_adapter_not_connected(self):
        adapter = BinanceTestnetAdapter('test_key', 'test_secret')
        # Without connecting, methods should return failures gracefully
        result = adapter.place_order('BTC/USDT', 'buy', 0.1, 50000.0)
        assert result.success is False
        assert 'Not connected' in result.error_message


class TestLiveBinanceAdapter:
    def test_adapter_init(self):
        adapter = LiveBinanceAdapter('test_key', 'test_secret')
        assert adapter.testnet is False


class TestGetAdapter:
    def test_get_testnet_adapter_no_keys(self):
        # Unset any existing keys for this test
        old_key = os.environ.pop('API_KEY_TESTNET', None)
        old_secret = os.environ.pop('API_SECRET_TESTNET', None)
        
        adapter = get_adapter('testnet')
        assert adapter is None
        
        # Restore
        if old_key:
            os.environ['API_KEY_TESTNET'] = old_key
        if old_secret:
            os.environ['API_SECRET_TESTNET'] = old_secret

    def test_get_live_adapter_no_keys(self):
        # Unset any existing keys
        old_key = os.environ.pop('API_KEY_LIVE', None)
        old_secret = os.environ.pop('API_SECRET_LIVE', None)
        
        adapter = get_adapter('live')
        assert adapter is None
        
        # Restore
        if old_key:
            os.environ['API_KEY_LIVE'] = old_key
        if old_secret:
            os.environ['API_SECRET_LIVE'] = old_secret

    def test_get_invalid_mode(self):
        adapter = get_adapter('invalid_mode')
        assert adapter is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
