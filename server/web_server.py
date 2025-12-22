#!/usr/bin/env python3
"""
🌐 CRYPTO DASHBOARD - WEB SERVER
================================
Browser-based interface with stunning UI
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from typing import Optional, Dict, Any, List
import json
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Binance WS manager
try:
    from server.binance_ws import get_binance_manager
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️  Binance WS module not available")

# Import with error handling
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    CORS = None

# Import dashboard components with error handling
try:
    from core.enhanced_crypto_dashboard import (
        EnhancedScalpingDashboard,
        ScalpingConfig,
        SignalFormatter
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    try:
        # Try direct import if running from server directory
        from enhanced_crypto_dashboard import (
            EnhancedScalpingDashboard,
            ScalpingConfig,
            SignalFormatter
        )
        DASHBOARD_AVAILABLE = True
    except ImportError:
        print("⚠️  Dashboard modules not found - running in demo mode")
        DASHBOARD_AVAILABLE = False
        EnhancedScalpingDashboard = None
        ScalpingConfig = None
        SignalFormatter = None

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / 'templates'
STATIC_DIR = PROJECT_ROOT / 'static'

# Initialize Flask with correct template folder
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR)
)

# Enable CORS if available
if CORS_AVAILABLE and CORS is not None:
    CORS(app)

# Global dashboard instance
dashboard: Optional[Any] = None

# Signal cache with timestamp (3 minutes = 180 seconds)
signal_cache: Dict[str, Dict[str, Any]] = {}
SIGNAL_CACHE_DURATION = 3 * 60  # 3 minutes in seconds

# Demo data for when dashboard is not available
DEMO_SIGNALS = [
    {
        'symbol': 'BTCUSDT',
        'direction': 'LONG',
        'entry_price': 98234.56,
        'entry_range': [98200.00, 98270.00],
        'stop_loss': 97800.00,
        'take_profits': [[98500.00, 40], [98750.00, 35], [99000.00, 25]],
        'confluence_score': 87,
        'accuracy_estimate': 92.5,
        'detected_patterns': ['bullish_engulfing', 'volume_surge'],
        'leverage': 20,
        'risk_percentage': 2.0,
        'timestamp': datetime.now()
    },
    {
        'symbol': 'ETHUSDT',
        'direction': 'LONG',
        'entry_price': 3456.78,
        'entry_range': [3450.00, 3460.00],
        'stop_loss': 3420.00,
        'take_profits': [[3480.00, 40], [3510.00, 35], [3550.00, 25]],
        'confluence_score': 82,
        'accuracy_estimate': 88.3,
        'detected_patterns': ['rsi_oversold', 'macd_crossover'],
        'leverage': 20,
        'risk_percentage': 2.0,
        'timestamp': datetime.now()
    },
    {
        'symbol': 'SOLUSDT',
        'direction': 'SHORT',
        'entry_price': 198.45,
        'entry_range': [198.20, 198.70],
        'stop_loss': 201.50,
        'take_profits': [[196.00, 40], [194.00, 35], [191.50, 25]],
        'confluence_score': 78,
        'accuracy_estimate': 85.7,
        'detected_patterns': ['bearish_divergence', 'resistance_rejection'],
        'leverage': 15,
        'risk_percentage': 2.0,
        'timestamp': datetime.now()
    },
    {
        'symbol': 'BNBUSDT',
        'direction': 'LONG',
        'entry_price': 612.34,
        'entry_range': [610.00, 615.00],
        'stop_loss': 605.00,
        'take_profits': [[620.00, 40], [628.00, 35], [640.00, 25]],
        'confluence_score': 75,
        'accuracy_estimate': 82.1,
        'detected_patterns': ['support_bounce', 'volume_increase'],
        'leverage': 20,
        'risk_percentage': 2.0,
        'timestamp': datetime.now()
    }
]


def init_dashboard() -> Optional[Any]:
    """Initialize dashboard"""
    global dashboard
    if DASHBOARD_AVAILABLE and dashboard is None and EnhancedScalpingDashboard is not None:
        try:
            dashboard = EnhancedScalpingDashboard()
            print("✅ Dashboard initialized successfully")
        except Exception as e:
            print(f"⚠️  Dashboard initialization failed: {e}")
            return None
    return dashboard


def clean_expired_signals() -> None:
    """Remove signals that have expired from cache"""
    global signal_cache
    current_time = time.time()
    expired_symbols = [
        symbol for symbol, data in signal_cache.items()
        if current_time - data['timestamp'] > SIGNAL_CACHE_DURATION
    ]
    for symbol in expired_symbols:
        del signal_cache[symbol]


def calculate_expected_times(signal_timestamp: datetime) -> Dict[str, Any]:
    """Calculate expected times to reach each trading level"""
    return {
        'entry_time': signal_timestamp.strftime("%H:%M:%S"),
        'entry_label': 'NOW',
        'tp1_time': (signal_timestamp + timedelta(minutes=2)).strftime("%H:%M:%S"),
        'tp1_label': 'Must Hit (2 min)',
        'tp2_time': (signal_timestamp + timedelta(minutes=3, seconds=30)).strftime("%H:%M:%S"),
        'tp2_label': 'Scalp (3:30 min)',
        'tp3_time': (signal_timestamp + timedelta(minutes=5)).strftime("%H:%M:%S"),
        'tp3_label': 'Final Exit (5 min)',
        'exit_time': (signal_timestamp + timedelta(minutes=5)).strftime("%H:%M:%S"),
        'exit_label': 'Hard Stop (5 min)',
        'total_window': '5 minutes',
        'strategy': 'INTELLIGENT_SCALP'
    }


def get_demo_signals() -> List[Dict[str, Any]]:
    """Get demo signals with updated timestamps"""
    signals = []
    for signal in DEMO_SIGNALS:
        sig = signal.copy()
        sig['timestamp'] = datetime.now()
        signals.append(sig)
    return signals


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index() -> str:
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/signals', methods=['GET'])
def get_signals() -> Response:
    """Get all current signals"""
    try:
        global signal_cache
        clean_expired_signals()
        
        signals_data = []
        
        # Try to get real signals from dashboard
        dash = init_dashboard()
        if dash:
            try:
                dash.generate_all_signals()
                current_time = time.time()
                
                for symbol, signal in dash.signals.items():
                    if symbol not in signal_cache:
                        signal_cache[symbol] = {
                            'data': signal,
                            'timestamp': current_time
                        }
                
                for symbol, cache_entry in signal_cache.items():
                    signal = cache_entry['data']
                    expected_times = calculate_expected_times(signal['timestamp'])
                    
                    signals_data.append({
                        'symbol': signal['symbol'],
                        'direction': signal['direction'],
                        'entry_price': signal['entry_price'],
                        'entry_range': signal.get('entry_range', [0, 0]),
                        'stop_loss': signal['stop_loss'],
                        'take_profits': signal['take_profits'],
                        'confluence_score': signal['confluence_score'],
                        'accuracy_estimate': signal['accuracy_estimate'],
                        'timestamp': signal['timestamp'].isoformat(),
                        'detected_patterns': signal.get('detected_patterns', []),
                        'leverage': signal.get('leverage', 20),
                        'risk_percentage': signal.get('risk_percentage', 2.0),
                        **expected_times
                    })
            except Exception as e:
                print(f"⚠️  Error generating signals: {e}")
                signals_data = format_demo_signals()
        else:
            # Use demo signals
            signals_data = format_demo_signals()
        
        return jsonify({
            'success': True,
            'count': len(signals_data),
            'signals': signals_data,
            'timestamp': datetime.now().isoformat(),
            'cache_duration_seconds': SIGNAL_CACHE_DURATION,
            'demo_mode': not DASHBOARD_AVAILABLE
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


def format_demo_signals() -> List[Dict[str, Any]]:
    """Format demo signals for API response"""
    signals_data = []
    for signal in get_demo_signals():
        expected_times = calculate_expected_times(signal['timestamp'])
        signals_data.append({
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'entry_price': signal['entry_price'],
            'entry_range': signal['entry_range'],
            'stop_loss': signal['stop_loss'],
            'take_profits': signal['take_profits'],
            'confluence_score': signal['confluence_score'],
            'accuracy_estimate': signal['accuracy_estimate'],
            'timestamp': signal['timestamp'].isoformat(),
            'detected_patterns': signal['detected_patterns'],
            'leverage': signal['leverage'],
            'risk_percentage': signal['risk_percentage'],
            **expected_times
        })
    return signals_data


@app.route('/api/signals/<symbol>', methods=['GET'])
def get_signal(symbol: str) -> Response:
    """Get signal for specific symbol"""
    try:
        clean_expired_signals()
        
        # Check cache first
        if symbol in signal_cache:
            signal = signal_cache[symbol]['data']
            expected_times = calculate_expected_times(signal['timestamp'])
            
            return jsonify({
                'success': True,
                'signal': {
                    'symbol': signal['symbol'],
                    'direction': signal['direction'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profits': signal['take_profits'],
                    'confluence_score': signal['confluence_score'],
                    'accuracy_estimate': signal['accuracy_estimate'],
                    'timestamp': signal['timestamp'].isoformat(),
                    **expected_times
                }
            })
        
        # Check demo signals
        for signal in DEMO_SIGNALS:
            if signal['symbol'] == symbol:
                sig = signal.copy()
                sig['timestamp'] = datetime.now()
                expected_times = calculate_expected_times(sig['timestamp'])
                
                return jsonify({
                    'success': True,
                    'signal': {
                        **sig,
                        'timestamp': sig['timestamp'].isoformat(),
                        **expected_times
                    },
                    'demo_mode': True
                })
        
        return jsonify({
            'success': False,
            'error': f'No signal found for {symbol}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/statistics', methods=['GET'])
def get_statistics() -> Response:
    """Get trading statistics"""
    try:
        # Try to get real statistics
        dash = init_dashboard()
        
        if dash and hasattr(dash, 'get_statistics'):
            stats = dash.get_statistics()
        else:
            # Demo statistics
            stats = {
                'total_trades': 47,
                'winning_trades': 38,
                'losing_trades': 9,
                'win_rate': 80.85,
                'total_profit': 1234.56,
                'avg_profit': 26.27,
                'best_trade': 156.78,
                'worst_trade': -45.23,
                'active_signals': len(DEMO_SIGNALS)
            }
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/trades', methods=['GET'])
def get_trades() -> Response:
    """Get trade history"""
    try:
        # Return demo trades for now
        trades = [
            {
                'symbol': 'BTCUSDT',
                'direction': 'LONG',
                'entry_price': 97234.50,
                'exit_price': 97567.80,
                'pnl': 156.25,
                'pnl_percentage': 3.2,
                'status': 'COMPLETED',
                'entry_time': (datetime.now() - timedelta(minutes=4)).isoformat(),
                'exit_time': datetime.now().isoformat(),
                'duration': '4m 12s'
            },
            {
                'symbol': 'ETHUSDT',
                'direction': 'SHORT',
                'entry_price': 3456.00,
                'exit_price': 3423.45,
                'pnl': 98.15,
                'pnl_percentage': 1.9,
                'status': 'COMPLETED',
                'entry_time': (datetime.now() - timedelta(minutes=7)).isoformat(),
                'exit_time': (datetime.now() - timedelta(minutes=1)).isoformat(),
                'duration': '6m 34s'
            },
            {
                'symbol': 'SOLUSDT',
                'direction': 'LONG',
                'entry_price': 198.45,
                'exit_price': 196.23,
                'pnl': -45.00,
                'pnl_percentage': -2.2,
                'status': 'COMPLETED',
                'entry_time': (datetime.now() - timedelta(minutes=10)).isoformat(),
                'exit_time': (datetime.now() - timedelta(minutes=7)).isoformat(),
                'duration': '2m 45s'
            },
            {
                'symbol': 'BNBUSDT',
                'direction': 'LONG',
                'entry_price': 612.34,
                'exit_price': 625.67,
                'pnl': 87.40,
                'pnl_percentage': 2.1,
                'status': 'COMPLETED',
                'entry_time': (datetime.now() - timedelta(minutes=13)).isoformat(),
                'exit_time': (datetime.now() - timedelta(minutes=8)).isoformat(),
                'duration': '5m 18s'
            }
        ]
        
        return jsonify({
            'success': True,
            'trades': trades,
            'count': len(trades),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'trades': []
        })


@app.route('/api/bot/status', methods=['GET'])
def get_bot_status() -> Response:
    """Get bot status"""
    return jsonify({
        'success': True,
        'status': {
            'running': DASHBOARD_AVAILABLE,
            'mode': 'live' if DASHBOARD_AVAILABLE else 'demo',
            'uptime': '00:00:00',
            'last_signal': datetime.now().isoformat()
        }
    })


@app.route('/api/discord-notify', methods=['POST'])
def discord_notify() -> Response:
    """Send notification to Discord"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'UNKNOWN')
        signal = data.get('signal', 'UNKNOWN')
        confidence = data.get('confidence', 0)
        price = data.get('price', 0)
        
        print(f"📢 Discord notification requested for {symbol}")
        
        # Get Discord webhook URL from config
        from config.settings import APP_CONFIG
        webhook_url = APP_CONFIG.get('DISCORD_WEBHOOK')
        
        if not webhook_url:
            print("⚠️  Discord webhook not configured")
            return jsonify({
                'success': False,
                'error': 'Discord webhook not configured'
            }), 503
        
        # Send to Discord
        import requests
        
        # Create Discord embed message
        embed = {
            "title": f"🚀 {signal.upper()} Signal - {symbol}",
            "description": f"**Price**: ${price:,.2f}\n**Confidence**: {confidence}%",
            "color": 3066993 if signal.lower() == 'long' else 15158332,  # Green for LONG, Red for SHORT
            "fields": [
                {
                    "name": "Symbol",
                    "value": symbol,
                    "inline": True
                },
                {
                    "name": "Signal",
                    "value": signal.upper(),
                    "inline": True
                },
                {
                    "name": "Confidence",
                    "value": f"{confidence}%",
                    "inline": True
                },
                {
                    "name": "Time",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "inline": False
                }
            ],
            "footer": {
                "text": "Crypto Trading System"
            }
        }
        
        payload = {
            "embeds": [embed]
        }
        
        # Send to Discord webhook
        resp = requests.post(webhook_url, json=payload, timeout=10)
        
        if resp.status_code in [200, 204]:
            print(f"✅ Discord notification sent for {symbol}")
            return jsonify({
                'success': True,
                'message': f"Notification sent for {symbol}"
            })
        else:
            print(f"⚠️  Discord webhook error: {resp.status_code} - {resp.text}")
            return jsonify({
                'success': False,
                'error': f'Discord webhook failed: {resp.status_code}'
            }), 503
        
    except Exception as e:
        print(f"❌ Discord notification error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/account/balance', methods=['GET'])
def get_account_balance() -> Response:
    """Get account balance from Binance (authenticated)"""
    try:
        print("[ACCOUNT] Fetching account balance...")
        
        if not BINANCE_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Binance integration not available'
            }), 503
        
        binance = get_binance_manager()
        account_data = binance.get_account_balance()
        
        if not account_data:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch account balance'
            }), 503
        
        # Extract balances
        balances = {}
        for asset in account_data.get('balances', []):
            free = float(asset.get('free', 0))
            locked = float(asset.get('locked', 0))
            if free > 0 or locked > 0:
                balances[asset['asset']] = {
                    'free': free,
                    'locked': locked,
                    'total': free + locked
                }
        
        print(f"[ACCOUNT] ✓ Balance fetched: {len(balances)} assets")
        return jsonify({
            'success': True,
            'balances': balances,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"[ACCOUNT ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/order/test', methods=['POST'])
def test_order() -> Response:
    """Place a test order on Binance testnet (no real money)"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        side = data.get('side', '').upper()
        quantity = float(data.get('quantity', 0))
        price = float(data.get('price', 0))
        
        print(f"[ORDER] Test order request: {symbol} {side} {quantity} @ {price}")
        
        if not BINANCE_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Binance integration not available'
            }), 503
        
        if not symbol or not side or quantity <= 0 or price <= 0:
            return jsonify({
                'success': False,
                'error': 'Invalid order parameters'
            }), 400
        
        binance = get_binance_manager()
        order_result = binance.place_test_order(symbol, side, quantity, price)
        
        if not order_result:
            return jsonify({
                'success': False,
                'error': 'Failed to place test order'
            }), 503
        
        print(f"[ORDER] ✓ Test order placed successfully")
        return jsonify({
            'success': True,
            'order': order_result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"[ORDER ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/download/csv', methods=['GET'])
def download_csv() -> Response:
    """Download trades as CSV"""
    try:
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            'Symbol', 'Direction', 'Entry Price', 'Stop Loss',
            'TP1', 'TP2', 'TP3', 'Confluence', 'Accuracy', 'Timestamp'
        ])
        
        # Data
        signals = get_demo_signals() if not DASHBOARD_AVAILABLE else list(signal_cache.values())
        
        for signal in signals:
            if isinstance(signal, dict) and 'data' in signal:
                signal = signal['data']
            
            writer.writerow([
                signal.get('symbol', ''),
                signal.get('direction', ''),
                signal.get('entry_price', ''),
                signal.get('stop_loss', ''),
                signal.get('take_profits', [[0]])[0][0] if signal.get('take_profits') else '',
                signal.get('take_profits', [[0], [0]])[1][0] if len(signal.get('take_profits', [])) > 1 else '',
                signal.get('take_profits', [[0], [0], [0]])[2][0] if len(signal.get('take_profits', [])) > 2 else '',
                signal.get('confluence_score', ''),
                signal.get('accuracy_estimate', ''),
                signal.get('timestamp', datetime.now()).isoformat() if isinstance(signal.get('timestamp'), datetime) else signal.get('timestamp', '')
            ])
        
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/health', methods=['GET'])
def health_check() -> Response:
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'dashboard_available': DASHBOARD_AVAILABLE,
        'cors_available': CORS_AVAILABLE
    })


@app.route('/api/chart/<symbol>', methods=['GET'])
def get_chart(symbol: str) -> Response:
    """Get real Binance candlestick chart data"""
    try:
        # Get and normalize interval (convert 15M to 15m, 5M to 5m, etc)
        interval = request.args.get('interval', '1h').lower()
        limit = request.args.get('limit', 100, type=int)
        
        print(f"[CHART] Request: {symbol} {interval} limit={limit}")
        
        if not BINANCE_AVAILABLE:
            print(f"[CHART] Binance NOT available")
            return jsonify({
                'success': False,
                'error': 'Binance integration not available'
            }), 503
        
        binance = get_binance_manager()
        candles = binance.fetch_klines(symbol, interval, limit)
        
        if not candles:
            print(f"[CHART] No candles for {symbol} {interval}")
            return jsonify({
                'success': False,
                'error': f'No data available for {symbol}'
            }), 404
        
        current_price = candles[-1]['close'] if candles else 0
        
        print(f"[CHART] SUCCESS: {len(candles)} candles for {symbol}")
        return jsonify({
            'success': True,
            'symbol': symbol,
            'interval': interval,
            'candles': candles,
            'current_price': current_price,
            'server_time': int(time.time() * 1000),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[CHART ERROR] {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/price/<symbol>', methods=['GET'])
def get_price(symbol: str) -> Response:
    """Get current price for a symbol from Binance"""
    try:
        # Normalize symbol: BTC -> BTCUSDT, ETH -> ETHUSDT, etc
        symbol = symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        print(f"[PRICE] Request for {symbol}")
        
        if not BINANCE_AVAILABLE:
            print(f"[PRICE] Binance NOT available")
            return jsonify({
                'success': False,
                'error': 'Binance integration not available'
            }), 503
        
        binance = get_binance_manager()
        price = binance.fetch_price(symbol)
        
        if price is None:
            print(f"[PRICE] No price for {symbol}")
            return jsonify({
                'success': False,
                'error': f'No price available for {symbol}'
            }), 404
        
        print(f"[PRICE] SUCCESS: {symbol} = {price}")
        return jsonify({
            'success': True,
            'symbol': symbol,
            'price': price,
            'server_time': int(time.time() * 1000),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[PRICE ERROR] {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats/<symbol>', methods=['GET'])
def get_symbol_stats(symbol: str) -> Response:
    """Get 24h statistics for a symbol from Binance"""
    try:
        # Normalize symbol: BTC -> BTCUSDT, ETH -> ETHUSDT, etc
        symbol = symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        print(f"[STATS] Request for {symbol}")
        
        if not BINANCE_AVAILABLE:
            print(f"[STATS] Binance NOT available")
            return jsonify({
                'success': False,
                'error': 'Binance integration not available'
            }), 503
        
        binance = get_binance_manager()
        stats = binance.fetch_24h_stats(symbol)
        
        if not stats:
            print(f"[STATS] No stats for {symbol}")
            return jsonify({
                'success': False,
                'error': f'No stats available for {symbol}'
            }), 404
        
        print(f"[STATS] SUCCESS: returned stats for {symbol}")
        return jsonify({
            'success': True,
            'symbol': symbol,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[STATS ERROR] {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main function"""
    print("\n" + "="*60)
    print("🚀 CRYPTO TRADING DASHBOARD".center(60))
    print("="*60)
    
    # Check template
    if not TEMPLATES_DIR.exists():
        print(f"⚠️  Templates directory not found: {TEMPLATES_DIR}")
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    
    template_file = TEMPLATES_DIR / 'index.html'
    if template_file.exists():
        print(f"✅ Template found: {template_file}")
    else:
        print(f"⚠️  Template not found: {template_file}")
    
    print(f"\n📊 Dashboard Mode: {'LIVE' if DASHBOARD_AVAILABLE else 'DEMO'}")
    print("\n🌐 Dashboard available at:")
    print("\n   👉 http://localhost:5000")
    print("   👉 http://127.0.0.1:5000")
    print("\n💡 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )


if __name__ == "__main__":
    main()
