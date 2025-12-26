#!/usr/bin/env python3
"""
ADVANCED CRYPTO TRADING BOT WEB SERVER
Complete API with real-time updates, bot control, and live analytics
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import os
import sys
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import queue
import csv
from io import StringIO, BytesIO
import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import dashboard and bot components
try:
    from enhanced_crypto_dashboard import (
        EnhancedScalpingDashboard,
        DemoTradingBot,
        BinanceStreamingAPI,
        StreamingSignalProcessor,
        PredictionMetrics,
        EnhancedSignal,
        SignalQuality,
        ScalpingConfig,
        SignalFormatter
    )
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    DASHBOARD_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')
CORS(app, supports_credentials=True)

# Discord webhook configuration
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
if not DISCORD_WEBHOOK_URL:
    logger.warning('⚠️ DISCORD_WEBHOOK_URL not set. Discord notifications disabled.')

# Handle OPTIONS requests for CORS preflight
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Private-Network'] = 'true'
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Access-Control-Request-Private-Network'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response

# Add Private-Network-Access headers to all responses
@app.after_request
def add_private_network_headers(response):
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Access-Control-Request-Private-Network'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# Initialize SocketIO for real-time updates
# Use polling as fallback for development, disable WebSocket upgrade issues
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    engineio_logger=False,
    logger=False
)

# Global state
bot_state = {
    'running': False,
    'dashboard': None,
    'bot': None,
    'thread': None,
    'signals': [],
    'portfolio_history': [],
    'trade_history': [],
    'start_time': None,
    'stats': {
        'total_signals': 0,
        'total_trades': 0,
        'active_trades': 0,
        'pnl': 0.0,
        'pnl_percent': 0.0,
        'win_rate': 0.0,
        'max_drawdown': 0.0
    }
}

# Update queue for thread-safe communication
update_queue = queue.Queue()


def init_bot():
    """Initialize dashboard and bot"""
    if bot_state['dashboard'] is None:
        try:
            if DASHBOARD_AVAILABLE:
                bot_state['dashboard'] = EnhancedScalpingDashboard(  # type: ignore
                    use_streaming_ml=True,
                    enable_demo_bot=True,
                    use_binance_testnet=False
                )
                logger.info("✅ Dashboard initialized")
                return True
            else:
                logger.error("❌ Dashboard modules not available")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize dashboard: {e}")
            return False
    return True


def run_bot_loop():
    """Main bot loop - runs in separate thread"""
    bot_state['running'] = True
    bot_state['start_time'] = datetime.now()
    iteration = 0

    while bot_state['running']:
        iteration += 1
        
        try:
            # Generate signals
            dashboard = bot_state['dashboard']
            if dashboard and dashboard.demo_bot:
                # Generate signals from streaming processor
                signals = []
                if hasattr(dashboard, 'streaming_processor') and dashboard.streaming_processor:
                    try:
                        signals = dashboard.streaming_processor.process_symbols_batch()
                    except Exception as e:
                        logger.error(f"❌ Error processing symbols: {e}")
                        signals = []
                
                # Store signals
                bot_state['signals'] = [
                    {
                        'symbol': s.symbol,
                        'direction': s.direction,
                        'confidence': s.confidence,
                        'entry_price': s.entry_price,
                        'stop_loss': s.stop_loss,
                        'tp1': s.take_profit_1,
                        'tp2': s.take_profit_2,
                        'tp3': s.take_profit_3,
                        'timestamp': s.timestamp.isoformat() if hasattr(s, 'timestamp') else datetime.now().isoformat(),
                        'quality': s.quality.value if hasattr(s, 'quality') else 'MEDIUM'
                    }
                    for s in signals
                ]
                
                # Log signal count
                if bot_state['signals']:
                    logger.info(f"📊 {len(bot_state['signals'])} signals generated: {[s['symbol'] for s in bot_state['signals'][:3]]}")
                
                # Update bot trades
                if hasattr(dashboard, 'update_bot_trades'):
                    dashboard.update_bot_trades()
                
                # Update bot stats
                if dashboard.demo_bot:
                    bot_state['stats'] = {
                        'total_signals': bot_state['stats']['total_signals'] + len(signals),
                        'total_trades': dashboard.demo_bot.portfolio.total_trades,
                        'active_trades': len(dashboard.demo_bot.active_trades),
                        'pnl': dashboard.demo_bot.portfolio.total_pnl,
                        'pnl_percent': (dashboard.demo_bot.portfolio.total_pnl / dashboard.demo_bot.portfolio.initial_balance * 100) if dashboard.demo_bot.portfolio.initial_balance > 0 else 0,
                        'win_rate': (dashboard.demo_bot.portfolio.winning_trades / dashboard.demo_bot.portfolio.total_trades * 100) if dashboard.demo_bot.portfolio.total_trades > 0 else 0,
                        'max_drawdown': dashboard.demo_bot.portfolio.max_drawdown
                    }
                    
                    # Store portfolio history point
                    bot_state['portfolio_history'].append({
                        'timestamp': datetime.now().isoformat(),
                        'equity': dashboard.demo_bot.portfolio.equity,
                        'pnl': dashboard.demo_bot.portfolio.total_pnl,
                        'trades': dashboard.demo_bot.portfolio.total_trades
                    })
                
                # Emit updates via WebSocket to ALL clients
                socketio.emit('bot_update', {
                    'signals': bot_state['signals'][:5],
                    'stats': bot_state['stats'],
                    'portfolio_history': bot_state['portfolio_history'][-100:],  # Last 100 points
                    'active_trades': len(dashboard.demo_bot.active_trades) if dashboard.demo_bot else 0,
                    'running': bot_state['running'],
                    'timestamp': datetime.now().isoformat()
                }, broadcast=True)
                
                logger.info(f"🔄 Iteration {iteration}: {len(signals)} signals, {bot_state['stats']['total_trades']} trades")
        
        except Exception as e:
            logger.error(f"❌ Bot loop error: {e}")
        
        # Wait before next iteration
        time.sleep(30)


# ============================================================================
# WEB ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve main dashboard"""
    return send_from_directory('frontend/dist', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files and fallback to index.html for SPA routing"""
    try:
        return send_from_directory('frontend/dist', path)
    except:
        return send_from_directory('frontend/dist', 'index.html')


@app.route('/api/bot/status')
def get_bot_status():
    """Get current bot status"""
    return jsonify({
        'running': bot_state['running'],
        'start_time': bot_state['start_time'].isoformat() if bot_state['start_time'] else None,
        'uptime_seconds': (datetime.now() - bot_state['start_time']).total_seconds() if bot_state['start_time'] else 0,
        'stats': bot_state['stats'],
        'signals_count': len(bot_state['signals']),
        'portfolio_history_points': len(bot_state['portfolio_history'])
    })


@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    if bot_state['running']:
        return jsonify({'error': 'Bot is already running'}), 400
    
    try:
        if not init_bot():
            return jsonify({'error': 'Failed to initialize bot'}), 500
        
        # Set running flag BEFORE starting thread
        bot_state['running'] = True
        bot_state['start_time'] = datetime.now()
        
        # Start bot thread
        bot_state['thread'] = threading.Thread(target=run_bot_loop, daemon=True)
        bot_state['thread'].start()
        
        # Emit event to all connected clients
        socketio.emit('bot_started', {'timestamp': datetime.now().isoformat()}, broadcast=True)
        
        logger.info("✅ Bot started")
        return jsonify({'success': True, 'message': 'Bot started successfully'})
    
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}")
        bot_state['running'] = False
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get all current signals from bot"""
    try:
        # Get fresh signals if bot is running
        if bot_state['running'] and bot_state['dashboard']:
            dashboard = bot_state['dashboard']
            
            # Generate fresh signals
            if hasattr(dashboard, 'streaming_processor') and dashboard.streaming_processor:
                try:
                    fresh_signals = dashboard.streaming_processor.process_symbols_batch()
                    bot_state['signals'] = [
                        {
                            'symbol': s.symbol,
                            'direction': s.direction,
                            'confidence': s.confidence,
                            'entry_price': s.entry_price,
                            'stop_loss': s.stop_loss,
                            'tp1': s.take_profit_1,
                            'tp2': s.take_profit_2,
                            'tp3': s.take_profit_3,
                            'timestamp': s.timestamp.isoformat() if hasattr(s, 'timestamp') else datetime.now().isoformat(),
                            'quality': s.quality.value if hasattr(s, 'quality') else 'MEDIUM'
                        }
                        for s in fresh_signals
                    ]
                except Exception as e:
                    logger.error(f"❌ Error generating signals: {e}")
        
        return jsonify({
            'success': True,
            'signals': bot_state['signals'],
            'bot_running': bot_state['running'],
            'stats': bot_state['stats'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in get_signals: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    if not bot_state['running']:
        return jsonify({'error': 'Bot is not running'}), 400
    
    try:
        bot_state['running'] = False
        
        # Wait for thread to finish
        if bot_state['thread']:
            bot_state['thread'].join(timeout=5)
        
        logger.info("✅ Bot stopped")
        return jsonify({'success': True, 'message': 'Bot stopped successfully'})
    
    except Exception as e:
        logger.error(f"❌ Failed to stop bot: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/history')
def get_portfolio_history():
    """Get portfolio history for charting"""
    # Return last 100 points
    history = bot_state['portfolio_history'][-100:] if len(bot_state['portfolio_history']) > 100 else bot_state['portfolio_history']
    
    return jsonify({
        'history': history,
        'current_equity': bot_state['stats']['pnl'] if bot_state['dashboard'] and bot_state['dashboard'].demo_bot else 0
    })


@app.route('/api/trades')
def get_trades():
    """Get trade history"""
    trades = []
    
    if bot_state['dashboard'] and bot_state['dashboard'].demo_bot:
        # Get completed trades from bot
        bot = bot_state['dashboard'].demo_bot
        trades = [
            {
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.current_price,
                'pnl': t.pnl,
                'pnl_percentage': t.pnl_percentage,
                'status': t.status.value if hasattr(t, 'status') else 'COMPLETED',
                'entry_time': t.entry_time.isoformat() if hasattr(t, 'entry_time') else '',
                'exit_time': t.exit_time.isoformat() if hasattr(t, 'exit_time') and t.exit_time else ''
            }
            for t in getattr(bot, 'completed_trades', [])
        ]
    
    return jsonify({
        'trades': trades,
        'count': len(trades)
    })


@app.route('/api/stats')
def get_stats():
    """Get detailed statistics"""
    return jsonify({
        'stats': bot_state['stats'],
        'portfolio_points': len(bot_state['portfolio_history']),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/statistics')
def get_statistics():
    """Get statistics - alias for /api/stats to match HTML expectations"""
    trades = []
    total_pnl = 0
    winning_trades = 0
    
    if bot_state['dashboard'] and bot_state['dashboard'].demo_bot:
        bot = bot_state['dashboard'].demo_bot
        trades = getattr(bot, 'completed_trades', [])
        winning_trades = sum(1 for t in trades if hasattr(t, 'pnl') and t.pnl > 0)
        total_pnl = sum(getattr(t, 'pnl', 0) for t in trades)
    
    win_rate = (winning_trades / len(trades) * 100) if len(trades) > 0 else 0
    
    return jsonify({
        'success': True,
        'statistics': {
            'win_rate': win_rate,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': len(trades) - winning_trades,
            'total_pnl': total_pnl,
            'bot_running': bot_state['running'],
            'uptime_seconds': (datetime.now() - bot_state['start_time']).total_seconds() if bot_state['start_time'] else 0
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/coins')
def get_coins():
    """Get coin data with signals"""
    coins = []
    
    if bot_state['signals']:
        for signal in bot_state['signals']:
            coins.append({
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'confidence': signal['confidence'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'tp1': signal['tp1'],
                'tp2': signal['tp2'],
                'tp3': signal['tp3'],
                'quality': signal['quality'],
                'timestamp': signal['timestamp']
            })
    
    return jsonify({
        'coins': coins,
        'count': len(coins)
    })


@app.route('/api/discord-notify', methods=['POST'])
def discord_notify():
    """Send signal notification to Discord"""
    if not DISCORD_WEBHOOK_URL:
        return jsonify({
            'success': False,
            'error': 'Discord webhook not configured'
        }), 400
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'UNKNOWN')
        direction = data.get('direction', 'N/A')
        entry_price = data.get('entry_price', 0)
        stop_loss = data.get('stop_loss', 0)
        take_profits = data.get('take_profits', [])
        confluence_score = data.get('confluence_score', 0)
        
        # Format take profits
        tp_text = ""
        if isinstance(take_profits, list):
            for i, tp in enumerate(take_profits[:3], 1):
                if isinstance(tp, (list, tuple)):
                    tp_text += f"TP{i}: ${tp[0]:.2f}\n"
                else:
                    tp_text += f"TP{i}: ${tp:.2f}\n"
        
        # Create Discord embed
        embed = {
            "title": f"🎯 {symbol} - {direction}",
            "description": f"**Confluence Score:** {confluence_score}/100",
            "color": 65280 if direction == "LONG" else 16711680,  # Green for LONG, Red for SHORT
            "fields": [
                {"name": "📍 Entry Price", "value": f"${entry_price:.2f}", "inline": True},
                {"name": "🛑 Stop Loss", "value": f"${stop_loss:.2f}", "inline": True},
                {"name": "🎯 Take Profits", "value": tp_text or "N/A", "inline": False},
                {"name": "⏰ Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True}
            ],
            "footer": {"text": "CryptoTrader Pro Dashboard"}
        }
        
        # Send to Discord
        payload = {"embeds": [embed]}
        headers = {"Content-Type": "application/json"}
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, headers=headers, timeout=5)
        
        if response.status_code == 204:
            logger.info(f"✅ Discord notification sent for {symbol}")
            return jsonify({
                'success': True,
                'message': f'Signal {symbol} sent to Discord'
            })
        else:
            logger.error(f"❌ Discord API error: {response.status_code}")
            return jsonify({
                'success': False,
                'error': f'Discord API error: {response.status_code}'
            }), 500
    
    except Exception as e:
        logger.error(f"❌ Discord notification error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/download/csv', methods=['GET'])
def download_csv():
    """Download trades as CSV file"""
    try:
        trades = []
        
        if bot_state['dashboard'] and bot_state['dashboard'].demo_bot:
            bot = bot_state['dashboard'].demo_bot
            trades = getattr(bot, 'completed_trades', [])
        
        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Symbol',
            'Direction',
            'Entry Price',
            'Exit Price',
            'PnL',
            'PnL %',
            'Status',
            'Entry Time',
            'Exit Time',
            'Duration'
        ])
        
        # Write trade rows
        for trade in trades:
            entry_time = getattr(trade, 'entry_time', datetime.now())
            exit_time = getattr(trade, 'exit_time', datetime.now())
            
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time)
                except:
                    entry_time = datetime.now()
            
            if isinstance(exit_time, str):
                try:
                    exit_time = datetime.fromisoformat(exit_time)
                except:
                    exit_time = datetime.now()
            
            duration = exit_time - entry_time if exit_time and entry_time else timedelta(0)
            duration_str = str(duration).split('.')[0]  # Remove microseconds
            
            writer.writerow([
                getattr(trade, 'symbol', 'N/A'),
                getattr(trade, 'direction', 'N/A'),
                f"${getattr(trade, 'entry_price', 0):.2f}",
                f"${getattr(trade, 'current_price', 0):.2f}",
                f"${getattr(trade, 'pnl', 0):.2f}",
                f"{getattr(trade, 'pnl_percentage', 0):.2f}%",
                getattr(trade, 'status', 'UNKNOWN'),
                entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                exit_time.strftime('%Y-%m-%d %H:%M:%S') if exit_time else '',
                duration_str
            ])
        
        # Create response
        output.seek(0)
        csv_bytes = BytesIO(output.getvalue().encode('utf-8'))
        
        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    except Exception as e:
        logger.error(f"❌ CSV export error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    sid = request.sid if hasattr(request, 'sid') else 'unknown'  # type: ignore
    logger.info(f"🔌 Client connected: {sid}")
    emit('connection_response', {
        'data': 'Connected to trading bot server',
        'timestamp': datetime.now().isoformat()
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    sid = request.sid if hasattr(request, 'sid') else 'unknown'  # type: ignore
    logger.info(f"🔌 Client disconnected: {sid}")


@socketio.on('request_status')
def handle_status_request():
    """Client requests bot status"""
    emit('bot_status', {
        'running': bot_state['running'],
        'stats': bot_state['stats'],
        'timestamp': datetime.now().isoformat()
    })


@socketio.on('start_bot')
def handle_start_bot():
    """Client requests bot start"""
    if not bot_state['running']:
        if init_bot():
            bot_state['thread'] = threading.Thread(target=run_bot_loop, daemon=True)
            bot_state['thread'].start()
            emit('bot_started', {'timestamp': datetime.now().isoformat()}, broadcast=True)
            logger.info("✅ Bot started via WebSocket")
    else:
        emit('error', {'message': 'Bot already running'})


@socketio.on('stop_bot')
def handle_stop_bot():
    """Client requests bot stop"""
    if bot_state['running']:
        bot_state['running'] = False
        if bot_state['thread']:
            bot_state['thread'].join(timeout=5)
        emit('bot_stopped', {'timestamp': datetime.now().isoformat()}, broadcast=True)
        logger.info("✅ Bot stopped via WebSocket")
    else:
        emit('error', {'message': 'Bot is not running'})


# ============================================================================
# 404 HANDLER - Already handled by serve_static/<path:path>
# ============================================================================


@app.errorhandler(404)
def not_found(e):
    """Serve index.html for all other routes (SPA routing)"""
    return send_from_directory('frontend/public', 'index.html')


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(Exception)
def handle_error(error):
    """Handle all errors"""
    logger.error(f"❌ Server error: {error}")
    return jsonify({'error': str(error)}), 500


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Start the web server"""
    print("\n" + "="*80)
    print("🚀 ADVANCED CRYPTO TRADING BOT - WEB SERVER".center(80))
    print("="*80)
    print("\n📊 Dashboard: http://localhost:5000")
    print("🔌 WebSocket: ws://localhost:5000/socket.io")
    print("\n✅ Server starting...")
    print("="*80 + "\n")
    
    # Initialize bot
    if not init_bot():
        print("❌ Failed to initialize bot")
        sys.exit(1)
    
    try:
        # Start server
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            log_output=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        bot_state['running'] = False
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
