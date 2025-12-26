#!/usr/bin/env python3
"""
WEB DASHBOARD SERVER
Flask-based real-time dashboard for signal monitoring and performance tracking
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class DashboardDataProvider:
    """Provides data for the dashboard"""
    
    def __init__(self, db_path="data/backtest.db"):
        self.db_path = db_path
    
    def get_live_signals(self):
        """Get currently active signals"""
        if not Path(self.db_path).exists():
            return []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM live_signals 
                    WHERE status IN ('ACTIVE', 'PENDING')
                    ORDER BY entry_time DESC
                    LIMIT 20
                """)
                
                return [dict(row) for row in cursor.fetchall()]
        except:
            return []
    
    def get_performance_metrics(self):
        """Get overall performance metrics"""
        if not Path(self.db_path).exists():
            return self._empty_metrics()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get today's stats
                today = datetime.now().date()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(profit_loss) as total_pnl
                    FROM signal_outcomes
                    WHERE DATE(entry_time) = ?
                """, (today,))
                
                row = cursor.fetchone()
                total, wins, pnl = row if row else (0, 0, 0)
                
                return {
                    'today': {
                        'signals': total or 0,
                        'wins': wins or 0,
                        'win_rate': (wins / total * 100) if total else 0,
                        'pnl': pnl or 0
                    },
                    'all_time': self._get_all_time_stats()
                }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return self._empty_metrics()
    
    def _get_all_time_stats(self):
        """Get all-time statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(profit_loss) as total_pnl,
                        AVG(profit_loss) as avg_pnl
                    FROM signal_outcomes
                """)
                
                row = cursor.fetchone()
                if row:
                    total, wins, pnl, avg = row
                    return {
                        'signals': total or 0,
                        'wins': wins or 0,
                        'win_rate': (wins / total * 100) if total else 0,
                        'pnl': pnl or 0,
                        'avg_pnl': avg or 0
                    }
        except:
            pass
        
        return {
            'signals': 0,
            'wins': 0,
            'win_rate': 0,
            'pnl': 0,
            'avg_pnl': 0
        }
    
    def _empty_metrics(self):
        """Return empty metrics structure"""
        return {
            'today': {
                'signals': 0,
                'wins': 0,
                'win_rate': 0,
                'pnl': 0
            },
            'all_time': {
                'signals': 0,
                'wins': 0,
                'win_rate': 0,
                'pnl': 0,
                'avg_pnl': 0
            }
        }
    
    def get_pattern_breakdown(self):
        """Get win rates by pattern"""
        if not Path(self.db_path).exists():
            return {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        patterns,
                        COUNT(*) as total,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins
                    FROM backtest_signals bs
                    LEFT JOIN signal_outcomes so ON bs.id = so.signal_id
                    WHERE patterns IS NOT NULL
                    GROUP BY patterns
                    ORDER BY total DESC
                    LIMIT 10
                """)
                
                return {
                    row[0]: {
                        'total': row[1],
                        'wins': row[2],
                        'win_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0
                    }
                    for row in cursor.fetchall()
                }
        except:
            return {}
    
    def get_accuracy_by_score(self):
        """Get accuracy statistics by confidence score range"""
        if not Path(self.db_path).exists():
            return {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN confluence_score >= 85 THEN '85+'
                            WHEN confluence_score >= 75 THEN '75-84'
                            WHEN confluence_score >= 65 THEN '65-74'
                            ELSE '<65'
                        END as score_range,
                        COUNT(*) as total,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins
                    FROM backtest_signals bs
                    LEFT JOIN signal_outcomes so ON bs.id = so.signal_id
                    GROUP BY score_range
                    ORDER BY confluence_score DESC
                """)
                
                return {
                    row[0]: {
                        'total': row[1],
                        'wins': row[2],
                        'win_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0
                    }
                    for row in cursor.fetchall()
                }
        except:
            return {}

# Initialize data provider
data_provider = DashboardDataProvider()

# API Routes
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'database': Path('data/backtest.db').exists()
    })

@app.route('/api/signals/live', methods=['GET'])
def get_live_signals():
    """Get live active signals"""
    signals = data_provider.get_live_signals()
    return jsonify({
        'signals': signals,
        'count': len(signals),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics"""
    metrics = data_provider.get_performance_metrics()
    return jsonify(metrics)

@app.route('/api/patterns', methods=['GET'])
def get_patterns():
    """Get pattern breakdown"""
    patterns = data_provider.get_pattern_breakdown()
    return jsonify(patterns)

@app.route('/api/accuracy', methods=['GET'])
def get_accuracy():
    """Get accuracy by score range"""
    accuracy = data_provider.get_accuracy_by_score()
    return jsonify(accuracy)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask app"""
    logger.info(f"\n" + "="*70)
    logger.info("WEB DASHBOARD")
    logger.info("="*70)
    logger.info(f"🌐 Server: http://localhost:{port}")
    logger.info(f"📊 Dashboard: http://localhost:{port}/")
    logger.info(f"📡 API: http://localhost:{port}/api/")
    logger.info("="*70)
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_dashboard(debug=True)
