"""
Audit Logger: Log trade decisions and trades for audit trail
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class AuditLogger:
    """Logs trade decisions and trades to DB for auditing and compliance."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path(__file__).parent.parent / 'data' / 'crypto_historical.db')
        self._init_db()

    def _init_db(self):
        """Create audit tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                event_type TEXT,
                message TEXT,
                context TEXT,
                user TEXT
            )
        ''')
        
        # Trade decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                signal_strength REAL,
                ml_confidence REAL,
                risk_check_passed INTEGER,
                position_size REAL,
                reason TEXT,
                context TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def log_event(self, event_type: str, message: str, level: str = 'INFO', 
                  context: Optional[Dict[str, Any]] = None, user: str = 'system'):
        """Log an event to audit log."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.utcnow().isoformat()
            context_str = json.dumps(context or {})
            
            cursor.execute('''
                INSERT INTO audit_log (timestamp, level, event_type, message, context, user)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, level, event_type, message, context_str, user))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to log event: {e}")
            return False

    def log_trade_decision(self, symbol: str, action: str, signal_strength: float,
                          ml_confidence: float, risk_check_passed: bool, 
                          position_size: float, reason: str, 
                          context: Optional[Dict[str, Any]] = None):
        """Log a trade decision (buy/sell/hold)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.utcnow().isoformat()
            context_str = json.dumps(context or {})
            risk_passed = 1 if risk_check_passed else 0
            
            cursor.execute('''
                INSERT INTO trade_decisions 
                (timestamp, symbol, action, signal_strength, ml_confidence, 
                 risk_check_passed, position_size, reason, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, symbol, action, signal_strength, ml_confidence, 
                  risk_passed, position_size, reason, context_str))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to log trade decision: {e}")
            return False

    def get_recent_decisions(self, symbol: Optional[str] = None, limit: int = 10) -> list:
        """Retrieve recent trade decisions."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('''
                    SELECT * FROM trade_decisions 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM trade_decisions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception as e:
            print(f"[ERROR] Failed to retrieve decisions: {e}")
            return []

    def get_audit_log(self, event_type: Optional[str] = None, level: Optional[str] = None,
                     limit: int = 50) -> list:
        """Retrieve audit logs with optional filtering."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if event_type and level:
                cursor.execute('''
                    SELECT * FROM audit_log 
                    WHERE event_type = ? AND level = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (event_type, level, limit))
            elif event_type:
                cursor.execute('''
                    SELECT * FROM audit_log 
                    WHERE event_type = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (event_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM audit_log 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception as e:
            print(f"[ERROR] Failed to retrieve audit log: {e}")
            return []


# Global instance
_logger: Optional[AuditLogger] = None


def get_logger(db_path: Optional[str] = None) -> AuditLogger:
    """Get or create the global audit logger."""
    global _logger
    if _logger is None:
        _logger = AuditLogger(db_path)
    return _logger


if __name__ == '__main__':
    logger = AuditLogger()
    
    # Test logging
    logger.log_event('TEST', 'Test event', context={'test': 'data'})
    logger.log_trade_decision(
        symbol='BTC/USDT',
        action='BUY',
        signal_strength=0.85,
        ml_confidence=0.92,
        risk_check_passed=True,
        position_size=0.1,
        reason='Strong uptrend signal + high ML confidence'
    )
    
    print("[OK] Audit logger test passed")
