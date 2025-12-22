"""
Health Monitor: Basic system health checks and alerts
Checks DB health, data freshness, and sends alerts via Discord (opt-in)
"""

import os
import sys
import sqlite3
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class HealthCheck:
    """Health check result."""
    
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.timestamp = datetime.utcnow()

    def __str__(self):
        status = "[OK]" if self.passed else "[FAIL]"
        return f"{status} {self.name}: {self.message}"


class HealthMonitor:
    """System health monitor with optional Discord alerts."""
    
    def __init__(self, db_path: Optional[str] = None, discord_webhook: Optional[str] = None):
        """
        Initialize health monitor.
        
        Args:
            db_path: Path to historical data DB
            discord_webhook: Discord webhook URL for alerts (optional)
        """
        self.db_path = db_path or str(Path(__file__).parent.parent / 'data' / 'crypto_historical.db')
        self.discord_webhook = discord_webhook or os.getenv('DISCORD_WEBHOOK_URL')
        self.checks: List[HealthCheck] = []

    def check_db_health(self) -> HealthCheck:
        """Check if DB is accessible and valid."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM crypto_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            message = f"{count} records in DB"
            passed = count > 0
            return HealthCheck("DB Health", passed, message)
        except Exception as e:
            return HealthCheck("DB Health", False, str(e))

    def check_data_freshness(self, max_age_hours: int = 2) -> HealthCheck:
        """Check if data is recent (not stale)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(open_time) FROM crypto_data")
            result = cursor.fetchone()[0]
            conn.close()
            
            if not result:
                return HealthCheck("Data Freshness", False, "No data in DB")
            
            # Convert from ms to datetime
            last_update = datetime.fromtimestamp(result / 1000.0)
            age = datetime.utcnow() - last_update
            
            passed = age < timedelta(hours=max_age_hours)
            message = f"Last update: {age.seconds // 60} minutes ago"
            return HealthCheck("Data Freshness", passed, message)
        except Exception as e:
            return HealthCheck("Data Freshness", False, str(e))

    def check_disk_space(self, min_mb: int = 100) -> HealthCheck:
        """Check if sufficient disk space is available."""
        try:
            import shutil
            stat = shutil.disk_usage('/')
            available_mb = stat.free / (1024 * 1024)
            
            passed = available_mb > min_mb
            message = f"{available_mb:.1f} MB available"
            return HealthCheck("Disk Space", passed, message)
        except Exception as e:
            return HealthCheck("Disk Space", False, str(e))

    def check_api_connectivity(self) -> HealthCheck:
        """Check if we can reach external APIs (Binance)."""
        try:
            response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
            passed = response.status_code == 200
            message = f"Response: {response.status_code}"
            return HealthCheck("API Connectivity", passed, message)
        except Exception as e:
            return HealthCheck("API Connectivity", False, str(e))

    def run_all_checks(self) -> List[HealthCheck]:
        """Run all health checks."""
        self.checks = [
            self.check_db_health(),
            self.check_data_freshness(),
            self.check_disk_space(),
            self.check_api_connectivity(),
        ]
        return self.checks

    def send_alert(self, message: str):
        """Send alert to Discord if webhook is configured."""
        if not self.discord_webhook:
            return False
        
        try:
            payload = {
                'content': message,
                'username': 'HealthMonitor'
            }
            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            return response.status_code == 204
        except Exception as e:
            print(f"[WARN] Failed to send Discord alert: {e}")
            return False

    def report(self) -> str:
        """Generate a report of all checks."""
        lines = [
            "=" * 70,
            f"HEALTH REPORT - {datetime.utcnow().isoformat()}",
            "=" * 70,
        ]
        
        passed_count = 0
        for check in self.checks:
            lines.append(str(check))
            if check.passed:
                passed_count += 1
        
        summary = f"\nSummary: {passed_count}/{len(self.checks)} checks passed"
        lines.append(summary)
        
        return "\n".join(lines)

    def health_status(self) -> bool:
        """Return True if all checks passed."""
        return all(check.passed for check in self.checks)


def main():
    """CLI entry point for health monitor."""
    monitor = HealthMonitor()
    
    print("[*] Running health checks...")
    checks = monitor.run_all_checks()
    
    report = monitor.report()
    print(report)
    
    # Send alert if any checks failed
    if not monitor.health_status():
        alert_msg = f"[ALERT] Health check failed!\n{report}"
        if monitor.send_alert(alert_msg):
            print("\n[OK] Alert sent to Discord")
        else:
            print("\n[WARN] No Discord webhook configured")
    else:
        print("\n[OK] All health checks passed!")


if __name__ == '__main__':
    main()
