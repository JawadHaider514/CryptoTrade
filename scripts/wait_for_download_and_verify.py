#!/usr/bin/env python3
import os
import sys
import time
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
from types import ModuleType

# Optional psutil for process checks
psutil: Optional[ModuleType] = None
_HAS_PSUTIL = False
try:
    import psutil as _psutil  # type: ignore
    psutil = _psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

# Configuration - Check BOTH databases
HISTORICAL_DB = Path(__file__).parent.parent / 'data' / 'crypto_historical.db'
BACKTEST_DB = Path(__file__).parent.parent / 'data' / 'backtest.db'
REPORT_DIR = Path(__file__).parent.parent / 'reports'
VERIFY_SCRIPT = Path(__file__).parent.parent / 'scripts' / 'verify_data_quality.py'

# Polling settings
POLL_INTERVAL = 30
STABLE_CHECKS = 5
STABLE_THRESHOLD = 60


def get_candle_count():
    """Get current historical candle count and symbol count"""
    try:
        if not HISTORICAL_DB.exists():
            return 0, "DB not found"
        
        conn = sqlite3.connect(str(HISTORICAL_DB))
        cursor = conn.cursor()
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='historical_klines'
        """)
        if not cursor.fetchone():
            conn.close()
            return 0, "Table not created yet"
        
        # Get count
        cursor.execute("SELECT COUNT(*) FROM historical_klines")
        count = cursor.fetchone()[0]
        
        # Get symbol count
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM historical_klines")
        symbols = cursor.fetchone()[0]
        
        conn.close()
        return count, f"{symbols} symbols"
    except Exception as e:
        return 0, f"Error: {e}"


def get_db_age(db_path: Path):
    """Get seconds since database was last modified"""
    try:
        if not db_path.exists():
            return float('inf')
        mtime = db_path.stat().st_mtime
        return time.time() - mtime
    except Exception:
        return float('inf')


def check_download_status():
    """Check if download_historical_data.py process is running (optional psutil)"""
    if not _HAS_PSUTIL or psutil is None:
        return None, None
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'download_historical_data.py' in ' '.join(cmdline):
                    return True, proc.info.get('pid')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False, None
    except Exception:
        return None, None


def run_verify_and_save():
    """Run verification script and save output"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = REPORT_DIR / f'verify_download_{ts}.txt'
    
    print(f"\n🔍 Running verification...")
    print(f"📄 Report will be saved to: {report_path}")
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(
            [sys.executable, str(VERIFY_SCRIPT)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300,
            env=env
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DATA VERIFICATION REPORT - {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("STDOUT:\n")
            f.write("-" * 80 + "\n")
            f.write(result.stdout)
            f.write("\n\n")
            
            if result.stderr:
                f.write("STDERR:\n")
                f.write("-" * 80 + "\n")
                f.write(result.stderr)
                f.write("\n\n")
            
            f.write(f"Exit Code: {result.returncode}\n")
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Verification complete! Report: {report_path}")
        else:
            print(f"\n⚠️  Check report: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"ERROR: {e}\n")
        return False


def monitor_download():
    """Monitor database for download completion"""
    print("🔍 Watching for historical data download completion...")
    print(f"📁 Historical DB: {HISTORICAL_DB}")
    print(f"📁 Backtest DB: {BACKTEST_DB}")
    print("=" * 70)
    
    last_count = 0
    stable_count = 0
    
    while True:
        current_count, status = get_candle_count()
        db_age = get_db_age(HISTORICAL_DB)
        is_running, pid = check_download_status()
        
        # Status message
        status_msg = f"📊 Candles: {current_count:,} ({status})"
        
        if is_running is not None:
            if is_running:
                status_msg += f" | 🔄 Download running (PID: {pid})"
            else:
                status_msg += f" | ⏸️  Download process not found"
        
        # Check if count has changed
        if current_count != last_count and current_count > 0:
            stable_count = 0
            last_count = current_count
            print(f"{status_msg} | DB age: {db_age:.0f}s")
        else:
            stable_count += 1
            print(f"{status_msg} | Stable: {stable_count}/{STABLE_CHECKS} | Age: {db_age:.0f}s")
        
        # Download complete if:
        # 1. No process running (if we can check)
        # 2. Database stable for STABLE_CHECKS iterations
        # 3. Database not modified for STABLE_THRESHOLD seconds
        # 4. We have some data
        
        process_stopped = (is_running == False) if (is_running is not None) else True
        
        if (stable_count >= STABLE_CHECKS and 
            db_age > STABLE_THRESHOLD and 
            current_count > 0 and
            process_stopped):
            print(f"\n✅ Download appears complete!")
            print(f"   - Total candles: {current_count:,}")
            print(f"   - Status: {status}")
            print(f"   - DB stable for {stable_count * POLL_INTERVAL}s")
            break
        
        # Warning if stuck at 0
        if current_count == 0 and stable_count > 10:
            print(f"\n⚠️  WARNING: No data detected after {stable_count * POLL_INTERVAL}s")
            print(f"   Check if download_historical_data.py is running!")
        
        time.sleep(POLL_INTERVAL)
    
    # Run verification
    print("\n" + "=" * 70)
    success = run_verify_and_save()
    
    if success:
        print("\n🎉 Monitoring and verification complete!")
    else:
        print("\n⚠️  Verification had issues")
    
    return success


if __name__ == "__main__":
    try:
        monitor_download()
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
