#!/usr/bin/env python3
"""Monitor running backtest and generate ML integration report when complete.

Saves:
- reports/ml_integration_report_[timestamp].txt
- reports/ml_signals_[timestamp].csv

Run alongside a running backtest; it will poll the log and DB periodically.
"""
import os
import re
import time
import json
import sqlite3
from datetime import datetime

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(BASE, 'data', 'backtest.db')
LOG_PATH = os.path.join(BASE, 'backtest_output.log')
REPORT_DIR = os.path.join(BASE, 'reports')
ML_THRESHOLD = 0.56


def tail_lines(path, n=400):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()[-n:]
    except Exception:
        return []


def parse_progress(lines):
    prog_re = re.compile(r'Progress:\s*(\d+)\/(\d+)')
    last = None
    for line in reversed(lines):
        m = prog_re.search(line)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def check_errors(lines):
    joined = '\n'.join(lines)
    if 'Traceback' in joined or '\nERROR -' in joined or 'Exception:' in joined:
        return True, joined[-8000:]
    return False, ''


def db_stats(db_path):
    if not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    info = {}
    try:
        cur.execute("PRAGMA table_info(backtest_signals)")
        cols = [r[1] for r in cur.fetchall()]
        info['has_ml'] = 'ml_win_probability' in cols

        cur.execute("SELECT COUNT(*) FROM backtest_signals")
        info['total_signals'] = cur.fetchone()[0]

        if info['has_ml']:
            cur.execute("SELECT COUNT(*), AVG(ml_win_probability), MIN(ml_win_probability), MAX(ml_win_probability) FROM backtest_signals WHERE ml_win_probability IS NOT NULL")
            r = cur.fetchone()
            info['with_ml'] = r[0] or 0
            info['avg_ml'] = r[1] or 0.0
            info['min_ml'] = r[2] or 0.0
            info['max_ml'] = r[3] or 0.0

            cur.execute("SELECT COUNT(*) FROM backtest_signals WHERE ml_win_probability >= ?", (ML_THRESHOLD,))
            info['ml_passed'] = cur.fetchone()[0]
        else:
            info.update({'with_ml':0,'avg_ml':0.0,'min_ml':0.0,'max_ml':0.0,'ml_passed':0})

        # Try to find outcomes table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'outcome%';")
        outcome_tables = [r[0] for r in cur.fetchall()]
        info['outcome_table'] = outcome_tables[0] if outcome_tables else None

    except Exception as e:
        info['error'] = str(e)
    finally:
        conn.close()
    return info


def generate_report_and_csv(db_path, report_dir):
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'ml_integration_report_{ts}.txt')
    csv_path = os.path.join(report_dir, f'ml_signals_{ts}.csv')

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Load signals
    cur.execute("PRAGMA table_info(backtest_signals)")
    cols = [r[1] for r in cur.fetchall()]

    has_ml = 'ml_win_probability' in cols

    # Attempt to join outcomes if present
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'signal_outcome%';")
    outcome_tbl = cur.fetchone()
    outcome_tbl = outcome_tbl[0] if outcome_tbl else None

    join_clause = ''
    select_outcome = ''
    if outcome_tbl:
        join_clause = f" LEFT JOIN {outcome_tbl} o ON s.id = o.signal_id "
        select_outcome = ", o.result AS actual_result, o.pnl AS pnl"

    query = f"SELECT s.id, s.timestamp, s.symbol, s.patterns, s.ml_win_probability{select_outcome} FROM backtest_signals s {join_clause}"

    rows = []
    try:
        for r in cur.execute(query):
            rows.append(r)
    except Exception:
        # Fallback: select without outcome
        rows = list(cur.execute("SELECT id, timestamp, symbol, patterns, ml_win_probability FROM backtest_signals"))

    # Write CSV
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['id','timestamp','symbol','patterns','ml_win_probability','actual_result','pnl'])
        for r in rows:
            # normalize
            if len(r) == 5:
                rid, ts, sym, patterns, ml = r
                actual, pnl = '', ''
            else:
                rid, ts, sym, patterns, ml, actual, pnl = r
            writer.writerow([rid, ts, sym, patterns, ml if ml is not None else '', actual, pnl])

    # Compute report metrics
    total = len(rows)
    ml_scores = [r[4] for r in rows if r[4] is not None]
    ml_passed = [x for x in ml_scores if x >= ML_THRESHOLD]
    avg_ml = sum(ml_scores)/len(ml_scores) if ml_scores else 0
    min_ml = min(ml_scores) if ml_scores else 0
    max_ml = max(ml_scores) if ml_scores else 0

    # Pattern stats
    pattern_counts = {}
    pattern_wins = {}
    outcomes_present = (len(rows) and len(rows[0])>=6)
    for r in rows:
        patterns = r[3]
        try:
            p_list = json.loads(patterns)
        except Exception:
            p_list = []
        actual = None
        pnl = None
        if outcomes_present:
            actual = r[5]
            pnl = r[6]
        for p in p_list:
            pattern_counts[p] = pattern_counts.get(p,0)+1
            if outcomes_present and actual in (1, 'WIN', 'win', 'Win', True):
                pattern_wins[p] = pattern_wins.get(p,0)+1

    top5_by_count = sorted(pattern_counts.items(), key=lambda x:-x[1])[:5]
    top5_by_winrate = []
    for p,c in pattern_counts.items():
        wins = pattern_wins.get(p,0)
        rate = wins/c if c>0 else 0
        top5_by_winrate.append((p, rate, c))
    top5_by_winrate = sorted(top5_by_winrate, key=lambda x:-x[1])[:5]

    # Basic performance metrics if outcome/pnl available
    total_wins = 0
    total_pnl = 0.0
    pnls = []
    if outcomes_present:
        for r in rows:
            actual = r[5]
            pnl = r[6]
            if actual in (1, 'WIN', 'win', 'Win', True):
                total_wins += 1
            try:
                if pnl is not None:
                    total_pnl += float(pnl)
                    pnls.append(float(pnl))
            except Exception:
                pass

    win_rate_all = total_wins/total if total and outcomes_present else None
    win_rate_ml = None
    if outcomes_present and ml_scores:
        # map ids with ml >= threshold
        ml_ids = set([r[0] for r in rows if r[4] is not None and r[4]>=ML_THRESHOLD])
        wins_ml = 0
        total_ml = 0
        for r in rows:
            if r[0] in ml_ids:
                total_ml += 1
                if r[5] in (1, 'WIN', 'win', 'Win', True):
                    wins_ml += 1
        win_rate_ml = wins_ml/total_ml if total_ml else None

    avg_profit = sum(pnls)/len(pnls) if pnls else None
    # simple Sharpe: mean/std; assume risk-free 0 and daily returns not available; use trade-level
    sharpe = (sum(pnls)/len(pnls))/( ( (sum([(x - (sum(pnls)/len(pnls)))**2 for x in pnls])/len(pnls))**0.5) ) if pnls and len(pnls)>1 else None

    # ML effectiveness: correlation between ml prob and wins
    corr = None
    fp_rate = None
    fn_rate = None
    try:
        import numpy as np
        ys = []
        xs = []
        for r in rows:
            ml = r[4]
            actual = None
            if len(r)>=6:
                actual = 1 if r[5] in (1,'WIN','win','Win',True) else 0
            if ml is not None and actual is not None:
                xs.append(ml)
                ys.append(actual)
        if xs and ys:
            corr = float(np.corrcoef(xs, ys)[0,1])
            # false pos/neg
            preds = [1 if x>=ML_THRESHOLD else 0 for x in xs]
            tp = sum(1 for p,a in zip(preds,ys) if p==1 and a==1)
            fp = sum(1 for p,a in zip(preds,ys) if p==1 and a==0)
            tn = sum(1 for p,a in zip(preds,ys) if p==0 and a==0)
            fn = sum(1 for p,a in zip(preds,ys) if p==0 and a==1)
            fp_rate = fp/(fp+tp) if (fp+tp)>0 else None
            fn_rate = fn/(fn+tn) if (fn+tn)>0 else None
    except Exception:
        pass

    # Write report
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write(f"ML Integration Report - {ts} UTC\n")
        fh.write("="*60 + "\n\n")
        fh.write("ML Filtering Stats:\n")
        fh.write(f"- Total signals generated: {total}\n")
        fh.write(f"- Signals with ML score: {len(ml_scores)}\n")
        fh.write(f"- ML-passed signals (>= {ML_THRESHOLD*100:.0f}%): {len(ml_passed)}\n")
        fh.write(f"- ML-rejected signals (< {ML_THRESHOLD*100:.0f}%): {len(ml_scores)-len(ml_passed)}\n")
        fh.write(f"- Average ML win probability: {avg_ml*100:.2f}%\n")
        fh.write(f"- Min ML prob: {min_ml*100:.2f}%\n")
        fh.write(f"- Max ML prob: {max_ml*100:.2f}%\n\n")

        fh.write("Performance Metrics:\n")
        fh.write(f"- Win rate (all): {win_rate_all:.2%}\n" if win_rate_all is not None else "- Win rate (all): N/A\n")
        fh.write(f"- Win rate (ML-passed): {win_rate_ml:.2%}\n" if win_rate_ml is not None else "- Win rate (ML-passed): N/A\n")
        fh.write(f"- Total P&L: {total_pnl:.6f}\n" if pnls else "- Total P&L: N/A\n")
        fh.write(f"- Average profit per trade: {avg_profit:.6f}\n" if avg_profit is not None else "- Average profit per trade: N/A\n")
        fh.write(f"- Sharpe ratio (trade-level): {sharpe:.3f}\n" if sharpe is not None else "- Sharpe ratio: N/A\n")
        fh.write("\nPattern Distribution:\n")
        fh.write("- Top 5 patterns by signal count:\n")
        for p,c in top5_by_count:
            fh.write(f"   - {p}: {c}\n")
        fh.write("- Top 5 patterns by win rate:\n")
        for p,rate,c in top5_by_winrate:
            fh.write(f"   - {p}: {rate:.2%} ({c} signals)\n")

        fh.write("\nML Model Effectiveness:\n")
        fh.write(f"- Correlation (ML prob vs actual wins): {corr:.3f}\n" if corr is not None else "- Correlation: N/A\n")
        fh.write(f"- False positive rate: {fp_rate:.3f}\n" if fp_rate is not None else "- False positive rate: N/A\n")
        fh.write(f"- False negative rate: {fn_rate:.3f}\n" if fn_rate is not None else "- False negative rate: N/A\n")

    conn.close()

    print(f"Report written: {report_path}")
    print(f"CSV written: {csv_path}")
    return report_path, csv_path


def monitor_loop():
    print('Monitoring backtest log and DB. Polling every 30s...')
    last_progress = (0, 1)
    start_time = time.time()
    while True:
        lines = tail_lines(LOG_PATH, 800)
        prog = parse_progress(lines)
        if prog:
            pnum, pden = prog
            percent = pnum / pden * 100
            elapsed = time.time() - start_time
            rate = pnum / elapsed if elapsed>0 else 0
            eta = (pden - pnum) / rate if rate>0 else None
            print(f'Progress: {pnum}/{pden} ({percent:.1f}%) | Signals in DB: {db_stats(DB_PATH).get("total_signals",0)} | ETA: {eta:.0f}s' if eta else f'Progress: {pnum}/{pden} ({percent:.1f}%)')
        else:
            # fallback: show DB counts
            info = db_stats(DB_PATH)
            print(f"Signals in DB: {info.get('total_signals',0)} | ML scored: {info.get('with_ml',0)}")

        err, excerpt = check_errors(lines)
        if err:
            # write immediate alert file
            alert = os.path.join(REPORT_DIR, f'ERROR_alert_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.txt')
            os.makedirs(REPORT_DIR, exist_ok=True)
            with open(alert, 'w', encoding='utf-8') as fh:
                fh.write('Errors detected in backtest log:\n\n')
                fh.write(excerpt)
            print(f'ERROR detected; alert written: {alert}')
            return

        # detect completion: progress equals denominator OR last lines contain 'CALCULATE STATISTICS' or 'Generated comprehensive report' or process stopped
        joined = '\n'.join(lines).lower()
        if prog and prog[0] >= prog[1]:
            print('Progress reached total iterations -> generating report')
            break
        if 'generated comprehensive report' in joined or 'save_report_to_file' in joined or 'calculate statistics' in joined:
            print('Detected completion markers in log -> generating report')
            break

        # also break when no log writes and near completion
        last_mod = os.path.getmtime(LOG_PATH) if os.path.exists(LOG_PATH) else 0
        time.sleep(30)
        new_mod = os.path.getmtime(LOG_PATH) if os.path.exists(LOG_PATH) else 0
        if new_mod == last_mod and prog and prog[0] >= prog[1]-5:
            # likely finished
            print('Log write stalled near completion; generating report')
            break

    # generate report
    report, csv = generate_report_and_csv(DB_PATH, REPORT_DIR)
    print('REPORT COMPLETE')


if __name__ == '__main__':
    monitor_loop()
