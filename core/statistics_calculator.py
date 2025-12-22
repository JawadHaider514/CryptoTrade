#!/usr/bin/env python3
"""
TASK 1.4: Statistics Calculator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calculate REAL accuracy metrics from backtesting.
Show actual performance by confluence score, pattern, etc.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import json
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestStatisticsCalculator:
    """Calculate comprehensive backtesting statistics"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        """Initialize statistics calculator"""
        self.db_path = db_path
    
    def calculate_overall_stats(self, symbol: str) -> Dict:
        """Calculate overall performance statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all outcomes
                df = pd.read_sql_query("""
                    SELECT * FROM signal_outcomes
                    WHERE symbol = ?
                """, conn, params=(symbol,))
            
            if df.empty:
                return {'error': 'No outcomes found'}
            
            total = len(df)
            wins = len(df[df['result'] == 'WIN'])
            losses = len(df[df['result'] == 'LOSS'])
            timeouts = len(df[df['result'] == 'TIMEOUT'])
            
            win_pnl = df[df['result'] == 'WIN']['pnl_dollars'].sum()
            loss_pnl = df[df['result'] == 'LOSS']['pnl_dollars'].sum()
            timeout_pnl = df[df['result'] == 'TIMEOUT']['pnl_dollars'].sum()
            
            stats = {
                'total_signals': total,
                'wins': wins,
                'losses': losses,
                'timeouts': timeouts,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'loss_rate': (losses / total * 100) if total > 0 else 0,
                'timeout_rate': (timeouts / total * 100) if total > 0 else 0,
                'total_profit': win_pnl + timeout_pnl if timeout_pnl > 0 else win_pnl,
                'total_loss': loss_pnl,
                'net_profit': (win_pnl + timeout_pnl + loss_pnl) if timeout_pnl else (win_pnl + loss_pnl),
                'profit_factor': abs(win_pnl / loss_pnl) if loss_pnl != 0 else (1.0 if win_pnl > 0 else 0.0),
                'avg_win': (win_pnl / wins) if wins > 0 else 0,
                'avg_loss': abs(loss_pnl / losses) if losses > 0 else 0,
                'avg_pnl_percentage': df['pnl_percentage'].mean(),
                'best_trade': df['pnl_dollars'].max(),
                'worst_trade': df['pnl_dollars'].min(),
                'avg_time_in_trade_seconds': df['time_in_trade_seconds'].mean()
            }
            
            # Calculate expectancy
            if wins > 0 and losses > 0:
                avg_win_pct = df[df['result'] == 'WIN']['pnl_percentage'].mean()
                avg_loss_pct = abs(df[df['result'] == 'LOSS']['pnl_percentage'].mean())
                win_rate = wins / total
                loss_rate = losses / total
                
                stats['expectancy'] = (avg_win_pct * win_rate) - (avg_loss_pct * loss_rate)
            else:
                stats['expectancy'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating overall stats: {e}")
            return {}
    
    def calculate_accuracy_by_confluence_score(self, symbol: str) -> Dict:
        """Calculate accuracy for different confluence score ranges"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get signals with outcomes joined
                df = pd.read_sql_query("""
                    SELECT so.*, bs.confluence_score
                    FROM signal_outcomes so
                    JOIN backtest_signals bs ON so.signal_id = bs.id
                    WHERE so.symbol = ?
                """, conn, params=(symbol,))
            
            if df.empty:
                return {}
            
            results = {}
            
            # Define score ranges
            ranges = [
                ('85+', 85, 9999),
                ('75-84', 75, 84),
                ('65-74', 65, 74),
                ('<65', 0, 64)
            ]
            
            for range_name, min_score, max_score in ranges:
                range_df = df[(df['confluence_score'] >= min_score) & 
                             (df['confluence_score'] <= max_score)]
                
                if range_df.empty:
                    continue
                
                total = len(range_df)
                wins = len(range_df[range_df['result'] == 'WIN'])
                losses = len(range_df[range_df['result'] == 'LOSS'])
                
                win_pnl = range_df[range_df['result'] == 'WIN']['pnl_dollars'].sum()
                loss_pnl = range_df[range_df['result'] == 'LOSS']['pnl_dollars'].sum()
                
                results[range_name] = {
                    'signals': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'total_pnl': win_pnl + loss_pnl,
                    'avg_pnl': (win_pnl + loss_pnl) / total if total > 0 else 0,
                    'profit_factor': abs(win_pnl / loss_pnl) if loss_pnl != 0 else 1.0
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating accuracy by score: {e}")
            return {}
    
    def calculate_accuracy_by_pattern(self, symbol: str) -> Dict:
        """Calculate accuracy for different candlestick patterns"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get signals with outcomes
                df = pd.read_sql_query("""
                    SELECT so.*, bs.patterns
                    FROM signal_outcomes so
                    JOIN backtest_signals bs ON so.signal_id = bs.id
                    WHERE so.symbol = ?
                """, conn, params=(symbol,))
            
            if df.empty:
                return {}
            
            # Parse patterns JSON
            df['patterns'] = df['patterns'].apply(json.loads)
            
            results = {}
            
            # Explode patterns so each pattern is a separate row
            patterns_df = df.explode('patterns')
            
            # Calculate stats for each pattern
            for pattern in patterns_df['patterns'].unique():
                pattern_data = patterns_df[patterns_df['patterns'] == pattern]
                
                total = len(pattern_data)
                wins = len(pattern_data[pattern_data['result'] == 'WIN'])
                losses = len(pattern_data[pattern_data['result'] == 'LOSS'])
                
                win_pnl = pattern_data[pattern_data['result'] == 'WIN']['pnl_dollars'].sum()
                loss_pnl = pattern_data[pattern_data['result'] == 'LOSS']['pnl_dollars'].sum()
                
                results[pattern] = {
                    'signals': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'total_pnl': win_pnl + loss_pnl,
                    'avg_pnl': (win_pnl + loss_pnl) / total if total > 0 else 0
                }
            
            # Sort by win rate
            sorted_results = dict(sorted(results.items(), 
                                        key=lambda x: x[1]['win_rate'], 
                                        reverse=True))
            
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error calculating accuracy by pattern: {e}")
            return {}
    
    def calculate_accuracy_by_direction(self, symbol: str) -> Dict:
        """Calculate accuracy for LONG vs SHORT trades"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM signal_outcomes
                    WHERE symbol = ?
                """, conn, params=(symbol,))
            
            if df.empty:
                return {}
            
            results = {}
            
            for direction in ['LONG', 'SHORT']:
                dir_df = df[df['direction'] == direction]
                
                if dir_df.empty:
                    continue
                
                total = len(dir_df)
                wins = len(dir_df[dir_df['result'] == 'WIN'])
                losses = len(dir_df[dir_df['result'] == 'LOSS'])
                
                win_pnl = dir_df[dir_df['result'] == 'WIN']['pnl_dollars'].sum()
                loss_pnl = dir_df[dir_df['result'] == 'LOSS']['pnl_dollars'].sum()
                
                results[direction] = {
                    'signals': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'total_pnl': win_pnl + loss_pnl,
                    'avg_pnl': (win_pnl + loss_pnl) / total if total > 0 else 0
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating accuracy by direction: {e}")
            return {}
    
    def calculate_accuracy_by_hour(self, symbol: str) -> Dict:
        """Calculate accuracy by hour of day"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM signal_outcomes
                    WHERE symbol = ?
                """, conn, params=(symbol,))
            
            if df.empty:
                return {}
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'] / 1000, unit='s')
            # Use DatetimeIndex to avoid pandas/typing issues with .dt
            df['hour'] = pd.DatetimeIndex(df['datetime']).hour
            
            results = {}
            
            for hour in range(24):
                hour_df = df[df['hour'] == hour]
                
                if hour_df.empty:
                    continue
                
                total = len(hour_df)
                wins = len(hour_df[hour_df['result'] == 'WIN'])
                
                win_pnl = hour_df[hour_df['result'] == 'WIN']['pnl_dollars'].sum()
                loss_pnl = hour_df[hour_df['result'] == 'LOSS']['pnl_dollars'].sum()
                
                results[f"{hour:02d}:00"] = {
                    'signals': total,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'total_pnl': win_pnl + loss_pnl,
                    'avg_pnl': (win_pnl + loss_pnl) / total if total > 0 else 0
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating accuracy by hour: {e}")
            return {}
    
    def generate_comprehensive_report(self, symbol: str) -> str:
        """Generate a comprehensive backtesting report"""
        
        overall = self.calculate_overall_stats(symbol)
        by_score = self.calculate_accuracy_by_confluence_score(symbol)
        by_pattern = self.calculate_accuracy_by_pattern(symbol)
        by_direction = self.calculate_accuracy_by_direction(symbol)
        by_hour = self.calculate_accuracy_by_hour(symbol)
        
        report = []
        report.append("=" * 70)
        report.append("COMPREHENSIVE BACKTESTING REPORT")
        report.append("=" * 70)
        report.append(f"Symbol: {symbol}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        report.append("📊 OVERALL STATISTICS")
        report.append("-" * 70)
        report.append(f"Total Signals: {overall['total_signals']}")
        report.append(f"Wins: {overall['wins']} ({overall['win_rate']:.2f}%)")
        report.append(f"Losses: {overall['losses']} ({overall['loss_rate']:.2f}%)")
        report.append(f"Timeouts: {overall['timeouts']} ({overall['timeout_rate']:.2f}%)")
        report.append("")
        report.append("💰 FINANCIAL METRICS")
        report.append(f"Total Profit: ${overall['total_profit']:.2f}")
        report.append(f"Total Loss: ${overall['total_loss']:.2f}")
        report.append(f"Net Profit: ${overall['net_profit']:.2f}")
        report.append(f"Profit Factor: {overall['profit_factor']:.2f}")
        report.append(f"Avg Win: ${overall['avg_win']:.2f}")
        report.append(f"Avg Loss: ${overall['avg_loss']:.2f}")
        report.append(f"Best Trade: ${overall['best_trade']:.2f}")
        report.append(f"Worst Trade: ${overall['worst_trade']:.2f}")
        report.append(f"Expectancy: {overall['expectancy']:.4f}%")
        report.append("")
        
        # Accuracy by confluence score
        if by_score:
            report.append("📈 ACCURACY BY CONFLUENCE SCORE")
            report.append("-" * 70)
            for score_range, stats in by_score.items():
                report.append(f"{score_range:8s}: {stats['win_rate']:6.2f}% win rate | " +
                             f"{stats['signals']:3d} signals | " +
                             f"${stats['total_pnl']:8.2f} PnL")
            report.append("")
        
        # Accuracy by pattern
        if by_pattern:
            report.append("🎯 ACCURACY BY PATTERN")
            report.append("-" * 70)
            for pattern, stats in by_pattern.items():
                report.append(f"{pattern:20s}: {stats['win_rate']:6.2f}% win rate | " +
                             f"{stats['signals']:3d} signals")
            report.append("")
        
        # Accuracy by direction
        if by_direction:
            report.append("🔼 ACCURACY BY DIRECTION")
            report.append("-" * 70)
            for direction, stats in by_direction.items():
                report.append(f"{direction:6s}: {stats['win_rate']:6.2f}% win rate | " +
                             f"{stats['signals']:3d} signals | " +
                             f"${stats['total_pnl']:8.2f} PnL")
            report.append("")
        
        # Best hours
        if by_hour:
            best_hours = sorted(by_hour.items(), 
                              key=lambda x: x[1]['win_rate'], 
                              reverse=True)[:5]
            
            report.append("🕐 BEST TRADING HOURS")
            report.append("-" * 70)
            for hour, stats in best_hours:
                report.append(f"{hour}: {stats['win_rate']:6.2f}% win rate | " +
                             f"{stats['signals']:3d} signals")
            report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_report_to_file(self, symbol: str, filename: Optional[str] = None):
        """Save report to file"""
        if filename is None:
            filename = f"data/backtest_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report = self.generate_comprehensive_report(symbol)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Report saved to {filename}")
        return filename


if __name__ == "__main__":
    # Example usage
    calculator = BacktestStatisticsCalculator()
    
    print("\n" + "="*70)
    print("BACKTESTING STATISTICS CALCULATOR")
    print("="*70)
    
    # Generate comprehensive report
    report = calculator.generate_comprehensive_report("XRPUSDT")
    print(report)
    
    # Save to file
    calculator.save_report_to_file("XRPUSDT")
