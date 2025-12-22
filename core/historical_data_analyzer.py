#!/usr/bin/env python3
"""Utilities to analyze historical klines stored in the local DB.
"""
import os
from datetime import datetime

import sqlite3
import pandas as pd

DB_DEFAULT = os.path.join(os.path.dirname(__file__), '..', 'data', 'crypto_historical.db')


class HistoricalAnalyzer:
    def __init__(self, db_path=DB_DEFAULT):
        self.db_path = os.path.abspath(db_path)

    def get_symbol_data(self, symbol, start_date=None, end_date=None):
        conn = sqlite3.connect(self.db_path)
        params = [symbol]
        query = "SELECT * FROM historical_klines WHERE symbol = ?"
        if start_date:
            query += " AND timestamp >= ?"
            params.append(int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000))
        if end_date:
            query += " AND timestamp <= ?"
            params.append(int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000))
        query += " ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(com=period-1, adjust=False).mean()
        ma_down = down.ewm(com=period-1, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig

    def analyze_patterns(self, symbol):
        df = self.get_symbol_data(symbol)
        if df.empty:
            return df
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        return df


def integrate_with_signal_generator():
    analyzer = HistoricalAnalyzer()
    symbol = 'BTC/USDT'
    historical_data = analyzer.get_symbol_data(symbol, start_date='2023-01-01', end_date='2024-12-22')
    return historical_data


if __name__ == '__main__':
    ha = HistoricalAnalyzer()
    print('Loaded analyzer for DB:', ha.db_path)
