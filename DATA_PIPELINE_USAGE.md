# OHLCV Data Pipeline - Usage Guide

## Overview
The data pipeline downloads historical OHLCV (Open, High, Low, Close, Volume) data from Binance for 35 cryptocurrency trading pairs, stores it in optimized Parquet format, and supports incremental updates.

## Installation

### One-time setup:
```bash
cd <project_root>

# Install project in development mode (installs all dependencies)
pip install -e .

# Install pyarrow for parquet support (if not already installed)
pip install pyarrow
```

## Quick Start

### Download 365 days of data for all 35 coins (2 timeframes each):
```bash
python -m crypto_bot.data_pipeline.update_job --lookback_days 365
```

This will:
- Create 70 parquet files: `data/ohlcv/<SYMBOL>/<TIMEFRAME>.parquet`
- Download 15m and 1h candles for each of 35 symbols
- Automatically retry on network failures
- Print detailed summary when complete

### Fetch specific timeframes only:
```bash
python -m crypto_bot.data_pipeline.update_job --timeframes 15m --lookback_days 365
```

### Fetch specific symbols:
```bash
python -m crypto_bot.data_pipeline.update_job --lookback_days 7
```

### Full help:
```bash
python -m crypto_bot.data_pipeline.update_job --help
```

## Behavior

### First run (files don't exist):
- Downloads full `--lookback_days` range (default: 365 days)
- Example: `python -m crypto_bot.data_pipeline.update_job --lookback_days 365`
- Result: Creates `data/ohlcv/BTCUSDT/15m.parquet` with ~35,000 rows per year of 15m data

### Subsequent runs (files exist):
- Fetches only from last timestamp to now
- Appends new candles to existing parquet
- Removes duplicate timestamps (keeps latest)
- Re-cleans and saves

## Data Format

Each parquet file contains:
```
Columns: timestamp, open, high, low, close, volume
- timestamp: UTC datetime (timezone-aware)
- open, high, low, close, volume: float64

Example row:
timestamp                    open      high       low      close    volume
2025-12-27 21:30:00+00:00  43250.50  43280.15  43200.25  43275.80  123.456
```

## Configuration

### Symbols
Edit `config/coins.json` to add/remove coins:
```json
{
  "symbols": [
    "BTCUSDT",
    "ETHUSDT",
    ...
  ]
}
```

### Current symbols (35):
BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT, DOGEUSDT, DOTUSDT, MATICUSDT, LITUSDT, AVAXUSDT, UNIUSDT, LINKUSDT, XLMUSDT, ATOMUSDT, MANAUSDT, SANDUSDT, DASHUSDT, VETUSDT, ICPUSDT, GMTUSDT, PEOPLEUSDT, LUNCUSDT, CHZUSDT, NEARUSDT, FLOWUSDT, FILUSDT, QTUMUSDT, MKRUSDT, SNXUSDT, SHIBUSDT, PEPEUSDT, WIFUSDT, FLOKIUSDT, OPUSDT

## Timeframes

Default: `15m 1h`

Supported (Binance): 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

To fetch more timeframes:
```bash
python -m crypto_bot.data_pipeline.update_job --timeframes 1h 4h 1d --lookback_days 365
```

## Data Quality

Each parquet file is automatically:
- **Deduplicated** - Removes duplicate timestamps (keeps first)
- **Sorted** - By timestamp ascending
- **Validated** - Ensures OHLCV columns are float
- **Checked** - Logs warnings for missing candles (does not fill)

Sample output:
```
✅ BTCUSDT 15m: Before=0, After=35040, Added=35040
  BTCUSDT 15m: rows=35040 from 2024-12-27 21:00:00+00:00 to 2025-12-27 21:00:00+00:00
```

## Error Handling

- **Network timeout (10s)**: Retries up to 3 times with exponential backoff
- **Binance rate limit**: Sleeps 0.1s between chunks
- **Missing parquet**: Creates from scratch with full lookback
- **Network interruption**: Partial downloads are discarded, restarts from last saved timestamp

## Git

Data files are **NOT** committed to git:
```
# .gitignore
data/
```

Only code in `src/` is tracked. This allows each developer to maintain their own local data.

## Example Scripts

### Load and inspect parquet files:
```python
import pandas as pd
from pathlib import Path

ohlcv_dir = Path('data/ohlcv')
btc_15m = pd.read_parquet(ohlcv_dir / 'BTCUSDT' / '15m.parquet')

print(f"Rows: {len(btc_15m):,}")
print(f"Date range: {btc_15m['timestamp'].min()} to {btc_15m['timestamp'].max()}")
print(f"Latest close: ${btc_15m['close'].iloc[-1]:,.2f}")
```

### Merge timeframes:
```python
import pandas as pd
from pathlib import Path

ohlcv_dir = Path('data/ohlcv')
df_15m = pd.read_parquet(ohlcv_dir / 'BTCUSDT' / '15m.parquet')
df_1h = pd.read_parquet(ohlcv_dir / 'BTCUSDT' / '1h.parquet')

combined = pd.concat([df_15m, df_1h]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
```

## Troubleshooting

### ModuleNotFoundError: No module named 'crypto_bot'
```bash
cd <project_root>
pip install -e .
```

### ImportError: No module named 'pyarrow'
```bash
pip install pyarrow
```

### Files not found in `data/ohlcv/`
- Check that script completed successfully (check for final summary)
- Network may have timed out - check logs
- Rerun with same parameters to resume

### Parquet files created but empty
- Likely network error during download
- Check logs for "ERROR" messages
- Rerun to fetch data

## Modules

### crypto_bot.data_pipeline.binance_ohlcv
- `fetch_klines(symbol, interval, start, end) -> DataFrame`
- Downloads OHLCV from Binance API

### crypto_bot.data_pipeline.storage
- `save_parquet(df, path)`
- `load_parquet(path) -> DataFrame`

### crypto_bot.data_pipeline.cleaning
- `clean_ohlcv(df, symbol, interval) -> DataFrame`
- Deduplicates, sorts, validates

### crypto_bot.data_pipeline.update_job
- `main()` - CLI entry point
- `update_symbol_timeframe(symbol, interval, lookback_days)`
- `load_symbols(file) -> List[str]`
