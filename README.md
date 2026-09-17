# SmartTrade

Earnings-gap continuation (ER) and Black-Scholes gamma-exposure (GEX) scanners that rank names, paper-trade a $1,000 equity proxy, and optionally push a summary to ntfy.

## Data sources

- **Yahoo Finance** via `yfinance` (quotes, daily OHLCV, option chains, earnings dates).
- **Stooq** as a keyless OHLCV fallback when Yahoo history is empty. Fallback-served rows are flagged `price-src:stooq` / logged `[stooq]`, never silently mixed with Yahoo.

GEX sign is **customer** gamma (calls +, puts −). SpotGamma dealer GEX is the opposite sign. Walls come from open interest in a 12% band, not gamma-weighted ATM. Historical GEX backtests are **price-only** — Yahoo has no historical option chains.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional: `export NTFY_TOPIC=your-private-topic` for phone pushes. Leave it unset to print only.

## Usage

```bash
python3 gex_scanner.py
python3 er_dashboard.py

python3 -m paper_trading.trade_gex open   # also check|close|report|score
python3 -m paper_trading.trade_er open

python3 -m paper_trading.backtest_gex
python3 -m paper_trading.backtest_er

python3 -m unittest tests.test_paper_trading tests.test_gex_walls tests.test_er_gap -q
```

Local wrappers: `./run_gex_scan.sh`, `./run_er_scan.sh`. Watchlists: `WATCHLIST` in `gex_scanner.py`, `CORE_TICKERS` in `er_dashboard.py`.

Sample scan tables (illustrative, not live): [`examples/gex_scan.csv`](examples/gex_scan.csv), [`examples/er_dashboard.csv`](examples/er_dashboard.csv).

Thresholds are hand-set. Paper P&L is a direction scorecard on the underlying, not the recommended options spread.

More ops detail (GitHub Actions, cron, paper hold rules): [SETUP.md](SETUP.md).
