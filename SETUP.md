# Running the scanners natively with mobile push notifications

This repo has two independent scanners, each with its own script, its own
`run_*.sh` wrapper, and its own GitHub Actions workflow:

| Scanner | Script | Wrapper | Workflow | ntfy topic |
|---|---|---|---|---|
| Gamma exposure (GEX) | `gex_scanner.py` | `run_gex_scan.sh` | `.github/workflows/daily-gex-scan.yml` | `smarttrade-gex-bb4b87815d16` |
| Earnings reaction (ER) | `er_dashboard.py` | `run_er_scan.sh` | `.github/workflows/daily-er-scan.yml` | `smarttrade-er-7bfbe66211d7` |

On top of that, each scanner also has a **paper-trading** loop that opens
positions near market open and closes them by end of day — see
[Paper trading](#paper-trading) below.

Both are live on GitHub Actions already (see below) — local setup is only
needed if you also want to run them on your own machine.

## Running on GitHub Actions (already set up, no action needed)

Both workflows are scheduled on the repo's default branch (`main`), but their
checkout step is pinned to always pull the latest code from `develop`. So:

- Push changes to `develop` and the next scheduled/manual run picks them up
  automatically — no need to merge into `main` for code changes.
- To change the **schedule itself** (the cron time or trigger config), that
  change has to make it into `main` too, since GitHub only reads a workflow's
  trigger config from the default branch.
- You can always trigger either one early from the repo's **Actions** tab →
  pick the workflow → **Run workflow**.

Current schedules (weekdays only):
- GEX: 9:35 AM ET (13:35 UTC) — a few minutes after market open, so it has a
  live opening quote.
- ER: 4:45 PM ET (20:45 UTC) — shortly after market close, so the day's full
  session (gap, volume, follow-through) is captured.

(Both times are UTC offsets for US Eastern *daylight* time. After clocks fall
back in November, add an hour to keep the same ET time, e.g. 13:35 -> 14:35.)

## 1. One-time setup for running locally (macOS/Linux)

```bash
cd SmartTrade
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Get push notifications on your phone (ntfy.sh, no signup)

1. Install the **ntfy** app: [App Store](https://apps.apple.com/app/ntfy/id1625396347) or [Google Play](https://play.google.com/store/apps/details?id=io.heckel.ntfy).
2. Subscribe to whichever topic(s) you want alerts from (see table above) —
   tap **+** in the app and enter the topic name exactly.
3. That's it — no account, no API key. Topic names are public strings, so
   they're long random values on purpose; don't share them if you'd rather
   keep the channel private.

## 3. Run one locally to confirm it works

```bash
./run_gex_scan.sh
# or
./run_er_scan.sh
```

You should see output in the terminal, and a push notification arrive on
your phone summarizing the results.

## 4. Schedule local runs automatically (cron) — optional, only if not relying on GitHub Actions

```bash
crontab -e
```

```
35 9  * * 1-5 /absolute/path/to/SmartTrade/run_gex_scan.sh >> /absolute/path/to/SmartTrade/gex_scan.log 2>&1
45 16 * * 1-5 /absolute/path/to/SmartTrade/run_er_scan.sh  >> /absolute/path/to/SmartTrade/er_scan.log 2>&1
```

Times above are local-machine time assuming it's set to US Eastern — adjust
the hour/minute if your machine uses a different time zone. Replace the path
with the real one (`pwd` inside the repo).

## Notes

- GEX watchlist: edit `WATCHLIST` near the bottom of `gex_scanner.py`.
- ER watchlist: edit `CORE_TICKERS` near the top of `er_dashboard.py`.
- Both push notifications are capped to the top few results to keep them
  short; the full ranked/scored table is always printed to stdout (and, for
  ER, also saved to `er_dashboard.csv`).
- Per each script's own docstring: thresholds and scoring weights are
  hand-set, not fitted. The GEX paper loop and `paper_trading.backtest_gex`
  are sanity checks on direction, not a tested edge.

## Paper trading

Each scanner also drives its own same-day paper-trading loop: `paper_trading/trade_gex.py`
and `paper_trading/trade_er.py`, scheduled via `.github/workflows/paper-trading-gex.yml`
and `paper-trading-er.yml`. State lives in `paper_trading/ledger_gex.json` /
`ledger_er.json`, committed back to `develop` by a bot commit after any run that
changes it (`run_paper_trading.sh` handles the commit+push).

**What it trades**: the underlying stock at spot price as a directional proxy for
the signal — fixed $1,000 paper-notional per position, long or short. It does
**not** simulate the actual options strategy each scanner recommends (spread
pricing, IV, fills); treat the P&L as a scorecard for the signal's direction call,
not a return estimate for the recommended trade.

- **GEX**: only `OVERSOLD_BULL_PULLBACK` (LONG) and `VOLATILITY_EXPANSION_BEAR`
  (SHORT) get paper-traded, using their real `stop_loss`/`target_price`.
  `WALL_PIN`/`RESISTANCE_PINNED_SHORT_VOL` are skipped — they're premium-selling/range
  setups with no honest long/short equity proxy. Each `open` still snapshots the
  full scan into the ledger so those can be graded later (see below).
- **ER**: LONG only (the whole ER scoring system is built around upside earnings-reaction
  continuation), picks rows with `Src=="ER"`, `Final >= 3.5`, and a real `React Tgt` — same
  cutoff `er_dashboard.py`'s own notification uses. ER has no native stop, so a flat 3%
  synthetic stop is used (`ER_STOP_PCT` in `trade_er.py`).

**Schedule** (weekdays, both sources): OPEN ~9:31 AM ET · CHECK every 30 min ~10:00 AM–3:30
PM ET (fetches current price per open position, closes on stop/target hit) · CLOSE (force
EOD) ~3:55 PM ET, before the 4pm market close. A `check` that finds nothing to close is
silent — no push. Manually run any single step from the Actions tab → pick the workflow →
Run workflow → choose `open`/`check`/`close`.

Position sizing, the direction map, and the ER synthetic stop are all hand-set constants
at the top of `trade_gex.py`/`trade_er.py` — edit them directly if you want different
values.

### Are the GEX signals right?

Live paper trading only samples days when a directional setup fires, and option-derived
pieces (GEX regime, walls, gamma flip) cannot be rebuilt from Yahoo/Stooq history.
Two extra commands fill that gap:

```bash
# Historical same-day paper replay of the price-only directional rules
python3 -m paper_trading.backtest_gex

# Live ledger P&L (empty until directional setups actually open)
python3 -m paper_trading.trade_gex report

# Grade stored morning scans against that session's OHLC
# (includes WALL_PIN / RESISTANCE, which are not paper-traded)
python3 -m paper_trading.trade_gex score
```

`backtest_gex` writes `paper_trading/backtest_gex_results.json`. Read the caveats
printed at the top of its report before treating the numbers as an edge:
`OVERSOLD_BULL_PULLBACK` is an exact replay of the live gate (RSI + 200 EMA);
the bear sleeve is a **technical proxy** that over-fires vs production because
live `VOLATILITY_EXPANSION_BEAR` also requires `NEGATIVE_GEX`.
