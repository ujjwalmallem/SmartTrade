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
  hand-set, not fitted. The GEX/ER paper loops and `paper_trading.backtest_gex`
  / `backtest_er` are sanity checks on direction, not a tested edge.

## Paper trading

Each scanner also drives its own paper-trading loop: `paper_trading/trade_gex.py`
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
  Oversold is held up to **5 sessions** (same-day paper P&L was a coin flip;
  1d/5d drift was positive). Bear stays same-day (5d drift was negative).
  `WALL_PIN`/`RESISTANCE_PINNED_SHORT_VOL` are skipped — they're premium-selling/range
  setups with no honest long/short equity proxy. Each `open` still snapshots the
  full scan into the ledger so those can be graded later (see below).
  Names whose earnings calendar could not be fetched are classified but **not**
  paper-opened (`CONFIRM EARNINGS` prefix, score haircut).
- **ER**: LONG only. Paper opens *actionable* continuation — `Src=="ER"`, upside
  gap, `GapAge` ≤ 1 (fresh print or the next session), `EntryScore` ≥ 3.5
  (Watch cutoff **without** Fol3%, which would be look-ahead), RVOL ≥ 1.8,
  not Fighting the sector. Stale last-quarter winners with a high `Final` are
  listed on the dashboard but not re-opened. Held up to **3 sessions** (the
  dashboard's own follow-through window). Stop is `reaction_stop` (half the
  gap, clamped 2–6%), target is `React Tgt`. Each `open` snapshots the full
  dashboard into the ledger.

**Schedule** (weekdays, both sources): OPEN ~9:31 AM ET · CHECK every 30 min ~10:00 AM–3:30
PM ET (fetches current price per open position, closes on stop/target hit) · CLOSE ~3:55 PM
ET. `close` only expires holds that have reached `hold_days` (same-day names end today;
GEX oversold / ER continuation stay open). Both `open` jobs **always** push to ntfy,
even when nothing was paper-traded. A `check` that finds nothing to close is still
silent. If the 9:31 OPEN cron is dropped, the next CHECK runs open first. Manually
run any single step from the Actions tab → pick the workflow → Run workflow →
choose `open`/`check`/`close`. The ER *dashboard* scan itself is 4:45 PM ET
(`.github/workflows/daily-er-scan.yml`); paper opens the next morning so the fill
is the first session after the gap.

Position sizing and hold lengths are hand-set constants in `trade_gex.py` /
`er_dashboard.py` — edit them directly if you want different values.

### Are the GEX signals right?

Live paper trading only samples days when a directional setup fires, and option-derived
pieces (GEX regime, walls, gamma flip) cannot be rebuilt from Yahoo/Stooq history.
Two extra commands fill that gap:

```bash
# Historical paper replay (oversold = 5-session hold, bear = same-day)
python3 -m paper_trading.backtest_gex

# Live ledger P&L (empty until directional setups actually open)
python3 -m paper_trading.trade_gex report

# Grade stored morning scans against that session's OHLC
# (includes WALL_PIN / RESISTANCE, which are not paper-traded)
python3 -m paper_trading.trade_gex score
```

`backtest_gex` writes `paper_trading/backtest_gex_summary.json` (scorecard, committed
as a snapshot) and `paper_trading/backtest_gex_results.json` (full trade list, gitignored).
Read the caveats printed at the top of its report before treating the numbers as an edge:
`OVERSOLD_BULL_PULLBACK` is an exact replay of the live gate (RSI + 200 EMA);
the bear sleeve is a **technical proxy** that over-fires vs production because
live `VOLATILITY_EXPANSION_BEAR` also requires `NEGATIVE_GEX`.

### Are the ER signals right?

`Final` on the dashboard includes Fol3% — useful for ranking a print after the
move, not for entering it. Live paper uses `EntryScore` (gap + RVOL + EPS +
sector, **no** follow-through) and only on a fresh gap. Historical replay:

```bash
python3 -m paper_trading.backtest_er
python3 -m paper_trading.trade_er report
python3 -m paper_trading.trade_er score
```

`backtest_er` writes `paper_trading/backtest_er_summary.json` (committed) and
`paper_trading/backtest_er_results.json` (gitignored). It reconstructs every
earnings-window gap on `CORE_TICKERS`, fills the next open, and holds up to 3
sessions. EPS / sector RS are *not* in that replay (Yahoo history is incomplete),
so the live gate is stricter than the backtest universe.
