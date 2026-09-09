# Running the scanners natively with mobile push notifications

This repo has two independent scanners, each with its own script, its own
`run_*.sh` wrapper, and its own GitHub Actions workflow:

| Scanner | Script | Wrapper | Workflow | ntfy topic |
|---|---|---|---|---|
| Gamma exposure (GEX) | `gex_scanner.py` | `run_gex_scan.sh` | `.github/workflows/daily-gex-scan.yml` | `smarttrade-gex-bb4b87815d16` |
| Earnings reaction (ER) | `er_dashboard.py` | `run_er_scan.sh` | `.github/workflows/daily-er-scan.yml` | `smarttrade-er-7bfbe66211d7` |

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

- GEX watchlist: edit `watchlist` near the bottom of `gex_scanner.py`.
- ER watchlist: edit `CORE_TICKERS` near the top of `er_dashboard.py`.
- Both push notifications are capped to the top few results to keep them
  short; the full ranked/scored table is always printed to stdout (and, for
  ER, also saved to `er_dashboard.csv`).
- Per each script's own docstring: thresholds and scoring weights are
  hand-set, not fitted or validated against forward returns. Treat these as
  triage views, not tested trading signals.
