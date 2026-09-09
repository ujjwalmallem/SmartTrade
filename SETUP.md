# Running the GEX scanner natively with mobile push notifications

## 1. One-time setup (macOS/Linux)

```bash
cd SmartTrade
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Get push notifications on your phone (ntfy.sh, no signup)

1. Install the **ntfy** app: [App Store](https://apps.apple.com/app/ntfy/id1625396347) or [Google Play](https://play.google.com/store/apps/details?id=io.heckel.ntfy).
2. In the app, subscribe to this topic (already wired into `run_gex_scan.sh`):
   `smarttrade-gex-bb4b87815d16`
   Topic names on ntfy.sh are public — anyone who guesses this exact string can also
   subscribe/publish to it. It's a long random string so that's unlikely, but if you
   want a private channel, either self-host ntfy or generate your own random topic
   and replace it in `run_gex_scan.sh`'s `NTFY_TOPIC` line.
3. That's it — no account, no API key.

## 3. Run it once manually to confirm it works

```bash
./run_gex_scan.sh
```

You should see the ranked setups printed in the terminal, and a push notification
arrive on your phone summarizing the top setups (or "no setups" if none hit).

## 4. Schedule it to run automatically (cron)

Edit your crontab:

```bash
crontab -e
```

Add a line to run it every weekday at 9:35am **in your machine's local time zone**
(a few minutes after the US market opens at 9:30am ET — adjust the hour/minute if
your machine isn't set to US Eastern time):

```
35 9 * * 1-5 /absolute/path/to/SmartTrade/run_gex_scan.sh >> /absolute/path/to/SmartTrade/gex_scan.log 2>&1
```

Replace `/absolute/path/to/SmartTrade` with the real path (run `pwd` inside the repo
to get it). Logs (including any errors) get appended to `gex_scan.log`.

## Notes

- The watchlist is hard-coded near the bottom of `gex_scanner.py` (`if __name__ ==
  "__main__":` block) — edit the `watchlist` list to change which tickers get scanned.
- The push notification only fires the top 5 ranked setups (to keep it short); the
  full ranked table is still printed to stdout / the log file every run.
- Per the scanner's own docstring: thresholds and scoring weights are hand-set, not
  fitted or validated against forward returns. Treat this as a triage view, not a
  tested trading signal.
