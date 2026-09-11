"""
Paper-trades GEX scanner setups on the underlying stock, not the actual
options strategy each signal recommends. See paper_trading/common.py for
exactly what that does and does not simulate.

Only signals with a clean directional read get paper-traded:
    OVERSOLD_BULL_PULLBACK      -> LONG
    VOLATILITY_EXPANSION_BEAR   -> SHORT
WALL_PIN and RESISTANCE_PINNED_SHORT_VOL are premium-selling / range
setups (Iron Condor, Bear Call Spread) with no honest long/short equity
proxy, so they're skipped here rather than force-mapped to a direction
that doesn't represent the signal. They are still snapshotted on each
`open` and graded by `score` against that session's OHLC.

Usage:
    python3 -m paper_trading.trade_gex open
    python3 -m paper_trading.trade_gex check
    python3 -m paper_trading.trade_gex close
    python3 -m paper_trading.trade_gex report
    python3 -m paper_trading.trade_gex score
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gex_scanner as gex
from paper_trading import common, evaluate

LEDGER_PATH = str(Path(__file__).resolve().parent / "ledger_gex.json")

DIRECTION_MAP = {
    "OVERSOLD_BULL_PULLBACK": "LONG",
    "VOLATILITY_EXPANSION_BEAR": "SHORT",
}
MAX_NEW_POSITIONS = 5
# Same-day paper P&L on OVERSOLD was a coin flip; 1d/5d hold-to-close was
# positive. Bear 5d was negative, so it stays a same-day trade.
HOLD_DAYS = {
    "OVERSOLD_BULL_PULLBACK": gex.HOLD_HORIZON_DAYS["OVERSOLD_BULL_PULLBACK"],
    "VOLATILITY_EXPANSION_BEAR": gex.HOLD_HORIZON_DAYS["VOLATILITY_EXPANSION_BEAR"],
}


def do_open():
    ledger = common.load_ledger(LEDGER_PATH)
    df_setups, df_residual, df_avoid = gex.generate_top_trades(gex.WATCHLIST)
    common.record_scan(ledger, common.today_str(), df_setups, df_residual, df_avoid)

    opened = []
    if not df_setups.empty:
        candidates = df_setups[df_setups["signal"].isin(DIRECTION_MAP)].head(MAX_NEW_POSITIONS)
        for _, row in candidates.iterrows():
            if row.get("has_earnings_data") is False:
                print(f"[paper-gex] skip {row['symbol']}: earnings calendar unconfirmed")
                continue
            pos = common.open_position(
                ledger, row["symbol"], DIRECTION_MAP[row["signal"]], row["price"],
                row["stop_loss"], row["target_price"], row["signal"],
                hold_days=HOLD_DAYS.get(row["signal"], 1),
            )
            if pos:
                opened.append(pos)

    common.save_ledger(LEDGER_PATH, ledger)

    if opened:
        lines = [
            f"{p['direction']} {p['symbol']} @ ${p['entry_price']} "
            f"(stop {p['stop_loss']}, tgt {p['target_price']}, hold {p.get('hold_days', 1)}d)"
            for p in opened
        ]
        gex.notify_ntfy(f"Paper GEX: opened {len(opened)}", "\n".join(lines))
        print(f"[paper-gex] opened {len(opened)} position(s)")
    else:
        print("[paper-gex] No directional setups to open today.")


def do_check():
    ledger = common.load_ledger(LEDGER_PATH)
    closed = []
    for pos in list(ledger["open"]):
        price = gex.cache.get_spot_price(pos["symbol"])
        reason = common.check_exit(pos, price)
        if reason:
            common.close_position(ledger, pos, price, reason)
            closed.append(pos)

    common.save_ledger(LEDGER_PATH, ledger)

    if closed:
        lines = [
            f"{p['exit_reason']} {p['symbol']} @ ${p['exit_price']} "
            f"P&L ${p['pnl_usd']} ({p['pnl_pct']:+.1f}%)"
            for p in closed
        ]
        gex.notify_ntfy(f"Paper GEX: {len(closed)} closed early", "\n".join(lines))
        print(f"[paper-gex] closed {len(closed)} position(s) early")
    else:
        print("[paper-gex] check: nothing hit stop/target")


def do_close():
    """Session-end: force-close names whose hold has expired (same-day
    signals expire today; OVERSOLD stays open up to hold_days sessions).
    Stop/target hits are handled by `check` during the day.
    """
    ledger = common.load_ledger(LEDGER_PATH)
    closed = []
    for pos in list(ledger["open"]):
        if not common.hold_expired(pos):
            continue
        price = gex.cache.get_spot_price(pos["symbol"])
        if pd.isna(price) or price <= 0:
            price = pos["entry_price"]
        hold_days = int(pos.get("hold_days") or 1)
        reason = "EOD" if hold_days <= 1 else "TIME"
        common.close_position(ledger, pos, price, reason)
        closed.append(pos)

    common.save_ledger(LEDGER_PATH, ledger)

    if closed:
        total_pnl = round(sum(p["pnl_usd"] for p in closed), 2)
        wins = sum(1 for p in closed if p["pnl_usd"] > 0)
        lines = [
            f"{p['symbol']} {p['exit_reason']} P&L ${p['pnl_usd']} ({p['pnl_pct']:+.1f}%)"
            for p in closed
        ]
        still_open = len(ledger["open"])
        summary = (f"Closed {len(closed)} | Total P&L: ${total_pnl} | "
                   f"{wins}/{len(closed)} winners | still open {still_open}\n"
                   + "\n".join(lines))
        gex.notify_ntfy(f"Paper GEX session: ${total_pnl}", summary)
        print(f"[paper-gex] session close: {len(closed)} trade(s), "
              f"total P&L ${total_pnl}, still open {still_open}")
    else:
        still_open = len(ledger["open"])
        print(f"[paper-gex] session close: nothing expired"
              + (f" ({still_open} swing(s) still open)" if still_open else ""))


def do_report():
    ledger = common.load_ledger(LEDGER_PATH)
    closed = ledger.get("closed", [])
    print(evaluate.format_scorecard(
        evaluate.summarize_trades(closed),
        title="Live GEX paper ledger (closed trades)",
    ))
    if not closed:
        print("No closed live paper trades yet. Historical read: "
              "python3 -m paper_trading.backtest_gex")
    scans = ledger.get("scans", [])
    print(f"\nStored scans: {len(scans)}"
          + (f" ({scans[0]['date']} .. {scans[-1]['date']})" if scans else ""))
    print("Grade stored scans against session OHLC with: "
          "python3 -m paper_trading.trade_gex score")


def _bars_for_symbols(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    try:
        raw = yf.download(
            symbols, start=start, end=end, auto_adjust=False,
            group_by="ticker", threads=True, progress=False,
        )
    except Exception as exc:
        print(f"[paper-gex] score download failed: {exc}")
        return {}

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            if raw is None or raw.empty:
                df = pd.DataFrame()
            elif isinstance(raw.columns, pd.MultiIndex):
                df = raw[sym] if sym in raw.columns.get_level_values(0) else pd.DataFrame()
            else:
                df = raw if len(symbols) == 1 else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        out[sym] = df
    return out


def do_score():
    ledger = common.load_ledger(LEDGER_PATH)
    scans = ledger.get("scans", [])
    if not scans:
        print("[paper-gex] No stored scans. They are written on each `open`.")
        return

    by_date_rows: Dict[str, List[Dict]] = defaultdict(list)
    symbols = set()
    for scan in scans:
        date = scan["date"]
        for row in scan.get("setups", []):
            row = dict(row)
            row["date"] = date
            by_date_rows[date].append(row)
            symbols.add(row["symbol"])

    if not by_date_rows:
        print("[paper-gex] Stored scans have no setups to grade.")
        return

    dates = sorted(by_date_rows)
    # yfinance `end` is exclusive; bump one day so the last scan date is included.
    end = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    bars = _bars_for_symbols(sorted(symbols), dates[0], end)

    scores = []
    missing = 0
    for date, rows in by_date_rows.items():
        day = pd.Timestamp(date).normalize()
        for row in rows:
            df = bars.get(row["symbol"])
            if df is None or day not in df.index:
                missing += 1
                continue
            bar = df.loc[day]
            scores.append(evaluate.score_setup_row(
                row,
                float(bar["Open"]), float(bar["High"]),
                float(bar["Low"]), float(bar["Close"]),
            ))

    by_signal = evaluate.summarize_scan_scores(scores)
    print(evaluate.format_scan_scores(by_signal))
    if missing:
        print(f"[{missing} row(s) skipped — no OHLC for that symbol/date]")
    directional = [s["paper_trade"] for s in scores if s.get("paper_trade")]
    if directional:
        print()
        print(evaluate.format_scorecard(
            evaluate.summarize_trades(directional),
            title="Directional setups as same-day paper trades",
        ))


ACTIONS = {
    "open": do_open,
    "check": do_check,
    "close": do_close,
    "report": do_report,
    "score": do_score,
}

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action not in ACTIONS:
        print("usage: python3 -m paper_trading.trade_gex open|check|close|report|score")
        sys.exit(1)
    ACTIONS[action]()
