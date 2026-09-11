"""
Paper-trades ER dashboard picks on the underlying stock, long only.

Only *actionable* continuation is opened: earnings-anchored upside gap,
GapAge <= MAX_ACTIONABLE_AGE, EntryScore (no Fol3 look-ahead) at the
Watch cutoff, RVOL at the first dashboard increment, not Fighting the
sector. That is a stricter, fresher subset than "Final >= 3.5", which
was re-opening last-quarter winners whose follow-through was already in
the score.

Hold is FOLLOW_SESSIONS (the dashboard's own Fol3 window), not same-day
EOD. Stop is the native reaction_stop (half the gap, 2–6%), not a flat 3%.

Usage:
    python3 -m paper_trading.trade_er open
    python3 -m paper_trading.trade_er check
    python3 -m paper_trading.trade_er close
    python3 -m paper_trading.trade_er report
    python3 -m paper_trading.trade_er score
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import er_dashboard as er
from paper_trading import common, evaluate

LEDGER_PATH = str(Path(__file__).resolve().parent / "ledger_er.json")
MAX_NEW_POSITIONS = 5
HOLD_DAYS = er.HOLD_HORIZON_DAYS


def _scan_setups(df) -> pd.DataFrame:
    """Normalize dashboard rows into the scan snapshot the grader expects."""
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "symbol": r["Ticker"],
            "signal": er.PAPER_SIGNAL,
            "price": r["Price"],
            "stop_loss": r["Stop"],
            "target_price": r["React Tgt"],
            "entry_score": r.get("EntryScore"),
            "final": r.get("Final"),
            "gap": r.get("Gap%"),
            "rvol": r.get("RVOL"),
            "gap_age": r.get("GapAge"),
            "actionable": bool(r.get("Actionable")),
            "src": r.get("Src"),
            "conv": r.get("Conv"),
            "flags": r.get("Flags"),
        })
    return pd.DataFrame(rows)


def format_open_push(opened, df) -> tuple:
    """Title + body for the daily paper-open ntfy. Always returns a message."""
    skipped = []
    if df is not None and not getattr(df, "empty", True):
        er_up = df[(df["Src"] == "ER") & (df["Gap%"] > 0) & df["React Tgt"].notna()]
        for _, r in er_up.iterrows():
            if r.get("Actionable"):
                continue
            why = []
            if pd.notna(r.get("GapAge")) and int(r["GapAge"]) > er.MAX_ACTIONABLE_AGE:
                why.append(f"age {int(r['GapAge'])}d")
            if r.get("Conv") == "Fighting":
                why.append("Fighting")
            if pd.isna(r.get("EntryScore")) or r["EntryScore"] < er.ENTRY_SCORE_THRESHOLD:
                why.append(f"Entry {r.get('EntryScore')}")
            if pd.isna(r.get("RVOL")) or r["RVOL"] < er.MIN_RVOL_FOR_PAPER:
                why.append("low RVOL")
            skipped.append(f"{r['Ticker']}" + (f" ({', '.join(why)})" if why else " (stale)"))

    if opened:
        title = f"Paper ER: opened {len(opened)}"
        lines = [
            f"LONG {p['symbol']} @ ${p['entry_price']} "
            f"(stop {p['stop_loss']}, tgt {p['target_price']}, hold {p.get('hold_days', HOLD_DAYS)}d)"
            for p in opened
        ]
        if skipped:
            lines.append("Not paper-traded: " + ", ".join(skipped[:8]))
        return title, "\n".join(lines)

    title = "Paper ER: no fresh continuation"
    lines = ["No actionable ER upside to paper-trade (need fresh gap + EntryScore)."]
    if skipped:
        lines.append("Present but skipped: " + ", ".join(skipped[:8]))
    return title, "\n".join(lines)


def do_open():
    ledger = common.load_ledger(LEDGER_PATH)
    df = er.build_dashboard()
    common.record_scan(ledger, common.today_str(), _scan_setups(df),
                       pd.DataFrame(), pd.DataFrame())

    opened = []
    candidates = er.paper_candidates(df).head(MAX_NEW_POSITIONS)
    for _, row in candidates.iterrows():
        stop = row["Stop"]
        if pd.isna(stop):
            stop = er.reaction_stop(row["Price"], row["Gap%"])
        pos = common.open_position(
            ledger, row["Ticker"], "LONG", row["Price"],
            stop, row["React Tgt"], er.PAPER_SIGNAL,
            hold_days=HOLD_DAYS,
        )
        if pos:
            opened.append(pos)

    common.save_ledger(LEDGER_PATH, ledger)

    title, body = format_open_push(opened, df)
    er.notify_ntfy(title, body, tags="chart_with_upwards_trend")
    if opened:
        print(f"[paper-er] opened {len(opened)} position(s)")
    else:
        print("[paper-er] No actionable continuation to open today.")


def do_check():
    ledger = common.load_ledger(LEDGER_PATH)
    today = common.today_str()
    if not any(s.get("date") == today for s in ledger.get("scans", [])):
        print("[paper-er] no scan for today; running open first")
        do_open()
        ledger = common.load_ledger(LEDGER_PATH)

    closed = []
    for pos in list(ledger["open"]):
        price = er.get_current_price(pos["symbol"])
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
        er.notify_ntfy(f"Paper ER: {len(closed)} closed early", "\n".join(lines),
                       tags="warning")
        print(f"[paper-er] closed {len(closed)} position(s) early")
    else:
        print("[paper-er] check: nothing hit stop/target")


def do_close():
    """Session-end: force-close names whose hold has expired. Stop/target
    hits are handled by `check` during the day."""
    ledger = common.load_ledger(LEDGER_PATH)
    closed = []
    for pos in list(ledger["open"]):
        if not common.hold_expired(pos):
            continue
        price = er.get_current_price(pos["symbol"])
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
        er.notify_ntfy(f"Paper ER session: ${total_pnl}", summary, tags="moneybag")
        print(f"[paper-er] session close: {len(closed)} trade(s), "
              f"total P&L ${total_pnl}, still open {still_open}")
    else:
        still_open = len(ledger["open"])
        print(f"[paper-er] session close: nothing expired"
              + (f" ({still_open} swing(s) still open)" if still_open else ""))


def do_report():
    ledger = common.load_ledger(LEDGER_PATH)
    closed = ledger.get("closed", [])
    print(evaluate.format_scorecard(
        evaluate.summarize_trades(closed),
        title="Live ER paper ledger (closed trades)",
    ))
    if not closed:
        print("No closed live paper trades yet. Historical read: "
              "python3 -m paper_trading.backtest_er")
    scans = ledger.get("scans", [])
    print(f"\nStored scans: {len(scans)}"
          + (f" ({scans[0]['date']} .. {scans[-1]['date']})" if scans else ""))
    print("Grade stored scans against session OHLC with: "
          "python3 -m paper_trading.trade_er score")


def _bars_for_symbols(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    try:
        raw = yf.download(
            symbols, start=start, end=end, auto_adjust=False,
            group_by="ticker", threads=True, progress=False,
        )
    except Exception as exc:
        print(f"[paper-er] score download failed: {exc}")
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
        print("[paper-er] No stored scans. They are written on each `open`.")
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
        print("[paper-er] Stored scans have no setups to grade.")
        return

    dates = sorted(by_date_rows)
    end = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=HOLD_DAYS + 2)).strftime("%Y-%m-%d")
    bars = _bars_for_symbols(sorted(symbols), dates[0], end)

    scores = []
    missing = 0
    for date, rows in by_date_rows.items():
        day = pd.Timestamp(date).normalize()
        for row in rows:
            if not row.get("actionable"):
                continue
            df = bars.get(row["symbol"])
            if df is None or df.empty:
                missing += 1
                continue
            future = df[df.index >= day]
            if future.empty:
                missing += 1
                continue
            hold_bars = []
            for ts, bar in future.head(HOLD_DAYS).iterrows():
                hold_bars.append((
                    pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    float(bar["Open"]), float(bar["High"]),
                    float(bar["Low"]), float(bar["Close"]),
                ))
            first = hold_bars[0]
            scores.append(evaluate.score_setup_row(
                row, first[1], first[2], first[3], first[4],
                hold_bars=hold_bars, max_days=HOLD_DAYS,
            ))

    by_signal = evaluate.summarize_scan_scores(scores)
    print(evaluate.format_scan_scores(by_signal, title="Live-scan ER continuation grades"))
    if missing:
        print(f"[{missing} row(s) skipped — no OHLC for that symbol/date]")
    directional = [s["paper_trade"] for s in scores if s.get("paper_trade")]
    if directional:
        print()
        print(evaluate.format_scorecard(
            evaluate.summarize_trades(directional),
            title="Actionable ER rows as paper trades",
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
        print("usage: python3 -m paper_trading.trade_er open|check|close|report|score")
        sys.exit(1)
    ACTIONS[action]()
