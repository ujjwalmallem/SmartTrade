"""
Paper-trades ER dashboard picks on the underlying stock, long only.

The whole ER scoring system is built around post-earnings upside
continuation ("React Tgt" is only ever computed for an upside gap), so
every paper trade here is LONG. Candidates are the same STRONG/Watch rows
er_dashboard.py itself would notify on: Src=="ER", Final >= 3.5, and a
real React Tgt (excludes downside-gap rows, which have no target).

ER doesn't compute a stop_loss the way GEX does, so a flat synthetic stop
is used (ER_STOP_PCT below) -- clearly not a fitted number, same spirit as
every other hand-set threshold in these scripts.

Usage:
    python3 -m paper_trading.trade_er open
    python3 -m paper_trading.trade_er check
    python3 -m paper_trading.trade_er close
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import er_dashboard as er
from paper_trading import common

LEDGER_PATH = str(Path(__file__).resolve().parent / "ledger_er.json")

ER_STOP_PCT = 0.03   # flat synthetic stop; ER dashboard has no native stop_loss field
FINAL_SCORE_THRESHOLD = 3.5   # same cutoff er_dashboard.py's own notify_summary uses for Watch/STRONG
MAX_NEW_POSITIONS = 5


def do_open():
    ledger = common.load_ledger(LEDGER_PATH)
    df = er.build_dashboard()

    opened = []
    if not df.empty:
        candidates = df[
            (df["Src"] == "ER")
            & (df["Final"] >= FINAL_SCORE_THRESHOLD)
            & df["React Tgt"].notna()
        ].sort_values("Final", ascending=False).head(MAX_NEW_POSITIONS)

        for _, row in candidates.iterrows():
            entry = row["Price"]
            stop = entry * (1 - ER_STOP_PCT) if pd.notna(entry) else None
            pos = common.open_position(
                ledger, row["Ticker"], "LONG", entry,
                stop, row["React Tgt"], f"ER Final={row['Final']}",
            )
            if pos:
                opened.append(pos)

    common.save_ledger(LEDGER_PATH, ledger)

    if opened:
        lines = [
            f"LONG {p['symbol']} @ ${p['entry_price']} "
            f"(stop {p['stop_loss']}, tgt {p['target_price']})"
            for p in opened
        ]
        er.notify_ntfy(f"Paper ER: opened {len(opened)}", "\n".join(lines))
        print(f"[paper-er] opened {len(opened)} position(s)")
    else:
        print("[paper-er] No qualifying picks to open today.")


def do_check():
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
        er.notify_ntfy(f"Paper ER: {len(closed)} closed early", "\n".join(lines))
        print(f"[paper-er] closed {len(closed)} position(s) early")
    else:
        print("[paper-er] check: nothing hit stop/target")


def do_close():
    ledger = common.load_ledger(LEDGER_PATH)
    for pos in list(ledger["open"]):
        price = er.get_current_price(pos["symbol"])
        if pd.isna(price) or price <= 0:
            price = pos["entry_price"]   # last resort so it isn't stuck open forever
        common.close_position(ledger, pos, price, "EOD")

    common.save_ledger(LEDGER_PATH, ledger)

    todays = common.todays_closed(ledger)
    if todays:
        total_pnl = round(sum(p["pnl_usd"] for p in todays), 2)
        wins = sum(1 for p in todays if p["pnl_usd"] > 0)
        lines = [
            f"{p['symbol']} {p['exit_reason']} P&L ${p['pnl_usd']} ({p['pnl_pct']:+.1f}%)"
            for p in todays
        ]
        summary = f"Total P&L: ${total_pnl} | {wins}/{len(todays)} winners\n" + "\n".join(lines)
        er.notify_ntfy(f"Paper ER EOD: ${total_pnl}", summary)
        print(f"[paper-er] EOD close: {len(todays)} trade(s), total P&L ${total_pnl}")
    else:
        print("[paper-er] EOD close: no trades today")


ACTIONS = {"open": do_open, "check": do_check, "close": do_close}

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action not in ACTIONS:
        print("usage: python3 -m paper_trading.trade_er open|check|close")
        sys.exit(1)
    ACTIONS[action]()
