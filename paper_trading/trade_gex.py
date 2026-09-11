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
that doesn't represent the signal.

Usage:
    python3 -m paper_trading.trade_gex open
    python3 -m paper_trading.trade_gex check
    python3 -m paper_trading.trade_gex close
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gex_scanner as gex
from paper_trading import common

LEDGER_PATH = str(Path(__file__).resolve().parent / "ledger_gex.json")

DIRECTION_MAP = {
    "OVERSOLD_BULL_PULLBACK": "LONG",
    "VOLATILITY_EXPANSION_BEAR": "SHORT",
}
MAX_NEW_POSITIONS = 5


def do_open():
    ledger = common.load_ledger(LEDGER_PATH)
    df_setups, _, _ = gex.generate_top_trades(gex.WATCHLIST)

    opened = []
    if not df_setups.empty:
        candidates = df_setups[df_setups["signal"].isin(DIRECTION_MAP)].head(MAX_NEW_POSITIONS)
        for _, row in candidates.iterrows():
            pos = common.open_position(
                ledger, row["symbol"], DIRECTION_MAP[row["signal"]], row["price"],
                row["stop_loss"], row["target_price"], row["signal"],
            )
            if pos:
                opened.append(pos)

    common.save_ledger(LEDGER_PATH, ledger)

    if opened:
        lines = [
            f"{p['direction']} {p['symbol']} @ ${p['entry_price']} "
            f"(stop {p['stop_loss']}, tgt {p['target_price']})"
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
    ledger = common.load_ledger(LEDGER_PATH)
    for pos in list(ledger["open"]):
        price = gex.cache.get_spot_price(pos["symbol"])
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
        gex.notify_ntfy(f"Paper GEX EOD: ${total_pnl}", summary)
        print(f"[paper-gex] EOD close: {len(todays)} trade(s), total P&L ${total_pnl}")
    else:
        print("[paper-gex] EOD close: no trades today")


ACTIONS = {"open": do_open, "check": do_check, "close": do_close}

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action not in ACTIONS:
        print("usage: python3 -m paper_trading.trade_gex open|check|close")
        sys.exit(1)
    ACTIONS[action]()
