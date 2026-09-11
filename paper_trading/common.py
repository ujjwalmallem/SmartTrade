"""
Shared ledger mechanics for the GEX and ER paper-trading drivers.

Positions are equity-only (long or short the underlying stock at spot
price), fixed-dollar sized, and always same-day: opened near market open,
closed the moment they hit their stop/target, or force-closed at end of
day if neither triggers first. This does NOT simulate the options
strategies the scanners actually recommend (spread pricing, IV, fills) --
it paper-trades the stock as a directional proxy for the signal. Treat the
P&L here as a rough scorecard for the signal's direction call, not a
return estimate for the trade a scanner's "recommended_strategy" names.

The ledger is a JSON file: {"open": [...], "closed": [...], "scans": [...]}.
Each driver script (trade_gex.py / trade_er.py) owns its own ledger file;
the calling shell wrapper is responsible for committing it back to git
after a run that changes it. GEX `open` also stores that morning's full
scan under `scans` so wall/pin setups can be graded later even though
they are not paper-traded.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

POSITION_SIZE_USD = 1000.0


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_ledger(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {"open": [], "closed": []}
    with p.open() as f:
        return json.load(f)


def save_ledger(path: str, ledger: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2)
        f.write("\n")


def open_position(ledger: Dict, symbol: str, direction: str, entry_price,
                   stop_loss, target_price, signal: str,
                   size_usd: float = POSITION_SIZE_USD) -> Optional[Dict]:
    """Adds an open position if this symbol isn't already open. Returns the
    new position dict, or None if skipped (already open, or bad price)."""
    if pd.isna(entry_price) or entry_price <= 0:
        return None
    if any(p["symbol"] == symbol for p in ledger["open"]):
        return None   # never carry two simultaneous positions in one symbol

    shares = round(size_usd / float(entry_price), 4)
    pos = {
        "symbol": symbol,
        "direction": direction,             # "LONG" or "SHORT"
        "signal": signal,
        "date": today_str(),
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "entry_price": round(float(entry_price), 4),
        "shares": shares,
        "size_usd": size_usd,
        "stop_loss": round(float(stop_loss), 4) if pd.notna(stop_loss) else None,
        "target_price": round(float(target_price), 4) if pd.notna(target_price) else None,
    }
    ledger["open"].append(pos)
    return pos


def _pnl_usd(pos: Dict, exit_price: float) -> float:
    delta = exit_price - pos["entry_price"]
    if pos["direction"] == "SHORT":
        delta = -delta
    return round(delta * pos["shares"], 2)


def check_exit(pos: Dict, current_price) -> Optional[str]:
    """Returns "STOP" or "TARGET" if current_price has crossed either level
    for this position's direction, else None. NaN/non-positive prices never
    trigger an exit -- a bad price read should not look like a stop-out."""
    if pd.isna(current_price) or current_price <= 0:
        return None
    long = pos["direction"] == "LONG"
    stop = pos.get("stop_loss")
    target = pos.get("target_price")
    if stop is not None:
        if (current_price <= stop) if long else (current_price >= stop):
            return "STOP"
    if target is not None:
        if (current_price >= target) if long else (current_price <= target):
            return "TARGET"
    return None


def close_position(ledger: Dict, pos: Dict, exit_price: float, reason: str) -> Dict:
    ledger["open"].remove(pos)
    pos["exit_time"] = datetime.now(timezone.utc).isoformat()
    pos["exit_price"] = round(float(exit_price), 4)
    pos["exit_reason"] = reason   # "STOP", "TARGET", "EOD"
    pos["pnl_usd"] = _pnl_usd(pos, pos["exit_price"])
    pos["pnl_pct"] = round((pos["pnl_usd"] / pos["size_usd"]) * 100, 2)
    ledger["closed"].append(pos)
    return pos


def todays_closed(ledger: Dict) -> List[Dict]:
    today = today_str()
    return [p for p in ledger["closed"] if p["date"] == today]


def _df_records(df) -> List[Dict]:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    if df is None or getattr(df, "empty", True):
        return []
    return json.loads(pd.DataFrame(df).to_json(orient="records", date_format="iso"))


def record_scan(ledger: Dict, date: str, setups, residual, avoid) -> Dict:
    """Store one day's full GEX scan on the ledger, replacing any prior copy.

    Live paper trading only opens the two directional setups. Scoring whether
    WALL_PIN / RESISTANCE / etc. were right needs the rest of the scan, so
    we keep it here — the same file the workflow already commits back.
    """
    payload = {
        "date": date,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "setups": _df_records(setups),
        "residual": _df_records(residual),
        "avoid": _df_records(avoid),
    }
    scans = [s for s in ledger.get("scans", []) if s.get("date") != date]
    scans.append(payload)
    ledger["scans"] = scans
    return payload
