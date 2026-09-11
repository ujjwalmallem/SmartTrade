"""
Scorecard + same-day OHLC path for GEX paper trading.

Live paper trading only sees a ticker every 30 minutes, so a stop or target
is filled at whatever quote the next check happens to catch. Historical
evaluation only has daily OHLC, so this module reconstructs a single fill:

    1. Gap through stop or target at the open -> fill at the open.
    2. Both levels sit inside the day's range -> conservative STOP, flagged
       `ambiguous=True` (we cannot know which printed first).
    3. Only one level in range -> fill at that level.
    4. Neither -> EOD at the close.

Direction accuracy (`direction_right`) is independent of that path: it asks
whether the session close moved the right way from entry. A trade can lose
money on a stop and still have been directionally right, or the reverse.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from paper_trading.common import POSITION_SIZE_USD


def simulate_same_day(
    direction: str,
    stop,
    target,
    bar_open: float,
    high: float,
    low: float,
    close: float,
) -> Tuple[str, float, bool]:
    """Returns (exit_reason, exit_price, ambiguous)."""
    if any(v is None or (isinstance(v, float) and not np.isfinite(v))
           for v in (bar_open, high, low, close)):
        return "EOD", float(close) if np.isfinite(close) else float(bar_open), False

    long = direction == "LONG"
    stop = None if stop is None or (isinstance(stop, float) and not np.isfinite(stop)) else float(stop)
    target = None if target is None or (isinstance(target, float) and not np.isfinite(target)) else float(target)

    stop_at_open = stop is not None and ((bar_open <= stop) if long else (bar_open >= stop))
    target_at_open = target is not None and ((bar_open >= target) if long else (bar_open <= target))

    if stop_at_open and target_at_open:
        return "STOP", float(bar_open), True
    if stop_at_open:
        return "STOP", float(bar_open), False
    if target_at_open:
        return "TARGET", float(bar_open), False

    stop_in_range = stop is not None and ((low <= stop) if long else (high >= stop))
    target_in_range = target is not None and ((high >= target) if long else (low <= target))

    if stop_in_range and target_in_range:
        return "STOP", float(stop), True
    if stop_in_range:
        return "STOP", float(stop), False
    if target_in_range:
        return "TARGET", float(target), False
    return "EOD", float(close), False


def simulate_hold(
    direction: str,
    stop,
    target,
    bars: List[Tuple[str, float, float, float, float]],
    max_days: int,
) -> Tuple[str, float, bool, int, str]:
    """Walk up to `max_days` OHLC bars. Returns
    (reason, exit_price, ambiguous, sessions_held, exit_date).

    `bars` is [(date, open, high, low, close), ...] starting at the entry
    session. Overnight gaps are handled by simulate_same_day on each bar's
    open. If nothing hits by the last bar, reason is TIME (max-hold exit).
    """
    if not bars or max_days < 1:
        return "TIME", np.nan, False, 0, ""
    use = bars[:max_days]
    ambiguous = False
    last_date, last_close = use[-1][0], use[-1][4]
    for i, (date, o, h, l, c) in enumerate(use):
        reason, price, amb = simulate_same_day(direction, stop, target, o, h, l, c)
        ambiguous = ambiguous or amb
        if reason != "EOD":
            return reason, price, ambiguous, i + 1, date
        last_date, last_close = date, c
    reason = "EOD" if max_days <= 1 else "TIME"
    return reason, float(last_close), ambiguous, len(use), last_date


def _signed_move(direction: str, entry: float, exit_price: float) -> float:
    delta = exit_price - entry
    return -delta if direction == "SHORT" else delta


def build_closed_trade(
    symbol: str,
    direction: str,
    signal: str,
    date: str,
    entry_price: float,
    stop_loss,
    target_price,
    bar_open: float,
    high: float,
    low: float,
    close: float,
    size_usd: float = POSITION_SIZE_USD,
    hold_bars: Optional[List[Tuple[str, float, float, float, float]]] = None,
    max_days: int = 1,
    extra: Optional[Dict] = None,
) -> Dict:
    """Paper-close an equity proxy off one bar (same-day) or a hold path."""
    if hold_bars:
        reason, exit_price, ambiguous, sessions, exit_date = simulate_hold(
            direction, stop_loss, target_price, hold_bars, max_days,
        )
        close = hold_bars[min(max(sessions, 1) - 1, len(hold_bars) - 1)][4]
        extra = dict(extra or {})
        extra.setdefault("sessions_held", sessions)
        extra.setdefault("exit_date", exit_date)
    else:
        reason, exit_price, ambiguous = simulate_same_day(
            direction, stop_loss, target_price, bar_open, high, low, close,
        )
    shares = round(size_usd / float(entry_price), 4)
    pnl_usd = round(_signed_move(direction, entry_price, exit_price) * shares, 2)
    close_move = _signed_move(direction, entry_price, close)
    trade = {
        "symbol": symbol,
        "direction": direction,
        "signal": signal,
        "date": date,
        "entry_price": round(float(entry_price), 4),
        "shares": shares,
        "size_usd": size_usd,
        "stop_loss": (round(float(stop_loss), 4)
                      if stop_loss is not None and np.isfinite(stop_loss) else None),
        "target_price": (round(float(target_price), 4)
                         if target_price is not None and np.isfinite(target_price) else None),
        "exit_price": round(float(exit_price), 4),
        "exit_reason": reason,
        "pnl_usd": pnl_usd,
        "pnl_pct": round((pnl_usd / size_usd) * 100, 2),
        "ambiguous": ambiguous,
        "close_price": round(float(close), 4),
        "close_pnl_pct": round((close_move / entry_price) * 100, 2),
        "direction_right": bool(close_move > 0),
    }
    if extra:
        trade.update(extra)
    return trade


def _bucket_stats(trades: List[Dict]) -> Dict:
    n = len(trades)
    if n == 0:
        return {
            "n": 0, "wins": 0, "losses": 0, "flats": 0,
            "win_rate": None, "total_pnl": 0.0, "avg_pnl": None,
            "avg_win": None, "avg_loss": None, "profit_factor": None,
            "direction_hit_rate": None, "avg_close_pnl_pct": None,
            "ambiguous": 0, "by_exit": {},
        }
    pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    flats = n - len(wins) - len(losses)
    loss_sum = abs(sum(losses))
    dir_known = [t for t in trades if "direction_right" in t]
    close_pcts = [t["close_pnl_pct"] for t in trades if t.get("close_pnl_pct") is not None]
    by_exit: Dict[str, int] = defaultdict(int)
    for t in trades:
        by_exit[t.get("exit_reason") or "UNKNOWN"] += 1
    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "flats": flats,
        "win_rate": round(len(wins) / n, 3),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(sum(pnls) / n, 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "profit_factor": (round(sum(wins) / loss_sum, 3) if loss_sum > 0 else None),
        "direction_hit_rate": (
            round(sum(1 for t in dir_known if t["direction_right"]) / len(dir_known), 3)
            if dir_known else None
        ),
        "avg_close_pnl_pct": round(sum(close_pcts) / len(close_pcts), 3) if close_pcts else None,
        "ambiguous": sum(1 for t in trades if t.get("ambiguous")),
        "by_exit": dict(by_exit),
    }


def summarize_trades(trades: Iterable[Dict]) -> Dict:
    trades = list(trades)
    by_signal: Dict[str, List[Dict]] = defaultdict(list)
    for t in trades:
        by_signal[t.get("signal") or "UNKNOWN"].append(t)
    return {
        "all": _bucket_stats(trades),
        "by_signal": {k: _bucket_stats(v) for k, v in sorted(by_signal.items())},
    }


def _fmt_pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def _fmt_num(x, money=False) -> str:
    if x is None:
        return "n/a"
    return f"${x:.2f}" if money else str(x)


def format_scorecard(summary: Dict, title: str = "Paper scorecard") -> str:
    lines = [f"=== {title} ==="]
    blocks = [("all", summary["all"])]
    blocks.extend((k, v) for k, v in summary.get("by_signal", {}).items())
    for name, s in blocks:
        label = "ALL" if name == "all" else name
        close_pnl = "n/a" if s["avg_close_pnl_pct"] is None else f"{s['avg_close_pnl_pct']}%"
        lines.append(
            f"{label}: n={s['n']}  win={_fmt_pct(s['win_rate'])}  "
            f"dir-right={_fmt_pct(s['direction_hit_rate'])}  "
            f"P&L={_fmt_num(s['total_pnl'], money=True)}  "
            f"avg={_fmt_num(s['avg_pnl'], money=True)}  "
            f"PF={_fmt_num(s['profit_factor'])}  "
            f"close-pnl={close_pnl}  "
            f"exits={s['by_exit']}  ambiguous={s['ambiguous']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live-scan scoring (includes non-directional setups the paper driver skips)
# ---------------------------------------------------------------------------

def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def score_setup_row(row: Dict, bar_open: float, high: float, low: float, close: float,
                    hold_bars: Optional[List[Tuple[str, float, float, float, float]]] = None,
                    max_days: int = 1) -> Dict:
    """Grade one scan row against that session's OHLC.

    Directional setups reuse the paper path. Range/pin setups are graded on
    whether the thesis held (no upside breach / close still near entry), not
    on a forced long/short P&L.
    """
    signal = row.get("signal")
    entry = _finite(row.get("price")) or _finite(bar_open)
    stop = _finite(row.get("stop_loss"))
    target = _finite(row.get("target_price"))
    call_wall = _finite(row.get("call_wall"))
    result = {
        "symbol": row.get("symbol"),
        "signal": signal,
        "entry_price": entry,
        "open": bar_open, "high": high, "low": low, "close": close,
        "right": None,
        "reason": "unscored",
        "paper_trade": None,
    }
    if entry is None or not all(np.isfinite(v) for v in (bar_open, high, low, close)):
        result["reason"] = "no_bar"
        return result

    if signal in ("OVERSOLD_BULL_PULLBACK", "ER_CONTINUATION"):
        trade = build_closed_trade(
            row["symbol"], "LONG", signal, row.get("date") or "",
            bar_open, stop, target, bar_open, high, low, close,
            hold_bars=hold_bars, max_days=max_days,
        )
        result["paper_trade"] = trade
        result["right"] = trade["direction_right"]
        result["reason"] = (
            "hold close > entry (long continuation)"
            if signal == "ER_CONTINUATION"
            else "same-day close > entry (long)"
        )
        return result

    if signal == "VOLATILITY_EXPANSION_BEAR":
        trade = build_closed_trade(
            row["symbol"], "SHORT", signal, row.get("date") or "",
            bar_open, stop, target, bar_open, high, low, close,
        )
        result["paper_trade"] = trade
        result["right"] = trade["direction_right"]
        result["reason"] = "same-day close < entry (short)"
        return result

    if signal == "RESISTANCE_PINNED_SHORT_VOL":
        breach = stop is not None and high >= stop
        faded = close <= entry
        result["right"] = (not breach) and faded
        result["reason"] = "no upside stop + close <= entry (fade from call wall)"
        return result

    if signal == "WALL_PIN":
        breach = stop is not None and high >= stop
        # Pin thesis: no upside wall breach and close stayed near the scan price.
        pinned = abs(close - entry) / entry <= 0.015 if entry else False
        result["right"] = (not breach) and pinned
        result["reason"] = "no upside stop + close within 1.5% of scan price"
        return result

    if signal in ("DAMPENED_BULL_TREND", "HIGH_VOLATILITY_DANGER_ZONE"):
        result["reason"] = "residual/avoid — not a setup"
        return result

    result["reason"] = "no scoring rule"
    return result


def summarize_scan_scores(scores: List[Dict]) -> Dict:
    by_signal: Dict[str, List[Dict]] = defaultdict(list)
    for s in scores:
        if s.get("right") is None:
            continue
        by_signal[s.get("signal") or "UNKNOWN"].append(s)
    out = {}
    for sig, rows in sorted(by_signal.items()):
        n = len(rows)
        hits = sum(1 for r in rows if r["right"])
        paper = [r["paper_trade"] for r in rows if r.get("paper_trade")]
        out[sig] = {
            "n": n,
            "right": hits,
            "hit_rate": round(hits / n, 3) if n else None,
            "paper": _bucket_stats(paper) if paper else None,
        }
    return out


def format_scan_scores(by_signal: Dict, title: str = "Live-scan signal grades") -> str:
    lines = [f"=== {title} ==="]
    if not by_signal:
        lines.append("No scored setups yet. Snapshots are stored on each paper-trading `open`.")
        return "\n".join(lines)
    for sig, s in by_signal.items():
        extra = ""
        if s.get("paper"):
            extra = (f"  paper-P&L={_fmt_num(s['paper']['total_pnl'], money=True)} "
                     f"win={_fmt_pct(s['paper']['win_rate'])}")
        lines.append(f"{sig}: n={s['n']}  right={s['right']}  hit={_fmt_pct(s['hit_rate'])}{extra}")
    return "\n".join(lines)
