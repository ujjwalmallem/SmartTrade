"""
Historical paper backtest of ER continuation trades.

The live dashboard ranks last-quarter prints with Fol3% already realized.
That score cannot be used to *enter* a trade. This replay asks the honest
question:

    After an earnings-window upside gap, using only gap + RVOL known that
    morning (no follow-through, no EPS, no sector RS), does buying the
    next session's open and holding up to FOLLOW_SESSIONS with the native
    stop/target make money as a $1,000 equity proxy?

Yahoo has no reliable historical EPS-surprise or sector-RS series in the
free endpoints we use, so EntryScore here is price/volume only. Live paper
adds EPS / short-interest / sector on top and is therefore *stricter*.

Usage:
    python3 -m paper_trading.backtest_er
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import er_dashboard as er
from paper_trading import evaluate
from paper_trading.backtest_gex import load_earnings_dates, load_history

RESULTS_PATH = Path(__file__).resolve().parent / "backtest_er_results.json"
SUMMARY_PATH = Path(__file__).resolve().parent / "backtest_er_summary.json"

LIVE_HOLD_DAYS = er.HOLD_HORIZON_DAYS
FLAT_STOP_PCT = 0.03  # old paper rule, kept as a sensitivity


def _price_entry_score(gap, rvol) -> float:
    """Live EntryScore minus EPS / SI / sector (unavailable historically)."""
    return er.score_reaction(gap, rvol, follow=np.nan,
                             after_hours_focus=er.AFTER_HOURS_FOCUS)


def _passes_live_gate(gap, rvol) -> bool:
    if pd.isna(gap) or gap <= 0:
        return False
    if pd.isna(rvol) or rvol < er.MIN_RVOL_FOR_PAPER:
        return False
    return _price_entry_score(gap, rvol) >= er.ENTRY_SCORE_THRESHOLD


def backtest_symbol(
    symbol: str,
    ohlcv: pd.DataFrame,
    earnings: Set[pd.Timestamp],
    hold_days: int = LIVE_HOLD_DAYS,
    require_live_gate: bool = True,
    stop_mode: str = "gap",
    non_overlapping: bool = True,
) -> List[Dict]:
    """Signal on the gap day; fill the next session's open; hold up to N bars."""
    if ohlcv is None or ohlcv.empty or not earnings:
        return []
    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    trades: List[Dict] = []
    busy_until = -1
    n = len(df)

    for event in er.iter_er_events(df, earnings):
        pos = event["pos"]
        fill_i = pos + 1
        if fill_i >= n:
            continue
        if non_overlapping and fill_i <= busy_until:
            continue
        gap = event["gap"]
        rvol = event["rvol"]
        if require_live_gate and not _passes_live_gate(gap, rvol):
            continue
        if not require_live_gate:
            if pd.isna(gap) or gap <= 0:
                continue
            if pd.isna(rvol) or rvol < er.MIN_RVOL_FOR_PAPER:
                continue

        nxt = df.iloc[fill_i]
        entry = float(nxt["Open"])
        if not np.isfinite(entry) or entry <= 0:
            continue

        target = er.reaction_target(entry, gap, rvol)
        if pd.isna(target):
            continue
        if stop_mode == "flat":
            stop = round(entry * (1 - FLAT_STOP_PCT), 2)
        else:
            stop = er.reaction_stop(entry, gap)
        if pd.isna(stop):
            continue

        hold_bars = []
        for k in range(fill_i, min(fill_i + hold_days, n)):
            bar = df.iloc[k]
            d = pd.Timestamp(df.index[k]).strftime("%Y-%m-%d")
            hold_bars.append((
                d,
                float(bar["Open"]), float(bar["High"]),
                float(bar["Low"]), float(bar["Close"]),
            ))
        if not hold_bars:
            continue

        gap_day = pd.Timestamp(event["gap_date"]).strftime("%Y-%m-%d")
        trade_day = pd.Timestamp(df.index[fill_i]).strftime("%Y-%m-%d")
        extra = {
            "signal_date": gap_day,
            "gap": round(float(gap), 2),
            "rvol": None if pd.isna(rvol) else round(float(rvol), 2),
            "entry_score": _price_entry_score(gap, rvol),
            "hold_days": hold_days,
            "stop_mode": stop_mode,
        }
        j1 = fill_i + 1
        j3 = fill_i + 3
        if j1 < n:
            extra["fwd_1d_pct"] = round(
                (float(df.iloc[j1]["Close"]) - entry) / entry * 100, 2)
        if j3 < n:
            extra["fwd_3d_pct"] = round(
                (float(df.iloc[j3]["Close"]) - entry) / entry * 100, 2)

        trade = evaluate.build_closed_trade(
            symbol, "LONG", er.PAPER_SIGNAL, trade_day,
            entry, stop, target,
            float(nxt["Open"]), float(nxt["High"]),
            float(nxt["Low"]), float(nxt["Close"]),
            hold_bars=hold_bars, max_days=hold_days,
            extra=extra,
        )
        trades.append(trade)
        sessions = int(trade.get("sessions_held") or 1)
        busy_until = fill_i + sessions - 1
    return trades


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def decorate_summary(summary: Dict, trades: List[Dict]) -> Dict:
    def add(bucket_trades):
        f1 = [t["fwd_1d_pct"] for t in bucket_trades if t.get("fwd_1d_pct") is not None]
        f3 = [t["fwd_3d_pct"] for t in bucket_trades if t.get("fwd_3d_pct") is not None]
        return {
            "avg_fwd_1d_pct": _mean(f1), "n_fwd_1d": len(f1),
            "avg_fwd_3d_pct": _mean(f3), "n_fwd_3d": len(f3),
        }

    summary["all"].update(add(trades))
    by_sig: Dict[str, List[Dict]] = {}
    for t in trades:
        by_sig.setdefault(t["signal"], []).append(t)
    for sig, stats in summary["by_signal"].items():
        stats.update(add(by_sig.get(sig, [])))
    return summary


CAVEATS = """
Caveats (read before treating these numbers as an edge):
- Signal is the earnings-window gap + RVOL only. Live EntryScore also adds
  EPS surprise, short interest, and sector RS, so production opens fewer
  names than this replay.
- Fol3% is never used as an input (that would be look-ahead). Final>=3.5
  on the dashboard often includes already-realized follow-through; those
  stale prints are not in this universe.
- Fill is the session after the gap (9:31 paper open). Gap-day close is
  not traded.
- No options: $1,000 long equity proxy vs React Tgt / native stop.
- Daily OHLC cannot sequence stop vs target; both-in-range days are STOP
  and flagged ambiguous.
- Thresholds are still hand-set. Sanity check on direction, not a fitted
  walk-forward edge.
""".strip()


def format_backtest_report(summary: Dict, n_symbols: int) -> str:
    lines = [
        "=== ER paper backtest (gap+RVOL reconstruction) ===",
        CAVEATS,
        "",
        evaluate.format_scorecard(summary, title="$1,000 long equity proxy (live hold rules)"),
        "",
        "Hold-to-close forward returns (not the paper path):",
    ]
    blocks = [("ALL", summary["all"])]
    blocks.extend(summary.get("by_signal", {}).items())
    for name, s in blocks:
        lines.append(
            f"  {name}: avg 1d={s.get('avg_fwd_1d_pct')}% (n={s.get('n_fwd_1d')})  "
            f"avg 3d={s.get('avg_fwd_3d_pct')}% (n={s.get('n_fwd_3d')})"
        )
    lines.append(f"\nUniverse: {n_symbols} symbols with enough history.")
    return "\n".join(lines)


def _run_sleeve(hist, earnings_map, **kwargs) -> List[Dict]:
    trades: List[Dict] = []
    for sym, df in hist.items():
        trades.extend(backtest_symbol(sym, df, earnings_map[sym], **kwargs))
    return trades


def run_backtest(symbols: List[str]) -> Dict:
    hist = load_history(symbols)
    earnings_map: Dict[str, Set[pd.Timestamp]] = {}
    for sym in hist:
        earnings_map[sym] = load_earnings_dates(sym)
        print(f"[backtest] {sym}: {len(earnings_map[sym])} earnings date(s)")

    live = _run_sleeve(hist, earnings_map, hold_days=LIVE_HOLD_DAYS,
                       require_live_gate=True, stop_mode="gap")
    print(f"[backtest] live gate 3d gap-stop: {len(live)} trade(s)")

    summary = decorate_summary(evaluate.summarize_trades(live), live)
    report = format_backtest_report(summary, len(hist))

    same_day = _run_sleeve(hist, earnings_map, hold_days=1,
                           require_live_gate=True, stop_mode="gap")
    flat3 = _run_sleeve(hist, earnings_map, hold_days=LIVE_HOLD_DAYS,
                        require_live_gate=True, stop_mode="flat")
    loose = _run_sleeve(hist, earnings_map, hold_days=LIVE_HOLD_DAYS,
                        require_live_gate=False, stop_mode="gap")
    sensitivities = {
        "same_day_gap_stop": decorate_summary(
            evaluate.summarize_trades(same_day), same_day),
        "hold_3d_flat_3pct_stop": decorate_summary(
            evaluate.summarize_trades(flat3), flat3),
        "loose_upside_rvol_3d": decorate_summary(
            evaluate.summarize_trades(loose), loose),
    }
    report += "\n\n" + evaluate.format_scorecard(
        sensitivities["same_day_gap_stop"],
        title="Sensitivity: live gate, same-day (old paper hold)",
    )
    report += "\n\n" + evaluate.format_scorecard(
        sensitivities["hold_3d_flat_3pct_stop"],
        title="Sensitivity: live gate, 3d hold, old flat 3% stop",
    )
    report += "\n\n" + evaluate.format_scorecard(
        sensitivities["loose_upside_rvol_3d"],
        title="Sensitivity: any upside ER gap with RVOL>=1.8, 3d gap-stop",
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": sorted(hist.keys()),
        "caveats": CAVEATS,
        "summary": summary,
        "sensitivities": sensitivities,
        "trades": live,
        "report": report,
    }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Historical ER paper backtest")
    parser.add_argument("--out", default=str(RESULTS_PATH),
                        help="Where to write the full JSON results (includes trades)")
    parser.add_argument("--summary-out", default=str(SUMMARY_PATH),
                        help="Where to write the summary JSON (no trade list)")
    args = parser.parse_args(argv)

    payload = run_backtest(list(er.CORE_TICKERS))
    print()
    print(payload["report"])

    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nWrote {out}")
    summary_path = Path(args.summary_out)
    slim = {k: v for k, v in payload.items() if k != "trades"}
    summary_path.write_text(json.dumps(slim, indent=2, default=str) + "\n")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
