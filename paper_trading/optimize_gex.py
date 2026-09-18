"""
Grid-search GEX price-side thresholds using the existing backtest engine.

Only OVERSOLD is an exact live replay. BEAR is a price proxy (no historical GEX).

Usage:
  python3 -m paper_trading.optimize_gex
  python3 -m paper_trading.optimize_gex --oversold-only
  python3 -m paper_trading.optimize_gex --write-best  # writes config/gex_params.suggested.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gex_scanner as gex
from gex_params import load_gex_params
from paper_trading import backtest_gex, evaluate

REPORT_DIR = Path(__file__).resolve().parent.parent / "results" / "gex" / "reports"
SUGGESTED_PATH = Path(__file__).resolve().parent.parent / "config" / "gex_params.suggested.yaml"


def apply_params_to_runtime(params: Dict[str, Any]) -> None:
    """Patch live module constants so classify_setup / HOLD use search values."""
    # Prefer going through gex_params cache when classify_setup reads load_gex_params().
    import gex_params as gp
    base = load_gex_params(force_reload=True)
    merged = gp._deep_merge(base, params)
    gp._cached = merged
    gex.HOLD_HORIZON_DAYS = gp.hold_horizon_days(merged)


def run_one(params: Dict[str, Any], symbols: List[str], oversold_only: bool) -> Dict[str, Any]:
    apply_params_to_runtime(params)
    hist = backtest_gex.load_history(symbols)
    all_trades: List[Dict] = []
    for sym, ohlcv in hist.items():
        earnings = backtest_gex.load_earnings_dates(sym)
        trades = backtest_gex.backtest_symbol(
            sym,
            ohlcv,
            earnings=earnings,
            include_bear_proxy=not oversold_only,
            hold_days_by_signal=gex.HOLD_HORIZON_DAYS,
            require_rsi_rising=bool(
                params.get("oversold", {}).get("require_rsi_rising", False)
            ),
        )
        all_trades.extend(trades)

    summary = evaluate.summarize_trades(all_trades)
    summary = backtest_gex.decorate_summary(summary, all_trades)
    return {
        "params": params,
        "n_trades": summary["all"]["n"],
        "profit_factor": summary["all"].get("profit_factor"),
        "win_rate": summary["all"].get("win_rate"),
        "total_pnl": summary["all"].get("total_pnl"),
        "avg_pnl": summary["all"].get("avg_pnl"),
        "avg_fwd_5d_pct": summary["all"].get("avg_fwd_5d_pct"),
        "by_signal": summary.get("by_signal", {}),
    }


def param_grid(oversold_only: bool) -> List[Dict[str, Any]]:
    rsi_os = [28.0, 30.0, 32.0, 35.0, 38.0]
    hold_os = [3, 5, 7]
    rising = [False, True]
    grid = []
    for r, h, rise in itertools.product(rsi_os, hold_os, rising):
        p: Dict[str, Any] = {
            "oversold": {
                "rsi_max": r,
                "hold_days": h,
                "require_rsi_rising": rise,
            }
        }
        if not oversold_only:
            for br, bh in itertools.product([35.0, 40.0, 45.0], [1, 2]):
                q = deepcopy(p)
                q["bear"] = {"rsi_max": br, "hold_days": bh}
                grid.append(q)
        else:
            grid.append(p)
    return grid


def score_result(row: Dict[str, Any], min_trades: int = 40) -> float:
    """Primary objective: OVERSOLD PF if present, else all-sample PF; penalize tiny N."""
    n = row["n_trades"] or 0
    if n < min_trades:
        return -1e9
    by = row.get("by_signal") or {}
    os_ = by.get("OVERSOLD_BULL_PULLBACK") or {}
    pf = os_.get("profit_factor")
    if pf is None:
        pf = row.get("profit_factor") or 0.0
    fwd = os_.get("avg_fwd_5d_pct")
    if fwd is None:
        fwd = row.get("avg_fwd_5d_pct") or 0.0
    # Prefer PF, slight bonus for positive 5d drift
    return float(pf) + 0.02 * float(fwd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oversold-only", action="store_true")
    ap.add_argument("--write-best", action="store_true")
    ap.add_argument("--min-trades", type=int, default=40)
    args = ap.parse_args()

    symbols = list(gex.WATCHLIST)
    grid = param_grid(args.oversold_only)
    print(f"Searching {len(grid)} configs on {len(symbols)} symbols...")

    results = []
    for i, params in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {params}")
        try:
            row = run_one(params, symbols, args.oversold_only)
            row["objective"] = score_result(row, min_trades=args.min_trades)
            results.append(row)
        except Exception as exc:
            print(f"  failed: {exc}")

    results.sort(key=lambda r: r.get("objective", -1e9), reverse=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORT_DIR / f"optimize_gex_{stamp}.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "generated_at": stamp,
                "oversold_only": args.oversold_only,
                "top": results[:20],
                "all_count": len(results),
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")

    if results and results[0]["objective"] > -1e8:
        best = results[0]
        print("BEST:", best["params"], "obj=", best["objective"], "n=", best["n_trades"])
        if args.write_best:
            try:
                import yaml
                import gex_params as gp
                base = load_gex_params(force_reload=True)
                suggested = gp._deep_merge(base, best["params"])
                suggested["updated"] = stamp
                suggested["note"] = (
                    "Suggested by optimize_gex; review before promoting to gex_params.yaml"
                )
                SUGGESTED_PATH.parent.mkdir(parents=True, exist_ok=True)
                with SUGGESTED_PATH.open("w") as f:
                    yaml.safe_dump(suggested, f, sort_keys=False)
                print(f"Wrote {SUGGESTED_PATH}")
            except Exception as exc:
                print(f"Could not write suggested yaml: {exc}")
    else:
        print("No config met min_trades / objective floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
