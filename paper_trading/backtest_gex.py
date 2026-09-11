"""
Historical same-day paper backtest of GEX directional signals.

Yahoo (and Stooq) do not publish historical option chains, so dealer GEX,
call/put walls, and the gamma flip cannot be reconstructed after the fact.
This backtest therefore only asks two honest questions:

    OVERSOLD_BULL_PULLBACK
        Live rule is RSI < 35 and price >= EMA200. GEX is not a gate, only
        a ranking term. Reconstruction here matches live classification.

    VOLATILITY_EXPANSION_BEAR (technical proxy)
        Live rule additionally requires NEGATIVE_GEX. That filter is skipped
        here, so this sleeve over-fires relative to production. Kept as a
        sensitivity on the price half of the signal, labeled as such.

Stops/targets use classify_setup's no-wall fallbacks (long 2%/4%, short
3%/5%), which is what a live scan uses when it cannot locate a wall.

Signal is taken on day t's close; the paper fill is day t+1's open; the
exit is simulated from day t+1's OHLC (see paper_trading.evaluate).

Usage:
    python3 -m paper_trading.backtest_gex
    python3 -m paper_trading.backtest_gex --skip-earnings
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
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gex_scanner as gex
from paper_trading import evaluate

RESULTS_PATH = Path(__file__).resolve().parent / "backtest_gex_results.json"
SUMMARY_PATH = Path(__file__).resolve().parent / "backtest_gex_summary.json"
EARNINGS_BLACKOUT_DAYS = 7
MIN_BARS = 200
# EMA200 / RSI need a filled-in window; skip this many sessions from the
# start of the downloaded series before emitting trades.
INDICATOR_WARMUP = 250
HISTORY_PERIOD = "5y"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).title() for c in out.columns]
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in out.columns]
    if len(keep) < 4:
        return pd.DataFrame()
    out = out[keep].dropna(subset=["Open", "High", "Low", "Close"])
    out = out[out["Close"] > 0]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def load_history(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Batch Yahoo download with per-symbol Stooq fallback."""
    hist: Dict[str, pd.DataFrame] = {}
    print(f"Downloading {len(symbols)} symbols (Yahoo, {HISTORY_PERIOD})...")
    try:
        raw = yf.download(
            symbols,
            period=HISTORY_PERIOD,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        print(f"[backtest] batch download failed ({exc}); trying per-symbol")
        raw = None

    for sym in symbols:
        df = pd.DataFrame()
        if raw is not None and not raw.empty:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym in raw.columns.get_level_values(0):
                        df = raw[sym]
                    elif (sym, "Close") in raw.columns:
                        df = raw[sym]
                elif "Close" in raw.columns and len(symbols) == 1:
                    df = raw
            except Exception:
                df = pd.DataFrame()
        df = _normalize_ohlcv(df)
        if df.empty or len(df) < MIN_BARS:
            fallback = gex.fetch_stooq_history(sym)
            fallback = _normalize_ohlcv(fallback)
            if not fallback.empty:
                print(f"[stooq] {sym}: history backfilled from Stooq ({len(fallback)} rows)")
                df = fallback
        if df.empty or len(df) < MIN_BARS:
            print(f"[backtest] {sym}: skipped (insufficient history)")
            continue
        hist[sym] = df
        print(f"  {sym:8s}  {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    return hist


def load_earnings_dates(symbol: str) -> Set[pd.Timestamp]:
    try:
        df = yf.Ticker(symbol).get_earnings_dates(limit=24)
    except Exception as exc:
        print(f"[backtest] {symbol}: earnings dates unavailable ({exc})")
        return set()
    if df is None or df.empty:
        return set()
    idx = pd.to_datetime(df.index)
    return {pd.Timestamp(ts).tz_localize(None).normalize() for ts in idx}


def in_earnings_blackout(signal_day: pd.Timestamp, earnings: Set[pd.Timestamp]) -> bool:
    """Match live scanner: 0 <= (earnings - today) <= 7 calendar days."""
    day = pd.Timestamp(signal_day).tz_localize(None).normalize()
    for ed in earnings:
        delta = (ed - day).days
        if 0 <= delta <= EARNINGS_BLACKOUT_DAYS:
            return True
    return False


def backtest_symbol(
    symbol: str,
    ohlcv: pd.DataFrame,
    earnings: Optional[Set[pd.Timestamp]] = None,
    include_bear_proxy: bool = True,
) -> List[Dict]:
    """Walk one symbol. Signal on close[t], fill open[t+1], exit on bar t+1."""
    df = gex.compute_indicators(ohlcv)
    if "RSI" not in df.columns or "EMA_200" not in df.columns:
        return []

    trades: List[Dict] = []
    n = len(df)
    earnings = earnings or set()
    start_i = max(INDICATOR_WARMUP, MIN_BARS) - 1

    for i in range(start_i, n - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        rsi = row.get("RSI", np.nan)
        ema = row.get("EMA_200", np.nan)
        close = float(row["Close"])
        if not np.isfinite(rsi) or not np.isfinite(ema) or close <= 0:
            continue

        signal_day = pd.Timestamp(df.index[i]).tz_localize(None).normalize()
        if in_earnings_blackout(signal_day, earnings):
            continue

        classified = gex.classify_setup(
            close, float(rsi), float(ema),
            "NO_GEX", np.nan, np.nan, np.nan,
        )
        signal = classified["signal"]
        gex_filter = "not_required"

        if signal != "OVERSOLD_BULL_PULLBACK" and include_bear_proxy:
            if rsi < 40 and close < ema:
                classified = gex.classify_setup(
                    close, float(rsi), float(ema),
                    "NEGATIVE_GEX", np.nan, ema, np.nan,
                )
                signal = classified["signal"]
                gex_filter = "skipped_unavailable"
            else:
                continue
        elif signal != "OVERSOLD_BULL_PULLBACK":
            continue

        if signal == "OVERSOLD_BULL_PULLBACK":
            direction = "LONG"
        elif signal == "VOLATILITY_EXPANSION_BEAR":
            direction = "SHORT"
        else:
            continue

        entry = float(nxt["Open"])
        if not np.isfinite(entry) or entry <= 0:
            continue

        # Re-check the gate at the fill (live uses 9:31 spot, not the prior
        # close). A gap through the EMA means this morning's scan would not
        # have printed the same directional setup.
        fill_regime = "NEGATIVE_GEX" if direction == "SHORT" else "NO_GEX"
        filled = gex.classify_setup(
            entry, float(rsi), float(ema),
            fill_regime, np.nan, np.nan, np.nan,
        )
        if filled["signal"] != signal:
            continue
        if not np.isfinite(filled["stop_loss"]) or not np.isfinite(filled["target_price"]):
            continue

        trade_day = pd.Timestamp(df.index[i + 1]).tz_localize(None).normalize()
        extra = {
            "signal_date": signal_day.strftime("%Y-%m-%d"),
            "rsi": round(float(rsi), 1),
            "gex_filter": gex_filter,
        }
        # 1d / 5d forward from entry, hold-to-close (not the paper path)
        j1 = i + 2
        j5 = i + 6
        if j1 < n:
            extra["fwd_1d_pct"] = round(
                evaluate._signed_move(direction, entry, float(df.iloc[j1]["Close"]))
                / entry * 100, 2
            )
        if j5 < n:
            extra["fwd_5d_pct"] = round(
                evaluate._signed_move(direction, entry, float(df.iloc[j5]["Close"]))
                / entry * 100, 2
            )

        trade = evaluate.build_closed_trade(
            symbol, direction, signal, trade_day.strftime("%Y-%m-%d"),
            entry, filled["stop_loss"], filled["target_price"],
            float(nxt["Open"]), float(nxt["High"]), float(nxt["Low"]), float(nxt["Close"]),
            extra=extra,
        )
        trades.append(trade)
    return trades


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def decorate_summary(summary: Dict, trades: List[Dict]) -> Dict:
    """Add hold-to-close forward-return averages onto the scorecard."""
    def add(bucket_trades):
        f1 = [t["fwd_1d_pct"] for t in bucket_trades if t.get("fwd_1d_pct") is not None]
        f5 = [t["fwd_5d_pct"] for t in bucket_trades if t.get("fwd_5d_pct") is not None]
        return {"avg_fwd_1d_pct": _mean(f1), "avg_fwd_5d_pct": _mean(f5), "n_fwd_1d": len(f1), "n_fwd_5d": len(f5)}

    summary["all"].update(add(trades))
    by_sig: Dict[str, List[Dict]] = {}
    for t in trades:
        by_sig.setdefault(t["signal"], []).append(t)
    for sig, stats in summary["by_signal"].items():
        stats.update(add(by_sig.get(sig, [])))
    return summary


CAVEATS = """
Caveats (read before treating these numbers as an edge):
- No historical option chains: walls/GEX/flip are missing. OVERSOLD is an
  exact replay of the live gate. The bear sleeve is a price-only proxy and
  will over-fire vs production (live also needs NEGATIVE_GEX).
- Stops/targets are the no-wall clamp fallbacks, not live wall-anchored
  levels, so paper P&L is not the options-spread P&L the scanner names.
- Daily OHLC cannot sequence stop vs target; days where both sit in range
  are counted as STOP and flagged ambiguous.
- Earnings blackout matches the live 0-7 day window when Yahoo returns
  dates; missing calendars mean those names are not filtered.
- Thresholds are still hand-set. This is a sanity check on direction, not
  a fitted / walk-forward-validated strategy.
""".strip()


def format_backtest_report(summary: Dict, n_symbols: int) -> str:
    lines = [
        "=== GEX paper backtest (price-only reconstruction) ===",
        CAVEATS,
        "",
        evaluate.format_scorecard(summary, title="Same-day $1,000 equity proxy"),
        "",
        "Hold-to-close forward returns (direction-adjusted, not the paper path):",
    ]
    blocks = [("ALL", summary["all"])]
    blocks.extend(summary.get("by_signal", {}).items())
    for name, s in blocks:
        lines.append(
            f"  {name}: avg 1d={s.get('avg_fwd_1d_pct')}% (n={s.get('n_fwd_1d')})  "
            f"avg 5d={s.get('avg_fwd_5d_pct')}% (n={s.get('n_fwd_5d')})"
        )
    lines.append(f"\nUniverse: {n_symbols} symbols with enough history.")
    return "\n".join(lines)


def run_backtest(symbols: List[str], skip_earnings: bool = False) -> Dict:
    hist = load_history(symbols)
    all_trades: List[Dict] = []
    for sym, df in hist.items():
        earnings: Set[pd.Timestamp] = set()
        if not skip_earnings:
            earnings = load_earnings_dates(sym)
            print(f"[backtest] {sym}: {len(earnings)} earnings date(s)")
        trades = backtest_symbol(sym, df, earnings=earnings, include_bear_proxy=True)
        print(f"[backtest] {sym}: {len(trades)} paper trade(s)")
        all_trades.extend(trades)

    summary = decorate_summary(evaluate.summarize_trades(all_trades), all_trades)
    report = format_backtest_report(summary, len(hist))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": sorted(hist.keys()),
        "skip_earnings": skip_earnings,
        "caveats": CAVEATS,
        "summary": summary,
        "trades": all_trades,
        "report": report,
    }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Historical GEX paper backtest")
    parser.add_argument("--skip-earnings", action="store_true",
                        help="Do not filter the live 7-day earnings blackout")
    parser.add_argument("--out", default=str(RESULTS_PATH),
                        help="Where to write the full JSON results (includes trades)")
    parser.add_argument("--summary-out", default=str(SUMMARY_PATH),
                        help="Where to write the summary JSON (no trade list)")
    args = parser.parse_args(argv)

    payload = run_backtest(list(gex.WATCHLIST), skip_earnings=args.skip_earnings)
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
