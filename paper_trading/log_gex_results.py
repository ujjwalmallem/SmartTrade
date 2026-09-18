"""
Append daily GEX scan + journal updates. Intended to run on results/gex branch.

Usage:
  python3 -m paper_trading.log_gex_results snapshot   # after scan / open
  python3 -m paper_trading.log_gex_results outcomes   # after close / score
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("GEX_RESULTS_DIR", ROOT / "results" / "gex"))
SCANS = RESULTS / "scans"
JOURNAL = RESULTS / "journal.csv"

JOURNAL_FIELDS = [
    "date", "symbol", "signal", "rank_score", "base_score", "price", "rsi",
    "gex_1pct_m", "gex_bps", "regime", "call_wall", "put_wall", "gamma_flip",
    "call_wall_edge", "put_wall_edge", "has_earnings_data",
    "stop_loss", "target_price", "hold_horizon", "recommended_strategy",
    "exit_price", "exit_reason", "pnl_usd", "pnl_pct",
    "fwd_1d_pct", "fwd_5d_pct", "logged_at",
]


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def snapshot_from_ledger(ledger_path: Path | None = None) -> Path:
    """Copy today's scan block from paper ledger into results/gex/scans/."""
    ledger_path = ledger_path or Path(
        os.environ.get("GEX_LEDGER_PATH", ROOT / "paper_trading" / "ledger_gex.json")
    )
    SCANS.mkdir(parents=True, exist_ok=True)
    data = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    today = _today()
    scans = [s for s in data.get("scans", []) if s.get("date") == today]
    out = SCANS / f"{today}.json"
    payload = {
        "date": today,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "scans": scans,
        "open": data.get("open", []),
        "closed_today": [p for p in data.get("closed", []) if p.get("date") == today],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")

    rows = []
    for scan in scans:
        for row in scan.get("setups", []):
            r = {k: row.get(k, "") for k in JOURNAL_FIELDS}
            r["date"] = today
            r["logged_at"] = payload["logged_at"]
            for k in JOURNAL_FIELDS:
                r.setdefault(k, "")
            rows.append(r)
    _append_journal(rows)
    return out


def _append_journal(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    new_file = not JOURNAL.is_file()
    with JOURNAL.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in JOURNAL_FIELDS})


def outcomes_from_ledger(ledger_path: Path | None = None) -> None:
    """Rewrite is expensive; append closed trades as outcome lines keyed by date+symbol."""
    ledger_path = ledger_path or Path(
        os.environ.get("GEX_LEDGER_PATH", ROOT / "paper_trading" / "ledger_gex.json")
    )
    if not ledger_path.is_file():
        print("no ledger")
        return
    data = json.loads(ledger_path.read_text())
    logged_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for p in data.get("closed", []):
        rows.append({
            "date": p.get("date", ""),
            "symbol": p.get("symbol", ""),
            "signal": p.get("signal", ""),
            "price": p.get("entry_price", ""),
            "stop_loss": p.get("stop_loss", ""),
            "target_price": p.get("target_price", ""),
            "exit_price": p.get("exit_price", ""),
            "exit_reason": p.get("exit_reason", ""),
            "pnl_usd": p.get("pnl_usd", ""),
            "pnl_pct": p.get("pnl_pct", ""),
            "hold_horizon": p.get("hold_days", ""),
            "logged_at": logged_at,
        })
    _append_journal(rows)
    print(f"appended {len(rows)} closed rows")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "snapshot":
        path = snapshot_from_ledger()
        print(f"wrote {path}")
    elif cmd == "outcomes":
        outcomes_from_ledger()
    else:
        print("usage: python3 -m paper_trading.log_gex_results snapshot|outcomes")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
