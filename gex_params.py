"""Load GEX thresholds from config/gex_params.yaml with safe defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

_DEFAULTS: Dict[str, Any] = {
    "version": "1.0.0",
    "oversold": {
        "rsi_max": 35.0,
        "require_rsi_rising": False,
        "base_score": 80.0,
        "rsi_bonus_cap": 10.0,
        "hold_days": 5,
    },
    "bear": {
        "rsi_max": 40.0,
        "base_score": 75.0,
        "rsi_bonus_cap": 10.0,
        "hold_days": 1,
    },
    "resistance": {
        "rsi_min": 68.0,
        "base_score": 60.0,
        "rsi_bonus_cap": 10.0,
        "hold_days": 1,
    },
    "wall_pin": {"base_score": 40.0, "hold_days": 1},
    "wall_band": {"below": 0.015, "above": 0.020},
    "ranking": {
        "gex_weight": 9.0,
        "min_rank_for_live": 0.0,
        "max_new_positions": 5,
    },
    "filters": {
        "require_earnings_data": True,
        "skip_wall_edge": False,
        "min_abs_gex_bps": 0.0,
    },
    "walls": {"max_dist": 0.12, "min_oi": 5},
    "earnings_blackout_days": 7,
}

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "gex_params.yaml"
_cached: Dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def reset_cache() -> None:
    """Drop the in-process cache. Tests and optimize_gex use this."""
    global _cached
    _cached = None


def load_gex_params(path: Path | None = None, force_reload: bool = False) -> Dict[str, Any]:
    global _cached
    if _cached is not None and not force_reload:
        return _cached

    cfg = copy.deepcopy(_DEFAULTS)
    p = path or _CONFIG_PATH
    if p.is_file():
        try:
            import yaml
            with p.open() as f:
                raw = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, raw)
        except Exception as exc:
            print(f"[gex_params] failed to load {p}: {exc}; using defaults")
    else:
        print(f"[gex_params] {p} missing; using built-in defaults")

    _cached = cfg
    return cfg


def hold_horizon_days(cfg: Dict[str, Any] | None = None) -> Dict[str, int]:
    cfg = cfg or load_gex_params()
    return {
        "OVERSOLD_BULL_PULLBACK": int(cfg["oversold"]["hold_days"]),
        "VOLATILITY_EXPANSION_BEAR": int(cfg["bear"]["hold_days"]),
        "RESISTANCE_PINNED_SHORT_VOL": int(cfg["resistance"]["hold_days"]),
        "WALL_PIN": int(cfg["wall_pin"]["hold_days"]),
        "DAMPENED_BULL_TREND": 1,
        "HIGH_VOLATILITY_DANGER_ZONE": 1,
        "NO_GEX_REGIME": 0,
    }
