"""GEX yaml params: defaults match live, overrides change classify_setup."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

import gex_params
import gex_scanner as gex
from paper_trading import log_gex_results, optimize_gex


class GexParamsTests(unittest.TestCase):
    def tearDown(self):
        gex_params.reset_cache()
        gex.HOLD_HORIZON_DAYS = gex._hold_horizon_days()

    def test_defaults_match_live_thresholds(self):
        cfg = gex_params.load_gex_params(force_reload=True)
        self.assertEqual(cfg["oversold"]["rsi_max"], 35.0)
        self.assertEqual(cfg["oversold"]["hold_days"], 5)
        self.assertFalse(cfg["oversold"]["require_rsi_rising"])
        self.assertEqual(cfg["bear"]["rsi_max"], 40.0)
        self.assertEqual(cfg["bear"]["hold_days"], 1)
        self.assertEqual(cfg["resistance"]["rsi_min"], 68.0)
        self.assertEqual(cfg["wall_pin"]["base_score"], 40.0)
        self.assertEqual(cfg["wall_band"]["below"], 0.015)
        self.assertEqual(cfg["wall_band"]["above"], 0.020)
        self.assertEqual(cfg["ranking"]["gex_weight"], 9.0)
        self.assertEqual(cfg["ranking"]["max_new_positions"], 5)
        self.assertFalse(cfg["filters"]["skip_wall_edge"])
        self.assertTrue(cfg["filters"]["require_earnings_data"])
        self.assertEqual(cfg["walls"]["max_dist"], 0.12)
        self.assertEqual(cfg["earnings_blackout_days"], 7)
        self.assertEqual(gex.HOLD_HORIZON_DAYS["OVERSOLD_BULL_PULLBACK"], 5)

    def test_rsi_max_override_changes_classify(self):
        gex_params.reset_cache()
        base = gex_params.load_gex_params(force_reload=True)
        gex_params._cached = gex_params._deep_merge(
            base, {"oversold": {"rsi_max": 30.0, "hold_days": 3}}
        )
        still = gex.classify_setup(
            100.0, rsi=32.0, ema200=90.0,
            regime="NO_GEX", call_wall=np.nan, put_wall=np.nan, gamma_flip=np.nan,
        )
        self.assertNotEqual(still["signal"], "OVERSOLD_BULL_PULLBACK")
        dumped = gex.classify_setup(
            100.0, rsi=28.0, ema200=90.0,
            regime="NO_GEX", call_wall=np.nan, put_wall=np.nan, gamma_flip=np.nan,
        )
        self.assertEqual(dumped["signal"], "OVERSOLD_BULL_PULLBACK")
        self.assertEqual(dumped["hold_horizon"], 3)

    def test_wall_band_override(self):
        gex_params.reset_cache()
        base = gex_params.load_gex_params(force_reload=True)
        gex_params._cached = gex_params._deep_merge(
            base, {"wall_band": {"below": 0.001, "above": 0.001}}
        )
        # 100 vs call wall 100.5 is ~0.5% below wall — outside a 0.1% band.
        row = gex.classify_setup(
            100.0, rsi=55.0, ema200=90.0,
            regime="POSITIVE_GEX", call_wall=100.5, put_wall=95.0, gamma_flip=98.0,
        )
        self.assertEqual(row["signal"], "DAMPENED_BULL_TREND")

    def test_load_from_temp_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gex_params.yaml"
            path.write_text(yaml.safe_dump({"oversold": {"rsi_max": 25.0}}))
            cfg = gex_params.load_gex_params(path=path, force_reload=True)
            self.assertEqual(cfg["oversold"]["rsi_max"], 25.0)
            self.assertEqual(cfg["bear"]["rsi_max"], 40.0)


class OptimizeGexHelperTests(unittest.TestCase):
    def test_oversold_grid_size(self):
        grid = optimize_gex.param_grid(oversold_only=True)
        self.assertEqual(len(grid), 5 * 3 * 2)

    def test_score_penalizes_tiny_n(self):
        row = {
            "n_trades": 3,
            "profit_factor": 9.0,
            "avg_fwd_5d_pct": 10.0,
            "by_signal": {},
        }
        self.assertLess(optimize_gex.score_result(row, min_trades=40), -1e8)


class LogGexResultsTests(unittest.TestCase):
    def test_snapshot_writes_scan_and_journal(self):
        today = log_gex_results._today()
        ledger = {
            "open": [{"symbol": "AMAT", "signal": "OVERSOLD_BULL_PULLBACK"}],
            "closed": [],
            "scans": [{
                "date": today,
                "setups": [{
                    "symbol": "NVDA",
                    "signal": "WALL_PIN",
                    "rank_score": 44.5,
                    "price": 177.0,
                    "rsi": 55.0,
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ledger_path = td_path / "ledger.json"
            ledger_path.write_text(json.dumps(ledger))
            results = td_path / "results"
            orig_results = log_gex_results.RESULTS
            orig_scans = log_gex_results.SCANS
            orig_journal = log_gex_results.JOURNAL
            try:
                log_gex_results.RESULTS = results
                log_gex_results.SCANS = results / "scans"
                log_gex_results.JOURNAL = results / "journal.csv"
                out = log_gex_results.snapshot_from_ledger(ledger_path)
                self.assertTrue(out.is_file())
                payload = json.loads(out.read_text())
                self.assertEqual(payload["date"], today)
                self.assertEqual(payload["scans"][0]["setups"][0]["symbol"], "NVDA")
                journal = (results / "journal.csv").read_text()
                self.assertIn("NVDA", journal)
                self.assertIn("WALL_PIN", journal)
            finally:
                log_gex_results.RESULTS = orig_results
                log_gex_results.SCANS = orig_scans
                log_gex_results.JOURNAL = orig_journal


if __name__ == "__main__":
    unittest.main()
