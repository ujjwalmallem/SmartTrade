"""GEX wall / flip / earnings-gate tests from the scanner handoff."""

import unittest

import numpy as np
import pandas as pd

import gex_scanner as gex


def _chain(rows):
    return pd.DataFrame(rows)


class WallFromOiTests(unittest.TestCase):
    def test_wall_from_oi_constrained(self):
        spot = 100.0
        chain = _chain([
            {"type": "call", "strike": 102.0, "openInterest": 50},
            {"type": "call", "strike": 108.0, "openInterest": 200},
            {"type": "call", "strike": 140.0, "openInterest": 5000},  # lottery, out of band
            {"type": "put", "strike": 97.0, "openInterest": 80},
            {"type": "put", "strike": 90.0, "openInterest": 300},
            {"type": "put", "strike": 60.0, "openInterest": 9000},  # far OTM, out of band
        ])
        call_wall, put_wall = gex.select_oi_walls(chain, spot)
        self.assertEqual(call_wall, 108.0)
        self.assertEqual(put_wall, 90.0)

    def test_gamma_weighted_atm_is_not_used(self):
        spot = 100.0
        chain = _chain([
            {"type": "call", "strike": 100.0, "openInterest": 10},
            {"type": "call", "strike": 107.0, "openInterest": 400},
            {"type": "put", "strike": 100.0, "openInterest": 10},
            {"type": "put", "strike": 93.0, "openInterest": 400},
        ])
        call_wall, put_wall = gex.select_oi_walls(chain, spot)
        self.assertEqual(call_wall, 107.0)
        self.assertEqual(put_wall, 93.0)


class GammaFlipSweepTests(unittest.TestCase):
    def _book(self, strikes, is_call, oi):
        K = np.asarray(strikes, dtype=float)
        T = np.full(len(K), 30 / 365.0)
        sig = np.full(len(K), 0.25)
        return K, T, sig, np.asarray(oi, dtype=float), np.asarray(is_call)

    def _cumsum_strike_flip(self, strikes, signed):
        order = np.argsort(strikes)
        k = np.asarray(strikes, dtype=float)[order]
        c = np.cumsum(np.asarray(signed, dtype=float)[order])
        cross = np.where(np.diff(np.signbit(c)))[0]
        if len(cross) == 0:
            return np.nan
        i = cross[0]
        return float(k[i])

    def test_gamma_flip_sweep_not_cumsum(self):
        spot = 100.0
        wide_k = np.array([72.0, 90.0, 100.0, 110.0])
        signed = np.array([10.0, -80.0, 5.0, 90.0])
        cum_wide = self._cumsum_strike_flip(wide_k, signed)
        mask = wide_k >= 80.0
        cum_tight = self._cumsum_strike_flip(wide_k[mask], signed[mask])
        self.assertNotEqual(cum_wide, cum_tight)

        is_call = [False, False, True, True]
        oi = [8000.0, 500.0, 400.0, 800.0]
        K, T, sig, oi_arr, call = self._book(wide_k, is_call, oi)
        sweep_wide = gex.compute_gamma_flip(spot, K, T, sig, oi_arr, call, 0.045)
        sweep_tight = gex.compute_gamma_flip(
            spot, K[mask], T[mask], sig[mask], oi_arr[mask], call[mask], 0.045)
        self.assertTrue(np.isfinite(sweep_wide) and np.isfinite(sweep_tight))
        self.assertLess(abs(sweep_wide - sweep_tight) / spot, 0.05)


class EarningsUnknownTests(unittest.TestCase):
    def test_earnings_unknown_haircut_not_discard(self):
        closes = [100.0 + i * 0.2 for i in range(220)]
        for factor in (0.97, 0.97, 0.96, 0.96, 0.97):
            closes.append(closes[-1] * factor)
        idx = pd.bdate_range("2025-01-02", periods=len(closes))
        close = pd.Series(closes, index=idx)
        hist = pd.DataFrame({
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1_000_000,
        })

        def fake_status(*_a, **_k):
            return "UNKNOWN"

        def fake_gex(symbol, r=0.045):
            px = float(hist["Close"].iloc[-1])
            return (12.0, 1.5, "POSITIVE_GEX", px * 1.05, px * 0.95, px, px)

        orig_status = gex.cache.check_earnings_status
        orig_gex = gex.calculate_gex_and_walls
        orig_hist = gex.cache.get_history
        try:
            gex.cache.check_earnings_status = fake_status
            gex.calculate_gex_and_walls = fake_gex
            gex.cache.get_history = lambda *a, **k: hist
            res = gex.calculate_equity_signal("FAKE")
        finally:
            gex.cache.check_earnings_status = orig_status
            gex.calculate_gex_and_walls = orig_gex
            gex.cache.get_history = orig_hist

        self.assertNotEqual(res["signal"], "EARNINGS_DATA_UNAVAILABLE")
        self.assertNotEqual(res["signal"], "NO_DATA")
        self.assertFalse(res["has_earnings_data"])
        self.assertIn("CONFIRM EARNINGS", res["recommended_strategy"])
        classified = gex.classify_setup(
            res["price"], res["rsi"], hist["Close"].ewm(span=200, adjust=False).mean().iloc[-1],
            "POSITIVE_GEX", res["call_wall"], res["put_wall"], res["gamma_flip"],
        )
        self.assertAlmostEqual(res["base_score"], round(classified["base_score"] * 0.5, 2), places=2)


class OversoldSpreadTests(unittest.TestCase):
    def test_oversold_uses_put_wall_not_spot_stop(self):
        row = gex.classify_setup(
            480.0, rsi=32.0, ema200=400.0,
            regime="NO_GEX", call_wall=np.nan, put_wall=450.0, gamma_flip=np.nan,
        )
        self.assertEqual(row["signal"], "OVERSOLD_BULL_PULLBACK")
        self.assertIn("450.0", row["recommended_strategy"])
        self.assertIn("430.0", row["recommended_strategy"])
        self.assertNotAlmostEqual(row["stop_loss"], 480.0 * 0.98, places=2)
        self.assertAlmostEqual(row["stop_loss"], 480.0 * 0.92, places=1)


if __name__ == "__main__":
    unittest.main()
