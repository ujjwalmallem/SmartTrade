"""ER gap / RVOL / EntryScore tests from the dashboard handoff."""

import unittest

import numpy as np
import pandas as pd

import er_dashboard as er
from paper_trading.backtest_er import backtest_symbol


class SignedGapTests(unittest.TestCase):
    def test_signed_gap_crash_has_no_upside_target(self):
        self.assertTrue(np.isnan(er.reaction_target(100.0, -8.0, 3.0)))
        self.assertFalse(er.is_paper_candidate({
            "Src": "ER", "Conv": "", "Gap%": -8.0, "React Tgt": np.nan,
            "GapAge": 0, "RVOL": 3.0, "EntryScore": 4.0, "Price": 100.0, "Stop": 96.0,
        }))
        idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
        df = pd.DataFrame({
            "Open": [100.0, 85.0],
            "High": [101.0, 86.0],
            "Low": [99.0, 84.0],
            "Close": [100.0, 85.5],
            "Volume": [1e6, 4e6],
        }, index=idx)
        gaps = er.signed_gaps(df)
        self.assertLess(float(gaps.iloc[-1]), 0)


class RvolTests(unittest.TestCase):
    def test_rvol_excludes_gap_day(self):
        n = 25
        idx = pd.bdate_range("2024-01-02", periods=n)
        vol = np.arange(1, n + 1, dtype=float) * 1_000_000.0
        vol[-1] = 100_000_000.0
        df = pd.DataFrame({
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
            "Volume": vol,
        }, index=idx)
        rvol = er.rvol_at(df, n - 1)
        leaked = float(df["Volume"].iloc[n - 1]) / float(
            df["Volume"].iloc[max(0, n - 1 - er.RVOL_LOOKBACK):n].median())
        self.assertNotAlmostEqual(rvol, leaked, places=6)
        baseline = float(df["Volume"].iloc[max(0, n - 1 - er.RVOL_LOOKBACK):n - 1].median())
        self.assertAlmostEqual(rvol, 100_000_000.0 / baseline, places=6)


class EntryScoreTests(unittest.TestCase):
    def test_entry_score_ignores_follow_through(self):
        with_follow = er.score_reaction(8.0, 3.0, follow=10.0, after_hours_focus=True)
        without = er.score_reaction(8.0, 3.0, follow=np.nan, after_hours_focus=True)
        self.assertGreater(with_follow, without)
        live = er.entry_score(8.0, 3.0, eps_surprise=np.nan, rs_20d=np.nan,
                              sector_vs_bench=np.nan)
        self.assertEqual(live, without)


class ErEventLookaheadTests(unittest.TestCase):
    def test_iter_er_events_does_not_yield_follow(self):
        n = 45
        idx = pd.bdate_range("2024-01-02", periods=n)
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        vol = np.full(n, 1_000_000.0)
        g = 40
        open_[g] = 108.0
        close[g] = 108.0
        high[g] = 109.0
        low[g] = 107.0
        vol[g] = 4_000_000.0
        close[g + 1] = 112.0  # follow-through the event must not consume
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )
        events = list(er.iter_er_events(df, {pd.Timestamp(idx[g])}))
        self.assertTrue(events)
        hit = next(e for e in events if abs(e["gap"] - 8.0) < 1e-6)
        self.assertNotIn("follow", hit)

    def test_replay_gap_fills_next_session_not_gap_open(self):
        """AAPL 2023 Q3 style: gap Friday 2023-08-04, fill Monday, not Friday open."""
        before = pd.bdate_range("2023-07-05", "2023-08-03")
        gap_day = pd.Timestamp("2023-08-04")
        after = pd.bdate_range("2023-08-07", periods=5)
        idx = before.append(pd.DatetimeIndex([gap_day])).append(after)
        n = len(idx)
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        vol = np.full(n, 1_000_000.0)
        g = list(idx).index(gap_day)
        open_[g] = 104.5
        close[g] = 104.5
        high[g] = 105.0
        low[g] = 104.0
        vol[g] = 3_200_000.0
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )
        trades = backtest_symbol("AAPL", df, {gap_day}, require_live_gate=False)
        self.assertTrue(trades)
        self.assertEqual(trades[0]["signal_date"], "2023-08-04")
        self.assertEqual(trades[0]["date"], "2023-08-07")
        self.assertNotAlmostEqual(trades[0]["entry_price"], float(open_[g]), places=4)


if __name__ == "__main__":
    unittest.main()
