"""Unit tests for GEX classify_setup, OHLC paper path, and backtest walk."""

import unittest

import numpy as np
import pandas as pd

import gex_scanner as gex
from paper_trading import common, evaluate
from paper_trading.backtest_gex import backtest_symbol, in_earnings_blackout


class ClassifySetupTests(unittest.TestCase):
    def test_oversold_does_not_need_gex_regime(self):
        row = gex.classify_setup(100.0, rsi=30.0, ema200=90.0,
                                 regime="NO_GEX", call_wall=np.nan,
                                 put_wall=np.nan, gamma_flip=np.nan)
        self.assertEqual(row["signal"], "OVERSOLD_BULL_PULLBACK")
        self.assertAlmostEqual(row["stop_loss"], 98.0, places=4)
        self.assertAlmostEqual(row["target_price"], 104.0, places=4)

    def test_oversold_still_wins_when_regime_is_negative(self):
        row = gex.classify_setup(100.0, rsi=30.0, ema200=90.0,
                                 regime="NEGATIVE_GEX", call_wall=np.nan,
                                 put_wall=np.nan, gamma_flip=np.nan)
        self.assertEqual(row["signal"], "OVERSOLD_BULL_PULLBACK")

    def test_wall_pin_needs_call_wall_band(self):
        row = gex.classify_setup(100.0, rsi=55.0, ema200=90.0,
                                 regime="POSITIVE_GEX", call_wall=100.5,
                                 put_wall=95.0, gamma_flip=98.0)
        self.assertEqual(row["signal"], "WALL_PIN")

    def test_resistance_needs_wall_and_high_rsi(self):
        row = gex.classify_setup(100.0, rsi=72.0, ema200=90.0,
                                 regime="POSITIVE_GEX", call_wall=100.5,
                                 put_wall=95.0, gamma_flip=98.0)
        self.assertEqual(row["signal"], "RESISTANCE_PINNED_SHORT_VOL")

    def test_bear_needs_negative_gex(self):
        bear = gex.classify_setup(80.0, rsi=35.0, ema200=90.0,
                                  regime="NEGATIVE_GEX", call_wall=np.nan,
                                  put_wall=np.nan, gamma_flip=np.nan)
        self.assertEqual(bear["signal"], "VOLATILITY_EXPANSION_BEAR")
        no_gex = gex.classify_setup(80.0, rsi=35.0, ema200=90.0,
                                    regime="NO_GEX", call_wall=np.nan,
                                    put_wall=np.nan, gamma_flip=np.nan)
        self.assertEqual(no_gex["signal"], "NO_GEX_REGIME")

    def test_dampened_is_positive_gex_residual(self):
        row = gex.classify_setup(100.0, rsi=55.0, ema200=90.0,
                                 regime="POSITIVE_GEX", call_wall=120.0,
                                 put_wall=80.0, gamma_flip=95.0)
        self.assertEqual(row["signal"], "DAMPENED_BULL_TREND")


class SimulateSameDayTests(unittest.TestCase):
    def test_long_target(self):
        reason, price, amb = evaluate.simulate_same_day(
            "LONG", stop=98, target=104, bar_open=100, high=105, low=99, close=103,
        )
        self.assertEqual(reason, "TARGET")
        self.assertEqual(price, 104)
        self.assertFalse(amb)

    def test_long_stop(self):
        reason, price, amb = evaluate.simulate_same_day(
            "LONG", stop=98, target=104, bar_open=100, high=101, low=97, close=99,
        )
        self.assertEqual(reason, "STOP")
        self.assertEqual(price, 98)
        self.assertFalse(amb)

    def test_long_eod(self):
        reason, price, amb = evaluate.simulate_same_day(
            "LONG", stop=98, target=104, bar_open=100, high=102, low=99, close=101,
        )
        self.assertEqual(reason, "EOD")
        self.assertEqual(price, 101)
        self.assertFalse(amb)

    def test_both_in_range_is_conservative_stop(self):
        reason, price, amb = evaluate.simulate_same_day(
            "LONG", stop=98, target=104, bar_open=100, high=105, low=97, close=101,
        )
        self.assertEqual(reason, "STOP")
        self.assertEqual(price, 98)
        self.assertTrue(amb)

    def test_gap_down_through_stop_fills_open(self):
        reason, price, amb = evaluate.simulate_same_day(
            "LONG", stop=98, target=104, bar_open=96, high=97, low=95, close=96.5,
        )
        self.assertEqual(reason, "STOP")
        self.assertEqual(price, 96)
        self.assertFalse(amb)

    def test_short_target(self):
        reason, price, amb = evaluate.simulate_same_day(
            "SHORT", stop=103, target=95, bar_open=100, high=101, low=94, close=96,
        )
        self.assertEqual(reason, "TARGET")
        self.assertEqual(price, 95)
        self.assertFalse(amb)


class ScorecardTests(unittest.TestCase):
    def test_summarize_win_rate_and_direction(self):
        trades = [
            evaluate.build_closed_trade(
                "AAA", "LONG", "OVERSOLD_BULL_PULLBACK", "2026-01-02",
                100, 98, 104, 100, 105, 99, 103,
            ),
            evaluate.build_closed_trade(
                "BBB", "LONG", "OVERSOLD_BULL_PULLBACK", "2026-01-03",
                100, 98, 104, 100, 101, 97, 99,
            ),
        ]
        summary = evaluate.summarize_trades(trades)
        self.assertEqual(summary["all"]["n"], 2)
        self.assertEqual(summary["all"]["wins"], 1)
        self.assertEqual(summary["all"]["losses"], 1)
        self.assertEqual(summary["all"]["win_rate"], 0.5)
        # first close 103 > 100 (right), second 99 < 100 (wrong)
        self.assertEqual(summary["all"]["direction_hit_rate"], 0.5)

    def test_record_scan_replaces_same_date(self):
        ledger = {"open": [], "closed": []}
        df = pd.DataFrame([{"symbol": "AAPL", "signal": "WALL_PIN"}])
        common.record_scan(ledger, "2026-09-11", df, pd.DataFrame(), pd.DataFrame())
        common.record_scan(ledger, "2026-09-11", df, pd.DataFrame(), pd.DataFrame())
        self.assertEqual(len(ledger["scans"]), 1)
        self.assertEqual(ledger["scans"][0]["setups"][0]["symbol"], "AAPL")


class ScanScoreTests(unittest.TestCase):
    def test_wall_pin_held(self):
        row = {"symbol": "AAPL", "signal": "WALL_PIN", "price": 100.0,
               "stop_loss": 106.0, "target_price": 100.5, "call_wall": 100.5}
        scored = evaluate.score_setup_row(row, 100, 101, 99.5, 100.4)
        self.assertTrue(scored["right"])

    def test_wall_pin_breached(self):
        row = {"symbol": "AAPL", "signal": "WALL_PIN", "price": 100.0,
               "stop_loss": 106.0, "target_price": 100.5, "call_wall": 100.5}
        scored = evaluate.score_setup_row(row, 100, 108, 99.5, 107)
        self.assertFalse(scored["right"])


def _grind_then_dump(n_up=240, start=100.0) -> pd.DataFrame:
    """Slow grind up (keeps price above a 200 EMA) then a sharp dump to tank RSI."""
    closes = []
    p = start
    for _ in range(n_up):
        p *= 1.003
        closes.append(p)
    for factor in (0.97, 0.97, 0.96, 0.96, 0.97):
        p *= factor
        closes.append(p)
    # One recovery session after the dump so the backtest has a next-day bar.
    closes.append(p * 1.01)
    idx = pd.bdate_range("2023-01-02", periods=len(closes))
    close = pd.Series(closes, index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": 1_000_000,
    })


class BacktestWalkTests(unittest.TestCase):
    def test_synthetic_oversold_emits_a_long(self):
        df = _grind_then_dump()
        trades = backtest_symbol("FAKE", df, earnings=set(), include_bear_proxy=False)
        self.assertTrue(trades, "expected at least one OVERSOLD long on the dump")
        self.assertTrue(all(t["signal"] == "OVERSOLD_BULL_PULLBACK" for t in trades))
        self.assertTrue(all(t["direction"] == "LONG" for t in trades))

    def test_earnings_blackout_matches_live_window(self):
        ed = pd.Timestamp("2026-09-15")
        earnings = {ed}
        self.assertTrue(in_earnings_blackout(pd.Timestamp("2026-09-15"), earnings))
        self.assertTrue(in_earnings_blackout(pd.Timestamp("2026-09-08"), earnings))
        self.assertFalse(in_earnings_blackout(pd.Timestamp("2026-09-07"), earnings))
        self.assertFalse(in_earnings_blackout(pd.Timestamp("2026-09-16"), earnings))


if __name__ == "__main__":
    unittest.main()
