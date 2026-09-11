"""Unit tests for GEX classify_setup, OHLC paper path, and backtest walk."""

import unittest

import numpy as np
import pandas as pd

import gex_scanner as gex
import er_dashboard as er
from paper_trading import common, evaluate
from paper_trading.backtest_gex import backtest_symbol, in_earnings_blackout
from paper_trading.backtest_er import backtest_symbol as backtest_er_symbol
from paper_trading.trade_gex import format_open_push
from paper_trading.trade_er import format_open_push as format_er_open_push


class OpenPushTests(unittest.TestCase):
    def test_no_setups_still_has_a_title(self):
        title, body = format_open_push([], pd.DataFrame())
        self.assertEqual(title, "Paper GEX: no directional trades")
        self.assertIn("No OVERSOLD / BEAR", body)

    def test_wall_pin_listed_as_skipped(self):
        df = pd.DataFrame([
            {"symbol": "AMD", "signal": "WALL_PIN"},
            {"symbol": "AAPL", "signal": "OVERSOLD_BULL_PULLBACK"},
        ])
        opened = [{"direction": "LONG", "symbol": "AAPL", "entry_price": 100,
                   "stop_loss": 98, "target_price": 104, "hold_days": 5}]
        title, body = format_open_push(opened, df)
        self.assertEqual(title, "Paper GEX: opened 1")
        self.assertIn("AMD WALL_PIN", body)
        self.assertIn("LONG AAPL", body)

    def test_only_skipped_setups(self):
        df = pd.DataFrame([{"symbol": "AMD", "signal": "WALL_PIN"}])
        title, body = format_open_push([], df)
        self.assertEqual(title, "Paper GEX: no directional trades")
        self.assertIn("AMD WALL_PIN", body)


class ClassifySetupTests(unittest.TestCase):
    def test_oversold_does_not_need_gex_regime(self):
        row = gex.classify_setup(100.0, rsi=30.0, ema200=90.0,
                                 regime="NO_GEX", call_wall=np.nan,
                                 put_wall=np.nan, gamma_flip=np.nan)
        self.assertEqual(row["signal"], "OVERSOLD_BULL_PULLBACK")
        self.assertAlmostEqual(row["stop_loss"], 98.0, places=4)
        self.assertAlmostEqual(row["target_price"], 104.0, places=4)
        self.assertEqual(row["hold_horizon"], 5)

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

    def test_hold_hits_target_on_later_day(self):
        bars = [
            ("2026-01-02", 100, 101, 99, 100.5),
            ("2026-01-03", 100.5, 102, 100, 101),
            ("2026-01-06", 101, 105, 100.8, 104.5),
        ]
        reason, price, amb, sessions, date = evaluate.simulate_hold(
            "LONG", 98, 104, bars, max_days=5,
        )
        self.assertEqual(reason, "TARGET")
        self.assertEqual(price, 104)
        self.assertEqual(sessions, 3)
        self.assertEqual(date, "2026-01-06")
        self.assertFalse(amb)

    def test_hold_expires_at_time(self):
        bars = [
            ("2026-01-02", 100, 101, 99, 100.5),
            ("2026-01-03", 100.5, 101.2, 99.8, 100.8),
        ]
        reason, price, amb, sessions, date = evaluate.simulate_hold(
            "LONG", 98, 104, bars, max_days=2,
        )
        self.assertEqual(reason, "TIME")
        self.assertEqual(price, 100.8)
        self.assertEqual(sessions, 2)


class LastCompleteBarTests(unittest.TestCase):
    def test_uses_yesterday_during_rth(self):
        idx = pd.to_datetime(["2026-09-10", "2026-09-11"])
        hist = pd.DataFrame({"Close": [10.0, 11.0]}, index=idx)
        now = pd.Timestamp("2026-09-11 09:31", tz="America/New_York")
        row = gex.last_complete_daily_row(hist, now=now)
        self.assertEqual(float(row["Close"]), 10.0)

    def test_uses_today_after_close(self):
        idx = pd.to_datetime(["2026-09-10", "2026-09-11"])
        hist = pd.DataFrame({"Close": [10.0, 11.0]}, index=idx)
        now = pd.Timestamp("2026-09-11 16:05", tz="America/New_York")
        row = gex.last_complete_daily_row(hist, now=now)
        self.assertEqual(float(row["Close"]), 11.0)


class HoldExpiryTests(unittest.TestCase):
    def test_same_day_expires_today(self):
        pos = {"date": "2026-09-11", "hold_days": 1}
        self.assertTrue(common.hold_expired(pos, as_of="2026-09-11"))
        self.assertFalse(common.hold_expired({**pos, "hold_days": 5}, as_of="2026-09-11"))
        self.assertTrue(common.hold_expired({**pos, "hold_days": 5}, as_of="2026-09-17"))

    def test_weekdays_skip_weekend(self):
        self.assertEqual(common.weekdays_inclusive("2026-09-11", "2026-09-14"), 2)  # Fri+Mon


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


def _grind_then_dump(n_up=320, start=100.0) -> pd.DataFrame:
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
        self.assertTrue(all(t["stop_loss"] is not None for t in trades))

    def test_gap_below_ema_is_not_filled(self):
        """Prior close is oversold-above-EMA; next open gaps under the EMA."""
        df = _grind_then_dump()
        last = df.index[-1]
        dump_close = float(df.iloc[-2]["Close"])
        df.loc[last, "Open"] = dump_close * 0.5
        df.loc[last, "High"] = dump_close * 0.5
        df.loc[last, "Low"] = dump_close * 0.4
        df.loc[last, "Close"] = dump_close * 0.45
        trades = backtest_symbol("FAKE", df, earnings=set(), include_bear_proxy=False)
        self.assertFalse(
            any(abs(t["entry_price"] - dump_close * 0.5) < 1e-6 for t in trades),
            "gap-below-EMA open should not be filled",
        )

    def test_earnings_blackout_matches_live_window(self):
        ed = pd.Timestamp("2026-09-15")
        earnings = {ed}
        self.assertTrue(in_earnings_blackout(pd.Timestamp("2026-09-15"), earnings))
        self.assertTrue(in_earnings_blackout(pd.Timestamp("2026-09-08"), earnings))
        self.assertFalse(in_earnings_blackout(pd.Timestamp("2026-09-07"), earnings))
        self.assertFalse(in_earnings_blackout(pd.Timestamp("2026-09-16"), earnings))


class ErDashboardGateTests(unittest.TestCase):
    def _row(self, **overrides):
        row = {
            "Src": "ER",
            "Conv": "",
            "Gap%": 8.0,
            "React Tgt": 110.0,
            "GapAge": 0,
            "RVOL": 2.5,
            "EntryScore": 4.0,
            "Ticker": "AAA",
            "Price": 100.0,
            "Stop": 96.0,
            "Flags": "fresh",
        }
        row.update(overrides)
        return row

    def test_fresh_upside_is_actionable(self):
        self.assertTrue(er.is_paper_candidate(self._row()))

    def test_stale_gap_is_not_actionable(self):
        self.assertFalse(er.is_paper_candidate(self._row(GapAge=5)))

    def test_down_gap_is_not_actionable(self):
        self.assertFalse(er.is_paper_candidate(self._row(**{"Gap%": -8.0, "React Tgt": np.nan})))

    def test_fighting_is_not_actionable(self):
        self.assertFalse(er.is_paper_candidate(self._row(Conv="Fighting")))

    def test_low_entry_score_is_not_actionable(self):
        self.assertFalse(er.is_paper_candidate(self._row(EntryScore=2.0)))

    def test_entry_score_ignores_follow(self):
        with_follow = er.score_reaction(8.0, 3.0, follow=10.0, after_hours_focus=True)
        without = er.score_reaction(8.0, 3.0, follow=np.nan, after_hours_focus=True)
        self.assertGreater(with_follow, without)
        live = er.entry_score(8.0, 3.0, eps_surprise=np.nan, rs_20d=np.nan,
                              sector_vs_bench=np.nan)
        self.assertEqual(live, without)

    def test_stop_is_half_gap_clamped(self):
        self.assertAlmostEqual(er.reaction_stop(100.0, 8.0), 96.0, places=2)
        self.assertAlmostEqual(er.reaction_stop(100.0, 2.0), 98.0, places=2)  # floor 2%
        self.assertAlmostEqual(er.reaction_stop(100.0, 20.0), 94.0, places=2)  # cap 6%
        self.assertTrue(np.isnan(er.reaction_stop(100.0, -8.0)))

    def test_last_complete_bar_drops_rth_today(self):
        idx = pd.to_datetime(["2026-09-10", "2026-09-11"])
        df = pd.DataFrame({"Close": [10.0, 11.0]}, index=idx)
        now = pd.Timestamp("2026-09-11 09:31", tz="America/New_York")
        trimmed = er.last_complete_daily_frame(df, now=now)
        self.assertEqual(float(trimmed["Close"].iloc[-1]), 10.0)
        after = er.last_complete_daily_frame(
            df, now=pd.Timestamp("2026-09-11 16:05", tz="America/New_York"))
        self.assertEqual(float(after["Close"].iloc[-1]), 11.0)


class ErOpenPushTests(unittest.TestCase):
    def test_no_candidates_still_has_a_title(self):
        title, body = format_er_open_push([], pd.DataFrame())
        self.assertEqual(title, "Paper ER: no fresh continuation")
        self.assertIn("No actionable", body)

    def test_opened_lists_hold(self):
        opened = [{"symbol": "NVDA", "entry_price": 100, "stop_loss": 96,
                   "target_price": 108, "hold_days": 3}]
        title, body = format_er_open_push(opened, pd.DataFrame())
        self.assertEqual(title, "Paper ER: opened 1")
        self.assertIn("hold 3d", body)


def _er_gap_frame(n_before=40, gap_pct=0.08, follow_up=True) -> pd.DataFrame:
    """Quiet tape, then an 8% earnings gap with 4x volume, then follow-through."""
    n = n_before + 5
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    vol = np.full(n, 1_000_000.0)
    g = n_before
    open_[g] = 100.0 * (1 + gap_pct)
    close[g] = open_[g]
    high[g] = close[g] * 1.01
    low[g] = close[g] * 0.99
    vol[g] = 4_000_000.0
    px = close[g]
    for i in range(g + 1, n):
        px = px * (1.02 if follow_up else 0.99)
        open_[i] = close[i - 1] * 1.001
        close[i] = px
        high[i] = max(open_[i], close[i]) * 1.01
        low[i] = min(open_[i], close[i]) * 0.99
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


class ErBacktestWalkTests(unittest.TestCase):
    def test_synthetic_gap_emits_a_long(self):
        df = _er_gap_frame()
        gap_day = df.index[40]
        trades = backtest_er_symbol("FAKE", df, {pd.Timestamp(gap_day)})
        self.assertTrue(trades, "expected an ER continuation long after the gap")
        self.assertTrue(all(t["signal"] == er.PAPER_SIGNAL for t in trades))
        self.assertTrue(all(t["direction"] == "LONG" for t in trades))
        self.assertEqual(trades[0]["hold_days"], er.HOLD_HORIZON_DAYS)

    def test_down_gap_is_not_traded(self):
        df = _er_gap_frame(gap_pct=-0.10)
        gap_day = df.index[40]
        trades = backtest_er_symbol("FAKE", df, {pd.Timestamp(gap_day)})
        self.assertEqual(trades, [])


if __name__ == "__main__":
    unittest.main()
