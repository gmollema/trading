"""Unit tests for trading_bot.cli.smc_fill_model.

would_fill carries the analysis: it decides which entries a resting limit
order catches, and the whole adverse-selection question is downstream of
that decision. Its edge cases -- the last bar of a series, and a signal on
a day's final bar where nothing rests overnight -- are the ones that would
silently inflate the fill rate if wrong.
"""

import unittest

import pandas as pd

from trading_bot.cli import smc_fill_model as m


def _bars(rows):
    """rows are (timestamp, low, close)."""
    return {
        "open": [c for _, _, c in rows],
        "high": [c for _, _, c in rows],
        "low": [low for _, low, _ in rows],
        "close": [c for _, _, c in rows],
        "date": [pd.Timestamp(ts) for ts, _, _ in rows],
    }


class WouldFillImmediateTest(unittest.TestCase):
    def setUp(self):
        self.bars = _bars([
            ("2026-01-02 14:30:00+00:00", 99.0, 99.5),   # closed below the level
            ("2026-01-02 14:35:00+00:00", 99.0, 100.5),  # closed above it
        ])

    def test_fills_when_the_signal_bar_closed_at_or_below_the_level(self):
        self.assertTrue(m.would_fill(self.bars, 0, 100.0, "immediate"))

    def test_misses_when_the_signal_bar_closed_above_the_level(self):
        self.assertFalse(m.would_fill(self.bars, 1, 100.0, "immediate"))

    def test_close_exactly_at_the_level_fills(self):
        bars = _bars([("2026-01-02 14:30:00+00:00", 99.0, 100.0)])
        self.assertTrue(m.would_fill(bars, 0, 100.0, "immediate"))

    def test_does_not_read_past_the_end_of_the_series(self):
        bars = _bars([("2026-01-02 14:30:00+00:00", 99.0, 99.0)])
        self.assertTrue(m.would_fill(bars, 0, 100.0, "immediate"))


class WouldFillNextBarTest(unittest.TestCase):
    def test_fills_when_the_next_bar_trades_back_to_the_level(self):
        bars = _bars([
            ("2026-01-02 14:30:00+00:00", 100.0, 101.0),
            ("2026-01-02 14:35:00+00:00", 99.5, 100.5),   # dipped to 99.5
        ])
        self.assertTrue(m.would_fill(bars, 0, 100.0, "next_bar"))

    def test_misses_when_the_next_bar_stays_above_the_level(self):
        # Price turned straight around off the block -- the adverse case.
        bars = _bars([
            ("2026-01-02 14:30:00+00:00", 100.0, 101.0),
            ("2026-01-02 14:35:00+00:00", 100.5, 102.0),
        ])
        self.assertFalse(m.would_fill(bars, 0, 100.0, "next_bar"))

    def test_last_bar_of_the_series_cannot_fill(self):
        bars = _bars([("2026-01-02 14:30:00+00:00", 100.0, 101.0)])
        self.assertFalse(m.would_fill(bars, 0, 100.0, "next_bar"))

    def test_nothing_rests_overnight(self):
        # Next bar would satisfy the price test, but it is the next SESSION.
        # The strategy force-closes daily and no order is carried across.
        bars = _bars([
            ("2026-01-02 20:55:00+00:00", 100.0, 101.0),
            ("2026-01-05 14:30:00+00:00", 98.0, 99.0),
        ])
        self.assertFalse(m.would_fill(bars, 0, 100.0, "next_bar"))

    def test_same_day_later_bar_still_fills(self):
        bars = _bars([
            ("2026-01-02 14:30:00+00:00", 100.0, 101.0),
            ("2026-01-02 14:35:00+00:00", 99.0, 99.5),
        ])
        self.assertTrue(m.would_fill(bars, 0, 100.0, "next_bar"))


class WouldFillValidationTest(unittest.TestCase):
    def test_unknown_model_is_rejected_not_silently_treated_as_a_miss(self):
        bars = _bars([("2026-01-02 14:30:00+00:00", 99.0, 99.0)])
        with self.assertRaises(ValueError):
            m.would_fill(bars, 0, 100.0, "typo_model")


class TradeReturnTest(unittest.TestCase):
    def test_sums_the_fill_ladder_weighted_by_fraction(self):
        trade = {
            "entry_price": 100.0,
            "fills": [
                {"qty_fraction": 0.25, "price": 104.0, "reason": "tp1"},
                {"qty_fraction": 0.75, "price": 99.0, "reason": "stop"},
            ],
        }
        # 0.25*4% + 0.75*(-1%) = 0.25%
        self.assertAlmostEqual(m.trade_return_pct(trade), 0.25, places=6)

    def test_non_positive_entry_is_zero_not_a_division_error(self):
        self.assertEqual(m.trade_return_pct({"entry_price": 0.0, "fills": []}), 0.0)


class ClassifyTradesTest(unittest.TestCase):
    def test_tags_fill_status_and_outcome_per_trade(self):
        bars = _bars([
            ("2026-01-02 14:30:00+00:00", 100.0, 99.0),    # closes below -> fills
            ("2026-01-02 14:35:00+00:00", 100.0, 101.0),   # closes above -> misses
        ])
        # signal_price is the level a limit would rest at; entry_price is
        # what the configured fill spec paid. They coincide under "level".
        trades = [
            {"entry_idx": 0, "entry_date": bars["date"][0], "entry_price": 100.0,
             "signal_price": 100.0,
             "fills": [{"qty_fraction": 1.0, "price": 101.0, "reason": "tp1"}]},
            {"entry_idx": 1, "entry_date": bars["date"][1], "entry_price": 100.0,
             "signal_price": 100.0,
             "fills": [{"qty_fraction": 1.0, "price": 99.0, "reason": "stop"}]},
        ]
        out = m.classify_trades(bars, trades, "immediate")
        self.assertEqual([r["filled"] for r in out], [True, False])
        self.assertEqual([r["hit_tp1"] for r in out], [True, False])
        self.assertEqual([r["stopped"] for r in out], [False, True])


class SummarizeTest(unittest.TestCase):
    def test_reports_fill_rate_and_both_cohort_means(self):
        df = pd.DataFrame([
            {"model": "immediate", "filled": True, "ret_pct": 1.0, "hit_tp1": True},
            {"model": "immediate", "filled": True, "ret_pct": -1.0, "hit_tp1": False},
            {"model": "immediate", "filled": False, "ret_pct": 5.0, "hit_tp1": True},
            {"model": "immediate", "filled": False, "ret_pct": 3.0, "hit_tp1": True},
        ])
        s = m.summarize(df).iloc[0]
        self.assertEqual(s["signals"], 4)
        self.assertAlmostEqual(s["fill_rate_pct"], 50.0)
        self.assertAlmostEqual(s["filled_mean_ret"], 0.0)
        self.assertAlmostEqual(s["missed_mean_ret"], 4.0)  # adverse selection
        self.assertAlmostEqual(s["filled_win_pct"], 50.0)
        self.assertAlmostEqual(s["missed_win_pct"], 100.0)

    def test_all_filled_reports_nan_for_the_missed_cohort(self):
        df = pd.DataFrame([
            {"model": "immediate", "filled": True, "ret_pct": 1.0, "hit_tp1": True},
        ])
        s = m.summarize(df).iloc[0]
        self.assertEqual(s["fill_rate_pct"], 100.0)
        self.assertTrue(pd.isna(s["missed_mean_ret"]))


if __name__ == "__main__":
    unittest.main()
