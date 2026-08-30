"""Unit tests for trading_bot.cli.smc_concentration.

The asymmetry between the two columns is the thing to pin. This file's
first draft reported only worst-days-removed under a heading asking
whether the edge was concentrated -- and removing losers can only ever
flatter a positive result, so that table could not answer its own
question. Both directions are tested so the confusion cannot return.
"""

import unittest

import pandas as pd

from trading_bot.cli import smc_concentration as c


def _pnl(day_pnls: dict) -> pd.DataFrame:
    return pd.DataFrame([{"date": d, "symbol": "AAA", "pnl": v, "reason": "stop"}
                         for d, v in day_pnls.items()])


class DependenceTest(unittest.TestCase):
    CAP = 100_000.0

    def test_removing_the_best_days_can_turn_a_winner_negative(self):
        """The whole point: a result carried by two days is not an edge."""
        pnl = _pnl({"d1": 5000.0, "d2": 4000.0, "d3": -1000.0, "d4": -1000.0, "d5": -1000.0})
        dep = c.dependence(pnl, self.CAP, drops=(0, 2))
        self.assertAlmostEqual(dep.loc[0, "best_removed_ret"], 6.0)
        self.assertAlmostEqual(dep.loc[1, "best_removed_ret"], -3.0)

    def test_removing_the_worst_days_only_ever_flatters(self):
        pnl = _pnl({"d1": 5000.0, "d2": -4000.0, "d3": -1000.0})
        rets = list(c.dependence(pnl, self.CAP, drops=(0, 1, 2)).worst_removed_ret)
        self.assertEqual(rets, sorted(rets), "worst-removed must increase monotonically")

    def test_zero_dropped_is_the_real_result_on_both_columns(self):
        dep = c.dependence(_pnl({"d1": 1000.0, "d2": -400.0}), self.CAP, drops=(0,))
        self.assertAlmostEqual(dep.loc[0, "best_removed_ret"], 0.6)
        self.assertAlmostEqual(dep.loc[0, "worst_removed_ret"], 0.6)

    def test_a_broad_edge_survives_losing_its_best_days(self):
        """The control case -- many small winners rather than a few big."""
        dep = c.dependence(_pnl({f"d{i}": 500.0 for i in range(20)}), self.CAP, drops=(0, 5))
        self.assertGreater(dep.loc[1, "best_removed_ret"], 0)

    def test_days_are_aggregated_before_ranking(self):
        """Two trades closing on one date are one day of risk, not two."""
        pnl = pd.DataFrame([
            {"date": "d1", "symbol": "A", "pnl": 3000.0, "reason": "stop"},
            {"date": "d1", "symbol": "B", "pnl": 3000.0, "reason": "stop"},
            {"date": "d2", "symbol": "A", "pnl": -1000.0, "reason": "stop"},
        ])
        dep = c.dependence(pnl, self.CAP, drops=(0, 1))
        self.assertAlmostEqual(dep.loc[1, "best_removed_ret"], -1.0)

    def test_days_left_tracks_the_cut(self):
        dep = c.dependence(_pnl({f"d{i}": 1.0 for i in range(10)}), self.CAP, drops=(0, 3))
        self.assertEqual(list(dep.days_left), [10, 7])


class DailyPnlTest(unittest.TestCase):
    def test_fifo_matches_sells_against_earlier_buys(self):
        result = {"trades": [
            {"timestamp_iso": "2026-01-02T15:00:00+00:00", "symbol": "AAA",
             "side": "BUY", "size": 10, "fill_price": 100.0, "reason": "entry"},
            {"timestamp_iso": "2026-01-02T19:00:00+00:00", "symbol": "AAA",
             "side": "SELL", "size": 10, "fill_price": 110.0, "reason": "stop"},
        ]}
        pnl = c.daily_pnl(result)
        self.assertEqual(len(pnl), 1)
        self.assertAlmostEqual(pnl.pnl.iloc[0], 100.0)

    def test_a_partial_exit_is_not_scored_as_a_completed_trade(self):
        """TP1 and the exit closing the remainder are separate fills of
        one round trip; each books only its own share."""
        result = {"trades": [
            {"timestamp_iso": "2026-01-02T15:00:00+00:00", "symbol": "AAA",
             "side": "BUY", "size": 10, "fill_price": 100.0, "reason": "entry"},
            {"timestamp_iso": "2026-01-02T16:00:00+00:00", "symbol": "AAA",
             "side": "SELL", "size": 4, "fill_price": 105.0, "reason": "tp1"},
            {"timestamp_iso": "2026-01-02T19:00:00+00:00", "symbol": "AAA",
             "side": "SELL", "size": 6, "fill_price": 95.0, "reason": "stop"},
        ]}
        pnl = c.daily_pnl(result).set_index("reason").pnl
        self.assertAlmostEqual(pnl["tp1"], 20.0)
        self.assertAlmostEqual(pnl["stop"], -30.0)

    def test_pnl_lands_on_the_sell_date(self):
        result = {"trades": [
            {"timestamp_iso": "2026-01-02T15:00:00+00:00", "symbol": "AAA",
             "side": "BUY", "size": 10, "fill_price": 100.0, "reason": "entry"},
            {"timestamp_iso": "2026-01-05T14:35:00+00:00", "symbol": "AAA",
             "side": "SELL", "size": 10, "fill_price": 90.0, "reason": "stop"},
        ]}
        self.assertEqual(str(c.daily_pnl(result).date.iloc[0]), "2026-01-05")


class TopDaysTest(unittest.TestCase):
    def test_reports_the_largest_days_as_a_share_of_capital(self):
        best = c.top_days(_pnl({"d1": 2000.0, "d2": 500.0, "d3": -100.0}), 100_000.0, n=2)
        self.assertEqual(list(best.date), ["d1", "d2"])
        self.assertAlmostEqual(best.pct_of_capital.iloc[0], 2.0)
