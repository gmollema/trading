"""Unit tests for trading_bot.cli.smc_rr_walkforward.

Covers the pure pieces -- R:R measurement, candidate filtering, fold
construction, round-trip pairing and metrics -- with synthetic inputs, so
nothing here touches the real backtest_data/ cache or runs a simulation.

round_trip_pnls carries most of the risk: simulate_smc_portfolio emits
individual FILLS, so a TP1 partial and the exit closing the remainder are
separate rows belonging to one round trip. Counting them as two would
inflate both the trade count and the win rate.
"""

import unittest

import pandas as pd

from trading_bot.cli import smc_rr_walkforward as w


def _trade(entry, stop, tp1):
    return {"entry_price": entry, "initial_stop_price": stop, "tp1_price": tp1}


def _fill(symbol, side, size, price):
    return {"symbol": symbol, "side": side, "size": size, "fill_price": price}


class SignalRRTest(unittest.TestCase):
    def test_measures_reward_over_risk(self):
        self.assertAlmostEqual(w.signal_rr(_trade(100.0, 99.0, 102.0)), 2.0)

    def test_sub_one_when_target_is_nearer_than_stop(self):
        self.assertAlmostEqual(w.signal_rr(_trade(100.0, 99.0, 100.5)), 0.5)

    def test_none_without_a_tp1(self):
        self.assertIsNone(w.signal_rr(_trade(100.0, 99.0, None)))

    def test_none_when_stop_is_not_below_entry(self):
        # Guards the zero/negative denominator rather than dividing by it.
        self.assertIsNone(w.signal_rr(_trade(100.0, 100.0, 105.0)))
        self.assertIsNone(w.signal_rr(_trade(100.0, 101.0, 105.0)))


class FilterByRRTest(unittest.TestCase):
    def setUp(self):
        d = pd.Timestamp("2026-01-02 15:00:00+00:00")
        self.cands = [
            (d, "LOW", _trade(100.0, 99.0, 100.5)),    # rr 0.5
            (d, "MID", _trade(100.0, 99.0, 102.0)),    # rr 2.0
            (d, "NONE", _trade(100.0, 99.0, None)),    # undefined
        ]

    def test_threshold_keeps_only_signals_at_or_above_it(self):
        kept = w.filter_by_rr(self.cands, 1.0, drop_no_tp1=True)
        self.assertEqual([c[1] for c in kept], ["MID"])

    def test_threshold_is_inclusive(self):
        kept = w.filter_by_rr(self.cands, 0.5, drop_no_tp1=True)
        self.assertEqual([c[1] for c in kept], ["LOW", "MID"])

    def test_no_tp1_signals_kept_when_not_dropping(self):
        kept = w.filter_by_rr(self.cands, 1.0, drop_no_tp1=False)
        self.assertEqual([c[1] for c in kept], ["MID", "NONE"])

    def test_no_tp1_signals_bypass_the_threshold_entirely(self):
        # They have no R:R to compare, so a high threshold must not be
        # read as excluding them when drop_no_tp1 is False.
        kept = w.filter_by_rr(self.cands, 99.0, drop_no_tp1=False)
        self.assertEqual([c[1] for c in kept], ["NONE"])

    def test_zero_threshold_keeps_everything(self):
        self.assertEqual(len(w.filter_by_rr(self.cands, 0.0, drop_no_tp1=False)), 3)


class ExpandingFoldsTest(unittest.TestCase):
    def test_n_boundaries_give_n_minus_one_folds(self):
        folds = w.expanding_folds("2025-01-01", ["2025-06-01", "2025-08-01", "2025-10-01"])
        self.assertEqual(len(folds), 2)

    def test_fit_start_is_fixed_and_fit_end_expands(self):
        folds = w.expanding_folds("2025-01-01", ["2025-06-01", "2025-08-01", "2025-10-01"])
        self.assertEqual([f["fit_start"] for f in folds],
                         [pd.Timestamp("2025-01-01")] * 2)
        self.assertEqual([f["fit_end"] for f in folds],
                         [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-08-01")])

    def test_each_test_window_follows_its_own_fit_window(self):
        for f in w.expanding_folds("2025-01-01", ["2025-06-01", "2025-08-01", "2025-10-01"]):
            self.assertLess(f["fit_start"], f["fit_end"])
            self.assertLess(f["fit_end"], f["test_end"])

    def test_single_boundary_yields_no_folds(self):
        self.assertEqual(w.expanding_folds("2025-01-01", ["2025-06-01"]), [])


class RoundTripPnlsTest(unittest.TestCase):
    def test_simple_win_and_loss(self):
        fills = [
            _fill("AAA", "BUY", 10, 100.0), _fill("AAA", "SELL", 10, 110.0),
            _fill("BBB", "BUY", 10, 100.0), _fill("BBB", "SELL", 10, 95.0),
        ]
        self.assertEqual(w.round_trip_pnls(fills), [100.0, -50.0])

    def test_tp1_partial_plus_exit_is_one_round_trip(self):
        # 25% off at +4, the rest stopped at -1: one trade, net +5.
        fills = [
            _fill("AAA", "BUY", 100, 100.0),
            _fill("AAA", "SELL", 25, 104.0),
            _fill("AAA", "SELL", 75, 99.0),
        ]
        pnls = w.round_trip_pnls(fills)
        self.assertEqual(len(pnls), 1)
        self.assertAlmostEqual(pnls[0], 25 * 4.0 + 75 * -1.0)

    def test_partial_sale_alone_is_not_booked(self):
        # An open position with only its TP1 filled must not count as a
        # completed winner -- that would inflate the win rate.
        fills = [_fill("AAA", "BUY", 100, 100.0), _fill("AAA", "SELL", 25, 104.0)]
        self.assertEqual(w.round_trip_pnls(fills), [])

    def test_lots_are_matched_fifo_per_symbol(self):
        fills = [
            _fill("AAA", "BUY", 10, 100.0),
            _fill("AAA", "BUY", 10, 200.0),
            _fill("AAA", "SELL", 10, 110.0),   # closes the 100.0 lot: +100
            _fill("AAA", "SELL", 10, 190.0),   # closes the 200.0 lot: -100
        ]
        self.assertEqual(w.round_trip_pnls(fills), [100.0, -100.0])

    def test_symbols_do_not_bleed_into_each_other(self):
        fills = [
            _fill("AAA", "BUY", 10, 100.0),
            _fill("BBB", "BUY", 10, 50.0),
            _fill("BBB", "SELL", 10, 55.0),
            _fill("AAA", "SELL", 10, 90.0),
        ]
        self.assertEqual(w.round_trip_pnls(fills), [50.0, -100.0])

    def test_sell_without_an_open_lot_is_ignored(self):
        self.assertEqual(w.round_trip_pnls([_fill("AAA", "SELL", 10, 100.0)]), [])

    def test_no_fills_gives_no_round_trips(self):
        self.assertEqual(w.round_trip_pnls([]), [])


class SummarizeTest(unittest.TestCase):
    def _result(self, equities, trades):
        return {"equity_curve": [{"equity": e} for e in equities], "trades": trades}

    def test_return_and_drawdown_from_the_equity_curve(self):
        r = self._result([100.0, 120.0, 90.0, 110.0], [])
        s = w.summarize(r, 100.0)
        self.assertAlmostEqual(s["ret_pct"], 10.0)
        self.assertAlmostEqual(s["max_dd_pct"], -25.0)  # 120 -> 90

    def test_profit_factor_and_win_rate_from_round_trips(self):
        trades = [
            _fill("AAA", "BUY", 10, 100.0), _fill("AAA", "SELL", 10, 110.0),   # +100
            _fill("BBB", "BUY", 10, 100.0), _fill("BBB", "SELL", 10, 95.0),    # -50
        ]
        s = w.summarize(self._result([100.0, 105.0], trades), 100.0)
        self.assertEqual(s["trades"], 2)
        self.assertAlmostEqual(s["pf"], 2.0)
        self.assertAlmostEqual(s["win_rate_pct"], 50.0)

    def test_profit_factor_is_infinite_with_no_losses(self):
        trades = [_fill("AAA", "BUY", 10, 100.0), _fill("AAA", "SELL", 10, 110.0)]
        s = w.summarize(self._result([100.0, 110.0], trades), 100.0)
        self.assertEqual(s["pf"], float("inf"))

    def test_empty_equity_curve_is_reported_as_flat_not_an_error(self):
        s = w.summarize({"equity_curve": [], "trades": []}, 100.0)
        self.assertEqual(s["trades"], 0)
        self.assertEqual(s["ret_pct"], 0.0)
        self.assertEqual(s["max_dd_pct"], 0.0)

    def test_monotonic_curve_has_no_drawdown(self):
        s = w.summarize(self._result([100.0, 105.0, 110.0], []), 100.0)
        self.assertAlmostEqual(s["max_dd_pct"], 0.0)


class AlignTzTest(unittest.TestCase):
    def test_naive_adopts_the_references_timezone(self):
        out = w.align_tz(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-01 00:00:00-04:00"))
        self.assertIsNotNone(out.tz)
        self.assertEqual(out.tz_localize(None), pd.Timestamp("2025-01-01"))

    def test_aware_is_stripped_against_a_naive_reference(self):
        out = w.align_tz(pd.Timestamp("2025-01-01 00:00:00-04:00"), pd.Timestamp("2025-06-01"))
        self.assertIsNone(out.tz)

    def test_matching_awareness_is_left_alone(self):
        ts = pd.Timestamp("2025-01-01 00:00:00-04:00")
        self.assertEqual(w.align_tz(ts, pd.Timestamp("2025-06-01 00:00:00-04:00")), ts)

    def test_no_reference_leaves_the_timestamp_untouched(self):
        ts = pd.Timestamp("2025-01-01")
        self.assertEqual(w.align_tz(ts, None), ts)


class RunSweepTest(unittest.TestCase):
    """run_sweep's own wiring, with the simulator stubbed out."""

    def setUp(self):
        self.rules = {"risk": {"max_risk_per_trade_pct": 1.0,
                               "max_position_size_pct_of_portfolio": 10.0,
                               "max_concurrent_positions": 2}}
        d = pd.Timestamp
        self.cands = [
            (d("2025-02-01 15:00:00+00:00"), "A", _trade(100.0, 99.0, 100.5)),  # rr 0.5, fit
            (d("2025-07-01 15:00:00+00:00"), "B", _trade(100.0, 99.0, 103.0)),  # rr 3.0, test
            (d("2025-07-02 15:00:00+00:00"), "C", _trade(100.0, 99.0, None)),   # no tp1, test
        ]
        self.folds = w.expanding_folds("2025-01-01", ["2025-06-01", "2025-12-01"])

    def _stub(self, monkey_result=None):
        seen = []

        def fake_sim(window, capital, **kw):
            seen.append((len(window), kw))
            return {"equity_curve": [{"equity": capital}, {"equity": capital * 1.01}],
                    "trades": []}

        return fake_sim, seen

    def test_sweeps_every_combination_and_passes_risk_settings_through(self):
        fake, seen = self._stub()
        orig = w.simulate_smc_portfolio
        w.simulate_smc_portfolio = fake
        try:
            df = w.run_sweep(self.cands, self.folds, [0.0, 2.0], self.rules,
                             initial_capital=1000.0, verbose=False)
        finally:
            w.simulate_smc_portfolio = orig

        # 2 thresholds x 2 drop_no_tp1 x 1 fold x 2 phases, minus empty windows.
        self.assertEqual(set(df["min_rr"]), {0.0, 2.0})
        self.assertEqual(set(df["drop_no_tp1"]), {False, True})
        self.assertEqual(set(df["phase"]), {"fit", "test"})
        for _, kw in seen:
            self.assertEqual(kw["risk_pct"], 1.0)
            self.assertEqual(kw["max_position_pct"], 10.0)
            self.assertEqual(kw["max_concurrent_positions"], 2)

    def test_higher_threshold_never_widens_the_signal_set(self):
        fake, _ = self._stub()
        orig = w.simulate_smc_portfolio
        w.simulate_smc_portfolio = fake
        try:
            df = w.run_sweep(self.cands, self.folds, [0.0, 2.0], self.rules,
                             initial_capital=1000.0, drop_no_tp1_options=(True,),
                             verbose=False)
        finally:
            w.simulate_smc_portfolio = orig
        test = df[df.phase == "test"].set_index("min_rr")["signals"]
        self.assertGreaterEqual(test.loc[0.0], test.loc[2.0])


if __name__ == "__main__":
    unittest.main()
