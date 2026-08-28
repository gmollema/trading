"""Unit tests for trading_bot.cli.smc_full_backtest.

The pieces worth pinning are the ones that would silently corrupt a
comparison across cost bases: sharing candidate generation between bases
that use the same slippage, and passing each basis's own commission
through to the simulator. A mix-up in either would produce a table that
looks fine and compares nothing.
"""

import unittest

import pandas as pd

from trading_bot.cli import smc_full_backtest as f


class SlippageKeyTest(unittest.TestCase):
    def test_none_has_its_own_identity(self):
        self.assertEqual(f.slippage_key(None), "none")

    def test_same_dict_in_any_order_is_one_key(self):
        # Candidate generation is shared off this key, so ordering must not
        # split one setting into two expensive builds.
        a = f.slippage_key({"entry": 64.0, "stop": 2.0})
        b = f.slippage_key({"stop": 2.0, "entry": 64.0})
        self.assertEqual(a, b)

    def test_different_values_are_different_keys(self):
        self.assertNotEqual(
            f.slippage_key({"entry": 64.0}), f.slippage_key({"entry": 10.0})
        )


class GroupBySlippageTest(unittest.TestCase):
    def test_bases_sharing_slippage_collapse_to_one_build(self):
        # zero_cost + both commission-only bases share slippage=None, and
        # both realistic bases share MEASURED_SLIPPAGE: 5 bases, 2 builds.
        groups = f.group_by_slippage(f.COST_BASES)
        self.assertEqual(len(groups), 2)
        self.assertIn("none", groups)

    def test_every_basis_maps_to_a_built_group(self):
        groups = f.group_by_slippage(f.COST_BASES)
        for basis in f.COST_BASES:
            self.assertIn(f.slippage_key(basis["slippage"]), groups)


class CostBasesTest(unittest.TestCase):
    def test_zero_cost_basis_really_is_costless(self):
        zero = next(b for b in f.COST_BASES if b["name"] == "zero_cost")
        self.assertIsNone(zero["slippage"])
        self.assertIsNone(zero["commission"])

    def test_measured_slippage_prices_the_stop_far_below_market_legs(self):
        # The stop rests as a real StopOrder and measured 0-5.9 bps live;
        # the market legs measured 48.5 and 79.4. Flattening these into one
        # rate is what made the first sensitivity run too pessimistic.
        self.assertLess(f.MEASURED_SLIPPAGE["stop"], f.MEASURED_SLIPPAGE["tp1"])
        self.assertEqual(f.MEASURED_SLIPPAGE["stop"], f.STOP_SLIPPAGE_BPS)
        self.assertEqual(f.MEASURED_SLIPPAGE["tp1"], f.MARKET_LEG_SLIPPAGE_BPS)

    def test_entry_borrows_the_tp1_rate_as_an_explicit_assumption(self):
        self.assertEqual(f.ENTRY_SLIPPAGE_BPS, f.MARKET_LEG_SLIPPAGE_BPS)
        self.assertEqual(f.MEASURED_SLIPPAGE["entry"], f.ENTRY_SLIPPAGE_BPS)

    def test_tiered_is_cheaper_than_fixed_on_both_terms(self):
        self.assertLess(f.TIERED[0], f.FIXED[0])
        self.assertLess(f.TIERED[1], f.FIXED[1])


class RunOneTest(unittest.TestCase):
    RULES = {"risk": {"max_risk_per_trade_pct": 1.0,
                      "max_position_size_pct_of_portfolio": 10.0,
                      "max_concurrent_positions": 2}}

    def setUp(self):
        d = pd.Timestamp
        self.cands = [
            (d("2025-02-01 15:00:00+00:00"), "A", {}),
            (d("2025-07-01 15:00:00+00:00"), "B", {}),
        ]
        self.calls = []

        def fake_sim(window, capital, **kw):
            self.calls.append((len(window), kw))
            return {"equity_curve": [{"equity": capital}, {"equity": capital * 1.02}],
                    "trades": []}

        self._orig = f.simulate_smc_portfolio
        f.simulate_smc_portfolio = fake_sim

    def tearDown(self):
        f.simulate_smc_portfolio = self._orig

    def test_commission_is_passed_through_per_basis(self):
        f.run_one(self.cands, self.RULES, 1000.0, f.TIERED)
        _, kw = self.calls[0]
        self.assertEqual(kw["commission_per_share"], 0.0035)
        self.assertEqual(kw["commission_min"], 0.35)

    def test_none_commission_means_no_cost_model(self):
        f.run_one(self.cands, self.RULES, 1000.0, None)
        _, kw = self.calls[0]
        self.assertIsNone(kw["commission_per_share"])

    def test_window_filters_by_entry_date(self):
        lo = pd.Timestamp("2025-06-01 00:00:00+00:00")
        hi = pd.Timestamp("2025-12-01 00:00:00+00:00")
        f.run_one(self.cands, self.RULES, 1000.0, None, lo, hi)
        self.assertEqual(self.calls[0][0], 1)  # only the July candidate

    def test_empty_window_returns_none_rather_than_a_zero_row(self):
        lo = pd.Timestamp("2030-01-01 00:00:00+00:00")
        hi = pd.Timestamp("2030-02-01 00:00:00+00:00")
        self.assertIsNone(f.run_one(self.cands, self.RULES, 1000.0, None, lo, hi))
        self.assertEqual(self.calls, [])

    def test_risk_settings_come_from_the_rules_file(self):
        f.run_one(self.cands, self.RULES, 1000.0, None)
        _, kw = self.calls[0]
        self.assertEqual(kw["risk_pct"], 1.0)
        self.assertEqual(kw["max_position_pct"], 10.0)
        self.assertEqual(kw["max_concurrent_positions"], 2)

    def test_reports_signal_count_alongside_the_metrics(self):
        stats = f.run_one(self.cands, self.RULES, 1000.0, None)
        self.assertEqual(stats["signals"], 2)
        self.assertAlmostEqual(stats["ret_pct"], 2.0)


if __name__ == "__main__":
    unittest.main()
