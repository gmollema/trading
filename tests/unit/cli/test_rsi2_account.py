"""Unit tests for cli.rsi2_account.

The thing most easily got wrong here is accruing collateral yield only
while flat. A futures position consumes margin, not cash, so the yield
runs every day including days a contract is open -- and getting that
wrong changes the headline CAGR by more than the strategy contributes at
larger capital bases.
"""

import argparse
import unittest

from trading_bot.backtest.rsi2_engine import ES, MES
from trading_bot.cli import rsi2_account as acct


def make_args(**kw):
    defaults = dict(contract="ES", contracts=1, margin_scaled=False, max_margin_pct=25.0,
                    flat_yield=None, no_yield=False, start="2005-01-01", slippage_ticks=0.0,
                    entry_level=10.0, exit_level=70.0, sma_period=200)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def bars_and_trade():
    """Ten bars in 2005, one trade holding bars 2-4 for +10 points."""
    import pandas as pd
    dates = list(pd.date_range("2005-01-03", periods=10, freq="B", tz="UTC"))
    closes = [100.0] * 10
    closes[4] = 110.0
    bars = {"date": dates, "open": list(closes), "high": closes, "low": closes, "close": closes}
    trade = {"entry_idx": 2, "exit_idx": 4, "entry_price": 100.0, "points": 10.0}
    return bars, [trade]


class TestYieldAccrual(unittest.TestCase):
    def test_no_yield_leaves_only_strategy_pnl(self):
        bars, trades = bars_and_trade()
        r = acct.run_account(trades, bars, 100_000.0, ES, make_args(no_yield=True))
        # 10 points * $50 * 1 contract, less two commissions.
        self.assertAlmostEqual(r["final_equity"], 100_000 + 500 - 2 * ES.commission_per_side, delta=1)
        self.assertEqual(r["interest_earned"], 0)

    def test_yield_accrues_on_every_bar_including_held_ones(self):
        bars, trades = bars_and_trade()
        r = acct.run_account(trades, bars, 100_000.0, ES, make_args(flat_yield=4.0))
        # 10 bars at 4%/252 on ~$100k is about $159; if it only accrued on
        # the 7 flat bars it would be about $111.
        self.assertGreater(r["interest_earned"], 150)
        self.assertLess(r["interest_earned"], 170)

    def test_zero_contracts_is_a_pure_cash_account(self):
        bars, trades = bars_and_trade()
        r = acct.run_account(trades, bars, 100_000.0, ES, make_args(contracts=0, flat_yield=4.0))
        self.assertEqual(r["contract_days"], 0)
        self.assertEqual(r["interest_share_pct"], 100.0)
        self.assertGreater(r["final_equity"], 100_000)

    def test_cash_only_return_is_scale_invariant(self):
        bars, trades = bars_and_trade()
        a = acct.run_account(trades, bars, 50_000.0, ES, make_args(contracts=0, flat_yield=4.0))
        b = acct.run_account(trades, bars, 500_000.0, ES, make_args(contracts=0, flat_yield=4.0))
        self.assertAlmostEqual(a["total_return_pct"], b["total_return_pct"], places=3)


class TestRateSelection(unittest.TestCase):
    def test_flat_overrides_the_historical_path(self):
        self.assertEqual(acct.rate_for(2013, make_args(flat_yield=4.0)), 4.0)

    def test_historical_path_is_used_by_default(self):
        # The ZIRP years are the whole reason a flat rate misleads.
        self.assertLess(acct.rate_for(2013, make_args()), 0.5)
        self.assertGreater(acct.rate_for(2023, make_args()), 4.0)

    def test_no_yield_wins_over_both(self):
        self.assertEqual(acct.rate_for(2023, make_args(no_yield=True, flat_yield=4.0)), 0.0)

    def test_unknown_year_is_zero_not_a_crash(self):
        self.assertEqual(acct.rate_for(1970, make_args()), 0.0)

    def test_flat_rate_beats_historical_over_the_zirp_decade(self):
        bars, trades = bars_and_trade()
        flat = acct.run_account(trades, bars, 100_000.0, ES, make_args(flat_yield=4.0))
        hist = acct.run_account(trades, bars, 100_000.0, ES, make_args())
        # 2005 sat near 3.2%, so flat 4% is only slightly ahead here; the
        # point is that they differ at all and flat is the higher one.
        self.assertGreater(flat["interest_earned"], hist["interest_earned"])


class TestSizing(unittest.TestCase):
    def test_margin_scaled_takes_more_contracts_with_more_capital(self):
        bars, trades = bars_and_trade()
        small = acct.run_account(trades, bars, 100_000.0, MES, make_args(margin_scaled=True))
        large = acct.run_account(trades, bars, 1_000_000.0, MES, make_args(margin_scaled=True))
        self.assertGreater(large["contract_days"], small["contract_days"])

    def test_margin_cap_limits_the_count(self):
        bars, trades = bars_and_trade()
        # 25% of $100k is $25k; MES margin is $2,400, so 10 contracts,
        # held for the 3 bars of the trade (entry bar counted at mark time).
        r = acct.run_account(trades, bars, 100_000.0, MES, make_args(margin_scaled=True))
        self.assertEqual(r["contract_days"] % 10, 0)

    def test_fixed_contracts_scale_pnl_linearly(self):
        bars, trades = bars_and_trade()
        one = acct.run_account(trades, bars, 1_000_000.0, ES, make_args(contracts=1, no_yield=True))
        two = acct.run_account(trades, bars, 1_000_000.0, ES, make_args(contracts=2, no_yield=True))
        gain_one = one["final_equity"] - 1_000_000
        gain_two = two["final_equity"] - 1_000_000
        self.assertAlmostEqual(gain_two, gain_one * 2, delta=1)

    def test_slippage_costs_money(self):
        bars, trades = bars_and_trade()
        clean = acct.run_account(trades, bars, 100_000.0, ES, make_args(no_yield=True))
        dirty = acct.run_account(trades, bars, 100_000.0, ES,
                                 make_args(no_yield=True, slippage_ticks=2.0))
        self.assertLess(dirty["final_equity"], clean["final_equity"])


class TestReportedFields(unittest.TestCase):
    def test_interest_share_measures_how_much_is_just_interest(self):
        bars, trades = bars_and_trade()
        # A large base makes one contract's P&L negligible next to interest.
        r = acct.run_account(trades, bars, 10_000_000.0, ES, make_args(flat_yield=4.0))
        self.assertGreater(r["interest_share_pct"], 90.0)

    def test_drawdown_is_reported_and_non_negative(self):
        bars, trades = bars_and_trade()
        r = acct.run_account(trades, bars, 100_000.0, ES, make_args(no_yield=True))
        self.assertGreaterEqual(r["max_dd_pct"], 0.0)

class TestSameBarTrade(unittest.TestCase):
    """A trade can open at a bar's OPEN and close at that same bar's CLOSE
    (entry_timing="next_open" with exit_timing="close"). Processing exits
    before entries left such a trade open forever, blocking every later
    trade -- it reported 1.98% CAGR on unchanged net points."""

    @staticmethod
    def bars_and_trades():
        import pandas as pd
        dates = list(pd.date_range("2005-01-03", periods=8, freq="B", tz="UTC"))
        closes = [100.0] * 8
        bars = {"date": dates, "open": list(closes), "high": closes, "low": closes, "close": closes}
        return bars, [
            {"entry_idx": 1, "exit_idx": 1, "entry_price": 100.0, "points": 4.0},
            {"entry_idx": 3, "exit_idx": 5, "entry_price": 100.0, "points": 6.0},
        ]

    def test_same_bar_trade_closes_and_does_not_block_later_trades(self):
        bars, trades = self.bars_and_trades()
        r = acct.run_account(trades, bars, 100_000.0, ES, make_args(no_yield=True))
        expected = 100_000 + (4.0 + 6.0) * ES.multiplier - 2 * 2 * ES.commission_per_side
        self.assertAlmostEqual(r["final_equity"], expected, delta=1)

    def test_the_later_trade_is_not_swallowed(self):
        bars, trades = self.bars_and_trades()
        both = acct.run_account(trades, bars, 100_000.0, ES, make_args(no_yield=True))
        first_only = acct.run_account(trades[:1], bars, 100_000.0, ES, make_args(no_yield=True))
        self.assertGreater(both["final_equity"], first_only["final_equity"])

if __name__ == "__main__":
    unittest.main()
