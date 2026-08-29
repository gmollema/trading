"""Unit tests for trading_bot.cli.smc_entry_breakeven.

The sweep needs the full bar cache. What is testable in isolation is the
zero crossing -- which is the whole output, and which would be easy to
report confidently from a curve that never actually crosses.
"""

import unittest

import pandas as pd

from trading_bot.cli import smc_entry_breakeven as be
from trading_bot.cli import smc_full_backtest as f


def _curve(pairs):
    return pd.DataFrame({"ret_pct": [v for _, v in pairs]},
                        index=pd.Index([r for r, _ in pairs], name="entry_slippage_bps"))


class CrossingTest(unittest.TestCase):
    def test_interpolates_between_the_bracketing_rates(self):
        # +1.0 at 0 bps, -1.0 at 10 bps -> halfway.
        self.assertAlmostEqual(be.crossing(_curve([(0.0, 1.0), (10.0, -1.0)])), 5.0)

    def test_lands_on_an_exact_zero(self):
        self.assertAlmostEqual(be.crossing(_curve([(0.0, 2.0), (4.0, 0.0), (8.0, -2.0)])), 4.0)

    def test_a_curve_that_never_crosses_returns_none(self):
        """Distinguishable from "crosses at the last point tested" -- the
        caller says "no crossing in range" rather than quoting an edge."""
        self.assertIsNone(be.crossing(_curve([(0.0, 3.0), (10.0, 2.0), (25.0, 1.0)])))

    def test_a_curve_already_negative_returns_none(self):
        self.assertIsNone(be.crossing(_curve([(0.0, -1.0), (10.0, -2.0)])))

    def test_takes_the_first_crossing_when_the_curve_is_noisy(self):
        """Sampled on a coarse grid over a small trade count, so it can
        wobble; the first downward crossing is the honest threshold."""
        self.assertAlmostEqual(be.crossing(_curve([(0.0, 1.0), (5.0, -1.0), (10.0, 1.0)])), 2.5)


class SlippageAtTest(unittest.TestCase):
    def test_only_the_entry_rate_moves(self):
        """The question is what the ENTRY can cost, so every other leg has
        to stay exactly where the configured basis put it."""
        base = f.leg_slippage("next_open", "next_open")
        swept = be.slippage_at(37.0, "next_open", "next_open", False)
        self.assertEqual(swept["entry"], 37.0)
        self.assertEqual({k: v for k, v in swept.items() if k != "entry"},
                         {k: v for k, v in base.items() if k != "entry"})

    def test_the_source_dict_is_not_mutated(self):
        be.slippage_at(37.0, "next_open", "next_open", False)
        self.assertEqual(f.leg_slippage("next_open", "next_open")["entry"],
                         f.RESIDUAL_ENTRY_SLIPPAGE_BPS)


class RateGridTest(unittest.TestCase):
    def test_the_configured_rate_is_on_the_grid(self):
        """So the swept curve includes the point the published figures
        were produced at, and the comparison is direct."""
        self.assertIn(f.RESIDUAL_ENTRY_SLIPPAGE_BPS, be.DEFAULT_RATES)

    def test_the_grid_is_ascending_and_starts_frictionless(self):
        self.assertEqual(be.DEFAULT_RATES, sorted(be.DEFAULT_RATES))
        self.assertEqual(be.DEFAULT_RATES[0], 0.0)

    def test_the_deciding_basis_is_one_of_the_swept_ones(self):
        self.assertIn(be.DECIDING_BASIS, [b["name"] for b in be.COMMISSIONS])
