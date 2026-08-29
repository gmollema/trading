"""Unit tests for trading_bot.cli.smc_exit_spec.

The sweep needs the full bar cache; what is testable in isolation is the
spec table, where a wrong rate still produces a plausible-looking result.
"""

import unittest

from trading_bot.backtest.smc_signals import EXIT_FILLS
from trading_bot.cli import smc_exit_spec as e
from trading_bot.cli import smc_full_backtest as f


class ExitSpecsTest(unittest.TestCase):
    def test_every_fill_is_one_the_signal_layer_accepts(self):
        for spec in e.EXIT_SPECS:
            self.assertIn(spec["exit_fill"], EXIT_FILLS, spec["name"])

    def test_names_are_unique(self):
        names = [s["name"] for s in e.EXIT_SPECS]
        self.assertEqual(len(names), len(set(names)))

    def test_the_historical_spec_is_included_as_the_reference(self):
        level = next(s for s in e.EXIT_SPECS if s["name"] == "level")
        self.assertEqual(level["exit_fill"], "level")
        self.assertFalse(level["tp1_resting_limit"])

    def test_the_resting_limit_variant_only_changes_tp1(self):
        plain = next(s for s in e.EXIT_SPECS if s["name"] == "next_open")
        limit = next(s for s in e.EXIT_SPECS if s["name"] == "next_open_tp1_limit")
        self.assertEqual(plain["exit_fill"], limit["exit_fill"])
        self.assertTrue(limit["tp1_resting_limit"])

        a = f.leg_slippage("next_open", plain["exit_fill"], plain["tp1_resting_limit"])
        b = f.leg_slippage("next_open", limit["exit_fill"], limit["tp1_resting_limit"])
        self.assertEqual({k: v for k, v in a.items() if k != "tp1"},
                         {k: v for k, v in b.items() if k != "tp1"})
        self.assertNotEqual(a["tp1"], b["tp1"])
