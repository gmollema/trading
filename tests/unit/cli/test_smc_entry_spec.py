"""Unit tests for trading_bot.cli.smc_entry_spec.

The sweep itself needs the full bar cache; what is testable in isolation is
the wiring that decides WHICH slippage each (spec, basis) pair runs on --
which is also where a mistake would be invisible, since a wrong entry rate
still produces a plausible-looking table.
"""

import unittest

from trading_bot.cli import smc_entry_spec as e
from trading_bot.cli import smc_full_backtest as f


class SpecSlippageTest(unittest.TestCase):
    LEVEL = next(s for s in e.ENTRY_SPECS if s["name"] == "level")
    NEXT_OPEN = next(s for s in e.ENTRY_SPECS if s["name"] == "next_open")

    def test_costless_basis_stays_costless(self):
        self.assertIsNone(e.spec_slippage(None, self.NEXT_OPEN))

    def test_entry_rate_comes_from_the_spec(self):
        slipped = e.spec_slippage(f.MEASURED_SLIPPAGE, self.NEXT_OPEN)
        self.assertEqual(slipped["entry"], f.RESIDUAL_ENTRY_SLIPPAGE_BPS)

    def test_exit_rates_are_untouched(self):
        """The exits are not respecified here, so they keep the full
        market-leg rate under every entry spec."""
        for spec in e.ENTRY_SPECS:
            slipped = e.spec_slippage(f.MEASURED_SLIPPAGE, spec)
            self.assertEqual(slipped["tp1"], f.MEASURED_SLIPPAGE["tp1"])
            self.assertEqual(slipped["new_high_exit"], f.MEASURED_SLIPPAGE["new_high_exit"])

    def test_the_source_dict_is_not_mutated(self):
        e.spec_slippage(f.MEASURED_SLIPPAGE, self.NEXT_OPEN)
        self.assertEqual(f.MEASURED_SLIPPAGE["entry"], f.MARKET_LEG_SLIPPAGE_BPS)


class EntrySpecsTest(unittest.TestCase):
    def test_level_specs_keep_the_chase_as_bps(self):
        for spec in e.ENTRY_SPECS:
            expected = f.entry_slippage_bps(spec["entry_fill"])
            self.assertEqual(spec["entry_slippage_bps"], expected, spec["name"])

    def test_every_fill_is_one_the_signal_layer_accepts(self):
        from trading_bot.backtest.smc_signals import ENTRY_FILLS
        for spec in e.ENTRY_SPECS:
            self.assertIn(spec["entry_fill"], ENTRY_FILLS)

    def test_names_are_unique(self):
        names = [s["name"] for s in e.ENTRY_SPECS]
        self.assertEqual(len(names), len(set(names)))
