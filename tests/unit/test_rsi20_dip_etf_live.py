"""Unit tests for trading_bot.rsi20_dip_etf_live.

The signal itself is already covered by test_rsi20_dip_signals and
test_rsi20_dip_live -- this module re-exports `decide` rather than
reimplementing it, so what is tested here is what the ETF variant
actually adds: rules validation, the equity cap, and share sizing.

One test asserts that `decide` really is the same function object as the
futures variant's, because the whole safety argument for this variant is
that it runs the code the backtest validated.

This variant was rewritten from a whole-share, 3x-leveraged ETN (3USL)
to fractional, unleveraged SPY at a hard $25 position ceiling. Tests
below reflect the current SPY-direct design; comments call out where a
number or invariant differs from the retired leveraged-ETP version.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_bot import rsi20_dip_etf_live as etf
from trading_bot import rsi20_dip_live

RULES = dict(etf.DEFAULT_RULES)


def write_rules(directory, payload) -> Path:
    path = Path(directory) / "rules.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestSignalLogicIsShared(unittest.TestCase):
    def test_decide_is_the_futures_variants_function(self):
        """Not a copy. A second implementation is a second thing to keep
        in step with the backtest."""
        self.assertIs(etf.decide, rsi20_dip_live.decide)
        self.assertIs(etf.bars_needed, rsi20_dip_live.bars_needed)


class TestLoadRules(unittest.TestCase):
    def test_defaults_when_the_file_is_absent(self):
        rules = etf.load_rules(Path("does-not-exist.json"))
        self.assertEqual(rules["entry_level"], 60)
        self.assertEqual(rules["exit_level"], 65)

    def test_file_overrides_merge_over_defaults(self):
        with TemporaryDirectory() as d:
            path = write_rules(d, {"capital": 250.0, "trade_symbol": "XS2D"})
            rules = etf.load_rules(path)
            self.assertEqual(rules["capital"], 250.0)
            self.assertEqual(rules["trade_symbol"], "XS2D")
            self.assertEqual(rules["rsi_period"], 20)  # untouched default

    def test_capital_above_max_capital_is_rejected(self):
        with TemporaryDirectory() as d:
            path = write_rules(d, {"capital": 5000.0, "max_capital": 500.0})
            with self.assertRaises(ValueError) as ctx:
                etf.load_rules(path)
            self.assertIn("max_capital", str(ctx.exception))

    def test_zero_or_negative_capital_is_rejected(self):
        for bad in (0.0, -100.0):
            with TemporaryDirectory() as d:
                path = write_rules(d, {"capital": bad})
                with self.assertRaises(ValueError):
                    etf.load_rules(path)

    def test_currency_mismatch_is_a_hard_error(self):
        """The bot does not do FX; a EUR figure sizing a USD instrument
        would oversize every order by the exchange rate."""
        with TemporaryDirectory() as d:
            path = write_rules(d, {"capital_currency": "EUR", "trade_currency": "USD"})
            with self.assertRaises(ValueError) as ctx:
                etf.load_rules(path)
            self.assertIn("does not do FX", str(ctx.exception))

    def test_matching_non_usd_currencies_are_fine(self):
        with TemporaryDirectory() as d:
            path = write_rules(d, {"capital_currency": "EUR", "trade_currency": "EUR",
                                   "trade_symbol": "CL2", "trade_exchange": "SBF"})
            rules = etf.load_rules(path)
            self.assertEqual(rules["trade_currency"], "EUR")

    def test_signal_symbol_equal_to_trade_symbol_is_allowed(self):
        """This variant trades the same instrument its signal is computed
        on. Unlike the retired leveraged-ETP version -- where RSI(20) on a
        3x fund was a different series from RSI(20) on its index -- there
        is no mismatch to guard against once trade_leverage is forced
        to 1."""
        with TemporaryDirectory() as d:
            path = write_rules(d, {"signal_symbol": "QQQ", "trade_symbol": "QQQ"})
            rules = etf.load_rules(path)
            self.assertEqual(rules["signal_symbol"], rules["trade_symbol"])

    def test_equal_levels_are_rejected(self):
        with TemporaryDirectory() as d:
            path = write_rules(d, {"entry_level": 65, "exit_level": 65})
            with self.assertRaises(ValueError):
                etf.load_rules(path)

    def test_inverted_levels_are_rejected(self):
        with TemporaryDirectory() as d:
            path = write_rules(d, {"entry_level": 70, "exit_level": 60})
            with self.assertRaises(ValueError):
                etf.load_rules(path)

    def test_position_fraction_above_five_percent_is_rejected(self):
        """This small-account version caps allocation at 5% per trade --
        a hard business rule, not the old (0, 1] range."""
        with TemporaryDirectory() as d:
            path = write_rules(d, {"position_fraction": 0.06})
            with self.assertRaises(ValueError) as ctx:
                etf.load_rules(path)
            self.assertIn("0.05", str(ctx.exception))

    def test_out_of_range_fractions_are_rejected(self):
        for key, bad in (("position_fraction", 0.0), ("position_fraction", 1.0),
                         ("sizing_buffer", 0.0), ("sizing_buffer", 1.5)):
            with TemporaryDirectory() as d:
                path = write_rules(d, {key: bad})
                with self.assertRaises(ValueError):
                    etf.load_rules(path)

    def test_leverage_other_than_one_is_rejected(self):
        """This version is intentionally unleveraged: broker leverage is
        disabled outright, not merely bounded."""
        for bad in (0, 2, 3):
            with TemporaryDirectory() as d:
                path = write_rules(d, {"trade_leverage": bad})
                with self.assertRaises(ValueError) as ctx:
                    etf.load_rules(path)
                self.assertIn("exactly 1", str(ctx.exception))

    def test_non_positive_size_increment_is_rejected(self):
        for bad in (0.0, -0.1, -1.0):
            with TemporaryDirectory() as d:
                path = write_rules(d, {"size_increment": bad})
                with self.assertRaises(ValueError):
                    etf.load_rules(path)

    def test_a_lot_size_increment_is_allowed(self):
        """Whole numbers above 1 are still legitimate: some instruments
        trade only in lots."""
        with TemporaryDirectory() as d:
            path = write_rules(d, {"size_increment": 10})
            self.assertEqual(etf.load_rules(path)["size_increment"], 10)

    def test_default_instrument_is_spy_unleveraged(self):
        """The strategy trades what it signals off, at 1x, in small
        size -- the opposite trade-off from the retired 3x, whole-share
        3USL version."""
        rules = etf.load_rules(Path("does-not-exist.json"))
        self.assertEqual(rules["trade_symbol"], "SPY")
        self.assertEqual(rules["trade_exchange"], "SMART")
        self.assertEqual(rules["trade_leverage"], 1)
        self.assertEqual(rules["position_fraction"], 0.05)
        self.assertEqual(rules["size_increment"], 0.0001)

    def test_max_shares_below_one_is_rejected(self):
        with TemporaryDirectory() as d:
            path = write_rules(d, {"max_shares": 0})
            with self.assertRaises(ValueError):
                etf.load_rules(path)

    def test_non_positive_max_position_value_is_rejected(self):
        for bad in (0.0, -5.0):
            with TemporaryDirectory() as d:
                path = write_rules(d, {"max_position_value": bad})
                with self.assertRaises(ValueError):
                    etf.load_rules(path)

    def test_max_position_value_above_the_hard_ceiling_is_rejected(self):
        with TemporaryDirectory() as d:
            path = write_rules(d, {"max_position_value": 30.0})
            with self.assertRaises(ValueError) as ctx:
                etf.load_rules(path)
            self.assertIn("safety ceiling", str(ctx.exception))

    def test_target_position_exceeding_the_ceiling_is_rejected(self):
        """capital * position_fraction must itself fit under
        max_position_value -- a lowered ceiling can still be violated
        even at a valid (<=5%) position_fraction."""
        with TemporaryDirectory() as d:
            path = write_rules(d, {"capital": 500.0, "position_fraction": 0.05,
                                   "max_position_value": 10.0})
            with self.assertRaises(ValueError) as ctx:
                etf.load_rules(path)
            self.assertIn("exceeds hard position limit", str(ctx.exception))

    def test_shipped_rules_file_is_valid(self):
        """The committed config must load, or the first scheduled run dies."""
        shipped = Path("rsi20_dip_etf_rules.json")
        if not shipped.exists():
            self.skipTest("rules file not present in the working directory")
        rules = etf.load_rules(shipped)
        self.assertGreater(rules["capital"], 0)
        self.assertEqual(rules["signal_symbol"], rules["trade_symbol"])
        self.assertEqual(rules["trade_leverage"], 1)
        self.assertLessEqual(rules["capital"] * rules["position_fraction"],
                             rules["max_position_value"] + 1e-9)


class TestUsableCapital(unittest.TestCase):
    def test_equity_below_the_allocation_caps_it(self):
        rules = {**RULES, "capital": 500.0, "position_fraction": 1.0}
        self.assertAlmostEqual(etf.usable_capital(rules, 300.0, "USD"), 300.0)

    def test_allocation_below_equity_caps_it(self):
        """A 500 allocation inside a 50k account still buys 500 of SPY."""
        rules = {**RULES, "capital": 500.0, "position_fraction": 1.0}
        self.assertAlmostEqual(etf.usable_capital(rules, 50_000.0, "USD"), 500.0)

    def test_position_fraction_scales_both_sides(self):
        rules = {**RULES, "capital": 500.0, "position_fraction": 0.5}
        self.assertAlmostEqual(etf.usable_capital(rules, 50_000.0, "USD"), 250.0)
        self.assertAlmostEqual(etf.usable_capital(rules, 300.0, "USD"), 150.0)

    def test_default_position_fraction_allocates_five_percent(self):
        """The shipped default: $500 capital in a well-funded account
        still allocates only the configured $25 (5%)."""
        rules = {**RULES, "capital": 500.0}
        self.assertAlmostEqual(etf.usable_capital(rules, 50_000.0, "USD"), 25.0)

    def test_foreign_currency_equity_is_ignored_not_converted(self):
        """EUR equity must not cap a USD allocation at its face value."""
        rules = {**RULES, "capital": 500.0, "trade_currency": "USD",
                 "position_fraction": 1.0}
        self.assertAlmostEqual(etf.usable_capital(rules, 460.0, "EUR"), 500.0)

    def test_missing_equity_falls_back_to_the_allocation(self):
        rules = {**RULES, "capital": 500.0, "position_fraction": 1.0}
        self.assertAlmostEqual(etf.usable_capital(rules, None, ""), 500.0)
        self.assertAlmostEqual(etf.usable_capital(rules, 0.0, "USD"), 500.0)


class TestSharesForFractionalDefault(unittest.TestCase):
    """The shipped configuration: size_increment 0.0001, max_position_value
    25.0. `capital` here is the ALREADY-allocated figure usable_capital
    would hand it (typically $25 of a $500 account), not the account's
    total capital."""

    def test_sizes_within_the_buffer_and_reports_idle_cash(self):
        # budget = min(25*0.97, 25*0.97) = 24.25; at 100.00 that divides
        # evenly, so nothing is lost to flooring.
        sized = etf.shares_for(25.0, 100.0, RULES)
        self.assertAlmostEqual(sized["shares"], 0.2425)
        self.assertAlmostEqual(sized["notional"], 24.25)
        self.assertAlmostEqual(sized["idle_cash"], 0.75)
        self.assertAlmostEqual(sized["idle_pct"], 3.0, places=1)

    def test_floors_down_to_the_increment(self):
        # 24.25 / 96.0 / 0.0001 = 2526.04... -> floors to 2526 increments,
        # not 2526.04 or 2527.
        sized = etf.shares_for(25.0, 96.0, RULES)
        self.assertAlmostEqual(sized["shares"], 0.2526)
        self.assertAlmostEqual(sized["notional"], 24.2496, places=4)

    def test_zero_price_raises_rather_than_sizing_enormously(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                etf.shares_for(25.0, bad, RULES)

    def test_non_positive_capital_raises(self):
        """New in this version: shares_for now also guards capital, not
        just price."""
        for bad in (0.0, -25.0):
            with self.assertRaises(ValueError):
                etf.shares_for(bad, 100.0, RULES)

    def test_buffer_keeps_the_order_inside_the_cash(self):
        """Sized off yesterday's close, filled at today's open: a 2% gap
        up must still be affordable."""
        sized = etf.shares_for(25.0, 100.0, RULES)
        self.assertLessEqual(sized["shares"] * 100.0 * 1.02, 25.0)


class TestSharesForHardPositionCeiling(unittest.TestCase):
    """max_position_value caps the position independently of how much
    capital the caller hands in -- the same ceiling load_rules enforces
    at config time, re-checked here at the sizing layer."""

    def test_a_bigger_capital_argument_is_still_capped(self):
        # Passing the account's full $500 (not the $25 allocation) must
        # not blow through max_position_value.
        sized = etf.shares_for(500.0, 100.0, RULES)
        self.assertLessEqual(sized["notional"], 25.0 + 1e-8)
        self.assertAlmostEqual(sized["notional"], 24.25)  # 25 * 0.97

    def test_raising_max_position_value_raises_the_cap(self):
        rules = {**RULES, "max_position_value": 100.0}
        sized = etf.shares_for(500.0, 100.0, rules)
        self.assertAlmostEqual(sized["notional"], 97.0)  # min(500, 100) * 0.97


class TestSharesForGenericQuantization(unittest.TestCase):
    """Flooring, idle-cash reporting and the max_shares ceiling are
    generic mechanics that do not depend on this strategy's $25 cap.
    max_position_value is raised here so that cap does not interfere
    with what is actually under test -- these mirror the numbers the
    retired whole-share (3USL) version relied on, showing the underlying
    math is unchanged even though the shipped config now uses a
    fractional increment."""

    RAISED = {**RULES, "max_position_value": 10_000.0}

    def test_floors_to_whole_units_after_the_buffer(self):
        rules = {**self.RAISED, "size_increment": 1.0}
        # 500 * 0.97 = 485, at 100 -> 4 shares, not 5.
        sized = etf.shares_for(500.0, 100.0, rules)
        self.assertEqual(sized["shares"], 4)
        self.assertAlmostEqual(sized["notional"], 400.0)

    def test_reports_the_idle_cash_rounding_leaves(self):
        rules = {**self.RAISED, "size_increment": 1.0}
        sized = etf.shares_for(500.0, 100.0, rules)
        self.assertAlmostEqual(sized["idle_cash"], 100.0)
        self.assertAlmostEqual(sized["idle_pct"], 20.0)

    def test_a_cheap_price_leaves_almost_nothing_idle(self):
        rules = {**self.RAISED, "size_increment": 1.0}
        sized = etf.shares_for(500.0, 5.0, rules)
        self.assertEqual(sized["shares"], 97)
        self.assertLess(sized["idle_pct"], 4.0)

    def test_max_shares_is_a_hard_ceiling(self):
        rules = {**self.RAISED, "size_increment": 1.0, "max_shares": 10}
        sized = etf.shares_for(500.0, 5.0, rules)
        self.assertEqual(sized["shares"], 10)

    def test_capital_smaller_than_one_unit_yields_zero(self):
        rules = {**self.RAISED, "size_increment": 1.0}
        sized = etf.shares_for(50.0, 100.0, rules)
        self.assertEqual(sized["shares"], 0)
        self.assertAlmostEqual(sized["notional"], 0.0)

    def test_quantises_down_to_the_lot(self):
        rules = {**self.RAISED, "size_increment": 10.0}
        # 500 * 0.97 = 485 at 12.00 -> 40.4 shares -> 40, not 40.4 or 50
        sized = etf.shares_for(500.0, 12.0, rules)
        self.assertEqual(sized["shares"], 40)
        self.assertAlmostEqual(sized["notional"], 480.0)

    def test_capital_below_one_lot_yields_zero(self):
        rules = {**self.RAISED, "size_increment": 100.0}
        self.assertEqual(etf.shares_for(500.0, 12.0, rules)["shares"], 0)

    def test_never_exceeds_the_budget(self):
        for increment in (1.0, 10.0, 100.0):
            rules = {**self.RAISED, "size_increment": increment}
            for price in (188.15, 12.34, 1.07):
                sized = etf.shares_for(500.0, price, rules)
                self.assertLessEqual(sized["notional"],
                                     500.0 * RULES["sizing_buffer"])


class TestEffectiveLeverage(unittest.TestCase):
    """This variant trades at 1x only, so effective_leverage is simply
    notional / capital -- unlike the retired leveraged-ETP version, it is
    no longer scaled by a configured trade_leverage (that invariant is
    now enforced once, at config time, by load_rules)."""

    def test_returns_notional_over_capital(self):
        sized = {"notional": 24.25}
        self.assertAlmostEqual(etf.effective_leverage(sized, 25.0, RULES), 0.97)

    def test_full_investment_reaches_one(self):
        self.assertAlmostEqual(
            etf.effective_leverage({"notional": 500.0}, 500.0, RULES), 1.0)

    def test_configured_trade_leverage_no_longer_scales_the_result(self):
        """A caller that still puts a non-1 trade_leverage in the rules
        dict (bypassing load_rules) must not see it silently reapplied."""
        rules = {**RULES, "trade_leverage": 3}
        self.assertAlmostEqual(
            etf.effective_leverage({"notional": 100.0}, 500.0, rules), 0.2)

    def test_zero_capital_is_zero_leverage_not_a_division_error(self):
        self.assertEqual(etf.effective_leverage({"notional": 0.0}, 0.0, RULES), 0.0)


class TestStatePathsAreSeparate(unittest.TestCase):
    def test_does_not_share_files_with_the_futures_variant(self):
        """Shared state would have one bot's exit wipe the other's
        position record."""
        for etf_path, fut_path in (
            (etf.RULES_PATH, rsi20_dip_live.RULES_PATH),
            (etf.POSITIONS_PATH, rsi20_dip_live.POSITIONS_PATH),
            (etf.TRADES_CSV_PATH, rsi20_dip_live.TRADES_CSV_PATH),
            (etf.HEARTBEAT_PATH, rsi20_dip_live.HEARTBEAT_PATH),
        ):
            self.assertNotEqual(etf_path, fut_path)


if __name__ == "__main__":
    unittest.main()
