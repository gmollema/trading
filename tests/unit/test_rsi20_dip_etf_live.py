"""Unit tests for trading_bot.rsi20_dip_etf_live.

The signal itself is already covered by test_rsi20_dip_signals and
test_rsi20_dip_live -- this module re-exports `decide` rather than
reimplementing it, so what is tested here is what the ETF variant
actually adds: rules validation, the equity cap, and share sizing.

One test asserts that `decide` really is the same function object as the
futures variant's, because the whole safety argument for this variant is
that it runs the code the backtest validated.
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

    def test_signal_symbol_equal_to_trade_symbol_is_rejected(self):
        """RSI(20) on a 3x fund is a different series from RSI(20) on its
        index, so the fitted 60/65 levels would silently change meaning."""
        with TemporaryDirectory() as d:
            path = write_rules(d, {"signal_symbol": "3USL", "trade_symbol": "3USL"})
            with self.assertRaises(ValueError) as ctx:
                etf.load_rules(path)
            self.assertIn("must differ", str(ctx.exception))

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

    def test_out_of_range_fractions_are_rejected(self):
        for key, bad in (("position_fraction", 0.0), ("position_fraction", 1.5),
                         ("sizing_buffer", 0.0), ("sizing_buffer", 1.5)):
            with TemporaryDirectory() as d:
                path = write_rules(d, {key: bad})
                with self.assertRaises(ValueError):
                    etf.load_rules(path)

    def test_a_fractional_size_increment_is_rejected(self):
        """IBKR error 10243 refuses fractional quantities over the API,
        whatever sizeIncrement the contract advertises. Configuring one
        would produce rejected orders, not small positions."""
        for bad in (0.0001, 0.5, 0.0, -0.1, 1.5):
            with TemporaryDirectory() as d:
                path = write_rules(d, {"size_increment": bad})
                with self.assertRaises(ValueError) as ctx:
                    etf.load_rules(path)
                self.assertIn("10243", str(ctx.exception))

    def test_a_lot_size_increment_is_allowed(self):
        """Whole numbers above 1 are legitimate: some instruments trade
        only in lots."""
        with TemporaryDirectory() as d:
            path = write_rules(d, {"size_increment": 10})
            self.assertEqual(etf.load_rules(path)["size_increment"], 10)

    def test_default_instrument_is_the_cheaper_whole_share_one(self):
        """Whole shares only, so the share price relative to the capital
        decides: 3USL at ~$188 deploys 75% of 500 for 2.26x, XS2D at
        ~$362 deploys 72% for only 1.45x."""
        rules = etf.load_rules(Path("does-not-exist.json"))
        self.assertEqual(rules["trade_symbol"], "3USL")
        self.assertEqual(rules["trade_exchange"], "LSEETF")
        self.assertEqual(rules["trade_leverage"], 3)
        self.assertEqual(rules["size_increment"], 1.0)

    def test_max_shares_below_one_is_rejected(self):
        with TemporaryDirectory() as d:
            path = write_rules(d, {"max_shares": 0})
            with self.assertRaises(ValueError):
                etf.load_rules(path)

    def test_shipped_rules_file_is_valid(self):
        """The committed config must load, or the first scheduled run dies."""
        shipped = Path("rsi20_dip_etf_rules.json")
        if not shipped.exists():
            self.skipTest("rules file not present in the working directory")
        rules = etf.load_rules(shipped)
        self.assertGreater(rules["capital"], 0)
        self.assertNotEqual(rules["signal_symbol"], rules["trade_symbol"])


class TestUsableCapital(unittest.TestCase):
    def test_equity_below_the_allocation_caps_it(self):
        self.assertAlmostEqual(
            etf.usable_capital({**RULES, "capital": 500.0}, 300.0, "USD"), 300.0)

    def test_allocation_below_equity_caps_it(self):
        """A 500 allocation inside a 50k account still buys 500 of ETP."""
        self.assertAlmostEqual(
            etf.usable_capital({**RULES, "capital": 500.0}, 50_000.0, "USD"), 500.0)

    def test_position_fraction_scales_both_sides(self):
        rules = {**RULES, "capital": 500.0, "position_fraction": 0.5}
        self.assertAlmostEqual(etf.usable_capital(rules, 50_000.0, "USD"), 250.0)
        self.assertAlmostEqual(etf.usable_capital(rules, 300.0, "USD"), 150.0)

    def test_foreign_currency_equity_is_ignored_not_converted(self):
        """EUR equity must not cap a USD allocation at its face value."""
        rules = {**RULES, "capital": 500.0, "trade_currency": "USD"}
        self.assertAlmostEqual(etf.usable_capital(rules, 460.0, "EUR"), 500.0)

    def test_missing_equity_falls_back_to_the_allocation(self):
        rules = {**RULES, "capital": 500.0}
        self.assertAlmostEqual(etf.usable_capital(rules, None, ""), 500.0)
        self.assertAlmostEqual(etf.usable_capital(rules, 0.0, "USD"), 500.0)


WHOLE = {**RULES, "size_increment": 1.0}


class TestSharesForWholeShares(unittest.TestCase):
    """size_increment 1.0 -- an ETN like 3USL, which IBKR reports as
    whole-share only."""

    def test_floors_to_whole_shares_after_the_buffer(self):
        # 500 * 0.97 = 485, at 100 -> 4 shares, not 5.
        sized = etf.shares_for(500.0, 100.0, WHOLE)
        self.assertEqual(sized["shares"], 4)
        self.assertAlmostEqual(sized["notional"], 400.0)

    def test_reports_the_idle_cash_rounding_leaves(self):
        """With a coarse increment this is the dominant sizing error, so
        it is reported rather than swallowed."""
        sized = etf.shares_for(500.0, 100.0, WHOLE)
        self.assertAlmostEqual(sized["idle_cash"], 100.0)
        self.assertAlmostEqual(sized["idle_pct"], 20.0)

    def test_the_measured_3usl_case(self):
        """The shipped default: 500 against a $188 whole-share ETN buys 2
        shares, leaves ~25% idle, and takes on 2.26x of a configured 3x.
        Still the best available, since XS2D at $362 manages only 1.45x."""
        rules = {**WHOLE, "trade_leverage": 3}
        sized = etf.shares_for(500.0, 188.15, rules)
        self.assertEqual(sized["shares"], 2)
        self.assertAlmostEqual(sized["idle_pct"], 24.7, places=1)
        self.assertAlmostEqual(etf.effective_leverage(sized, 500.0, rules),
                               2.26, places=2)

    def test_a_cheap_share_leaves_almost_nothing_idle(self):
        sized = etf.shares_for(500.0, 5.0, WHOLE)
        self.assertEqual(sized["shares"], 97)
        self.assertLess(sized["idle_pct"], 4.0)

    def test_max_shares_is_a_hard_ceiling(self):
        sized = etf.shares_for(500.0, 5.0, {**WHOLE, "max_shares": 10})
        self.assertEqual(sized["shares"], 10)

    def test_capital_smaller_than_one_share_yields_zero(self):
        sized = etf.shares_for(50.0, 100.0, WHOLE)
        self.assertEqual(sized["shares"], 0)
        self.assertAlmostEqual(sized["notional"], 0.0)

    def test_zero_price_raises_rather_than_sizing_enormously(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                etf.shares_for(500.0, bad, WHOLE)

    def test_buffer_keeps_the_order_inside_the_cash(self):
        """Sized off yesterday's close, filled at today's open: a 2% gap
        up must still be affordable."""
        sized = etf.shares_for(500.0, 100.0, WHOLE)
        self.assertLessEqual(sized["shares"] * 100.0 * 1.02, 500.0)


class TestSharesForLotSizes(unittest.TestCase):
    """size_increment above 1, for an instrument that trades in lots."""

    def test_quantises_down_to_the_lot(self):
        rules = {**RULES, "size_increment": 10.0}
        # 500 * 0.97 = 485 at 12.00 -> 40.4 shares -> 40, not 40.4 or 50
        sized = etf.shares_for(500.0, 12.0, rules)
        self.assertEqual(sized["shares"], 40)
        self.assertAlmostEqual(sized["notional"], 480.0)

    def test_capital_below_one_lot_yields_zero(self):
        rules = {**RULES, "size_increment": 100.0}
        self.assertEqual(etf.shares_for(500.0, 12.0, rules)["shares"], 0)

    def test_never_exceeds_the_budget(self):
        for increment in (1.0, 10.0, 100.0):
            rules = {**RULES, "size_increment": increment}
            for price in (188.15, 12.34, 1.07):
                sized = etf.shares_for(500.0, price, rules)
                self.assertLessEqual(sized["notional"],
                                     500.0 * RULES["sizing_buffer"])

    def test_every_quantity_is_a_whole_number(self):
        """A non-integer quantity is a rejected order (IBKR 10243), so no
        price may produce one."""
        for price in (188.15, 362.51, 3.7, 77.77, 1.03):
            shares = etf.shares_for(500.0, price, RULES)["shares"]
            self.assertEqual(shares, int(shares))


class TestEffectiveLeverage(unittest.TestCase):
    def test_idle_cash_dilutes_the_configured_leverage(self):
        # 4 whole shares at 100 out of 500 = 80% invested in a 3x fund.
        rules = {**RULES, "size_increment": 1.0, "trade_leverage": 3}
        sized = etf.shares_for(500.0, 100.0, rules)
        self.assertAlmostEqual(etf.effective_leverage(sized, 500.0, rules), 2.4)

    def test_full_investment_reaches_the_configured_leverage(self):
        self.assertAlmostEqual(
            etf.effective_leverage({"notional": 500.0}, 500.0, RULES), 3.0)

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
