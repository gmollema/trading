"""Unit tests for trading_bot.backtest.portfolio.

Covers position_size (cycle.entry_scan's risk-based sizing math) and
manage_position's pre_breakeven -> post_breakeven exit state machine
(cycle.manage_position, adapted to a bar-close simulation instead of a
live poll). See tests/unit/cli/test_cycle.py for the live-bot equivalents
these are ported from.
"""

import unittest

from trading_bot.backtest import portfolio


def make_exit_cfg(partial_trigger_R=0.75, breakeven_trigger_R=1.0, partial_profit_fraction=1 / 3):
    return {
        "partial_profit_trigger_R": partial_trigger_R,
        "breakeven_trigger_R": breakeven_trigger_R,
        "partial_profit_fraction": partial_profit_fraction,
    }


def make_pos(**overrides):
    pos = {
        "symbol": "AAPL",
        "entry_price": 100.0,
        "qty": 30,
        "initial_stop": 95.0,
        "current_stop_price": 95.0,
        "state": "pre_breakeven",
        "R": 5.0,
    }
    pos.update(overrides)
    return pos


class TestInitialStopFromLod(unittest.TestCase):
    def test_one_percent_below_lod(self):
        self.assertAlmostEqual(portfolio.initial_stop_from_lod(100.0), 99.0)


class TestCommission(unittest.TestCase):
    def test_per_share_binds_above_minimum(self):
        # 1000 shares @ $0.005/share = $5.00, above the $1.00 minimum.
        self.assertAlmostEqual(portfolio.commission(1000, 0.005, 1.0), 5.0)

    def test_minimum_binds_for_small_fills(self):
        # 50 shares @ $0.005/share = $0.25, below the $1.00 minimum.
        self.assertAlmostEqual(portfolio.commission(50, 0.005, 1.0), 1.0)

    def test_defaults_match_ibkr_standard_schedule(self):
        self.assertAlmostEqual(portfolio.commission(1000), 5.0)
        self.assertAlmostEqual(portfolio.commission(1), 1.0)

    def test_tiered_constants_match_ibkr_tiered_base_schedule(self):
        # 1000 shares @ $0.0035 = $3.50 (rate binds); 50 shares @ $0.0035
        # = $0.175 -> the $0.35 minimum binds instead.
        self.assertAlmostEqual(
            portfolio.commission(1000, portfolio.TIERED_COMMISSION_PER_SHARE, portfolio.TIERED_COMMISSION_MIN), 3.5
        )
        self.assertAlmostEqual(
            portfolio.commission(50, portfolio.TIERED_COMMISSION_PER_SHARE, portfolio.TIERED_COMMISSION_MIN), 0.35
        )


class TestFractionalCommission(unittest.TestCase):
    def test_pct_of_notional_binds_above_minimum(self):
        # 10 shares @ $50 = $500 notional; 1% = $5.00, above the $0.01 min.
        self.assertAlmostEqual(portfolio.fractional_commission(10, 50.0, 0.01, 0.01), 5.0)

    def test_minimum_binds_for_tiny_fractional_fills(self):
        # 0.02 shares @ $10 = $0.20 notional; 1% = $0.002, below the $0.01 min.
        self.assertAlmostEqual(portfolio.fractional_commission(0.02, 10.0, 0.01, 0.01), 0.01)

    def test_defaults_match_ibkr_published_fractional_schedule(self):
        self.assertAlmostEqual(portfolio.fractional_commission(10, 50.0), 5.0)
        self.assertAlmostEqual(portfolio.fractional_commission(0.001, 10.0), 0.01)


class TestFxCommission(unittest.TestCase):
    def test_bps_rate_binds_above_minimum(self):
        # $100,000 notional @ 0.20 bps = $2.00 -- exactly at the minimum,
        # so bump notional up to make the rate clearly bind instead.
        self.assertAlmostEqual(portfolio.fx_commission(500_000, 0.20, 2.0), 10.0)

    def test_minimum_binds_for_small_notional(self):
        # $10,000 notional @ 0.20 bps = $0.20, below the $2.00 minimum.
        self.assertAlmostEqual(portfolio.fx_commission(10_000, 0.20, 2.0), 2.0)

    def test_defaults_match_ibkr_idealpro_tier_1_schedule(self):
        # $100,000 notional (a standard lot) @ 0.20 bps = $2.00, which is
        # also exactly the minimum -- IBKR's own worked example.
        self.assertAlmostEqual(portfolio.fx_commission(100_000), 2.0)
        self.assertAlmostEqual(portfolio.fx_commission(1_000), 2.0)


class TestFxPipSize(unittest.TestCase):
    def test_jpy_pairs_use_the_two_decimal_pip(self):
        self.assertAlmostEqual(portfolio.fx_pip_size("USDJPY"), 0.01)
        self.assertAlmostEqual(portfolio.fx_pip_size("usdjpy"), 0.01)

    def test_non_jpy_pairs_use_the_four_decimal_pip(self):
        self.assertAlmostEqual(portfolio.fx_pip_size("GBPUSD"), 0.0001)
        self.assertAlmostEqual(portfolio.fx_pip_size("EURUSD"), 0.0001)


class TestFxHalfSpreadPrice(unittest.TestCase):
    def test_non_jpy_pair_converts_pips_at_four_decimals(self):
        # 2 pips on a non-JPY pair = 0.0002; half of that = 0.0001.
        self.assertAlmostEqual(portfolio.fx_half_spread_price("GBPUSD", 2.0), 0.0001)

    def test_jpy_pair_converts_pips_at_two_decimals(self):
        # 2 pips on USDJPY = 0.02; half of that = 0.01.
        self.assertAlmostEqual(portfolio.fx_half_spread_price("USDJPY", 2.0), 0.01)


class TestFxFillPrice(unittest.TestCase):
    def test_buy_fills_above_mid(self):
        self.assertAlmostEqual(portfolio.fx_fill_price(1.3000, "BUY", 0.0001), 1.3001)

    def test_sell_fills_below_mid(self):
        self.assertAlmostEqual(portfolio.fx_fill_price(1.3000, "SELL", 0.0001), 1.2999)


class TestPositionSize(unittest.TestCase):
    def test_risk_based_size_when_it_is_the_binding_constraint(self):
        # risk_dollars = 100_000 * 1% = 1000; R = 100-95 = 5 -> size_by_risk = 200
        # cap: 10% of 100_000 / 100 = 100 -> size_by_cap = 100 (binding)
        size = portfolio.position_size(100_000, 1.0, 100.0, 95.0, 10.0)
        self.assertEqual(size, 100)

    def test_position_cap_binds_before_risk_size(self):
        # risk_dollars = 100_000 * 1% = 1000; R = 100-99 = 1 -> size_by_risk = 1000
        # cap: 10% of 100_000 / 100 = 100 -> binding
        size = portfolio.position_size(100_000, 1.0, 100.0, 99.0, 10.0)
        self.assertEqual(size, 100)

    def test_non_positive_risk_returns_zero(self):
        self.assertEqual(portfolio.position_size(100_000, 1.0, 100.0, 100.0, 10.0), 0)
        self.assertEqual(portfolio.position_size(100_000, 1.0, 100.0, 105.0, 10.0), 0)

    def test_floors_to_whole_shares(self):
        # risk_dollars = 1000, R = 3 -> 333.33 -> floor 333; cap huge, not binding
        size = portfolio.position_size(100_000, 1.0, 10.0, 7.0, 100.0)
        self.assertEqual(size, 333)

    def test_allow_fractional_returns_raw_size(self):
        # Same math as test_floors_to_whole_shares, but unrounded: 1000/3 = 333.33...
        size = portfolio.position_size(100_000, 1.0, 10.0, 7.0, 100.0, allow_fractional=True)
        self.assertAlmostEqual(size, 1000 / 3)

    def test_allow_fractional_lets_a_tiny_account_size_a_pricey_stock(self):
        # $1,000 account, 10% cap = $100 max position -- a $500 stock would
        # floor to 0 whole shares, but 0.2 fractional shares is tradeable.
        size = portfolio.position_size(1_000, 1.0, 500.0, 490.0, 10.0, allow_fractional=True)
        self.assertGreater(size, 0)
        self.assertEqual(portfolio.position_size(1_000, 1.0, 500.0, 490.0, 10.0), 0)


class TestOpenPosition(unittest.TestCase):
    def test_shape_matches_cycle_new_pos(self):
        pos = portfolio.open_position("AAPL", 100.0, 96.0, 50)

        self.assertEqual(pos["symbol"], "AAPL")
        self.assertEqual(pos["entry_price"], 100.0)
        self.assertEqual(pos["qty"], 50)
        self.assertAlmostEqual(pos["initial_stop"], 95.04)
        self.assertAlmostEqual(pos["current_stop_price"], 95.04)
        self.assertEqual(pos["state"], "pre_breakeven")
        self.assertAlmostEqual(pos["R"], 100.0 - 95.04)


class TestManagePositionStopOut(unittest.TestCase):
    def test_low_at_or_below_stop_closes_full_position(self):
        pos = make_pos()

        updated, fills = portfolio.manage_position(pos, {"low": 94.5, "close": 96.0}, [96.0], make_exit_cfg())

        self.assertIsNone(updated)
        self.assertEqual(fills, [{"qty": 30, "price": 95.0, "reason": "stop"}])

    def test_low_exactly_at_stop_triggers(self):
        pos = make_pos(current_stop_price=95.0)

        updated, fills = portfolio.manage_position(pos, {"low": 95.0, "close": 96.0}, [96.0], make_exit_cfg())

        self.assertIsNone(updated)
        self.assertEqual(fills[0]["reason"], "stop")

    def test_low_above_stop_does_not_trigger(self):
        pos = make_pos()

        updated, fills = portfolio.manage_position(pos, {"low": 95.5, "close": 96.0}, [96.0], make_exit_cfg())

        self.assertIsNotNone(updated)
        self.assertEqual(fills, [])


class TestManagePositionPartialProfit(unittest.TestCase):
    def test_partial_taken_on_reaching_trigger(self):
        pos = make_pos()  # entry=100, R=5 -> partial trigger @ 103.75

        updated, fills = portfolio.manage_position(
            pos, {"low": 99.0, "close": 103.75}, [99.0, 103.75], make_exit_cfg()
        )

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["reason"], "partial_profit")
        self.assertEqual(fills[0]["qty"], 10)  # ceil(30/3)
        self.assertEqual(updated["qty"], 20)
        self.assertEqual(updated["state"], "post_breakeven_partial_done")
        self.assertAlmostEqual(updated["current_stop_price"], 99.0)  # entry * 0.99, breakeven not yet reached

    def test_fast_move_past_both_thresholds_gets_breakeven_stop(self):
        pos = make_pos()  # breakeven @ 105.0

        updated, fills = portfolio.manage_position(
            pos, {"low": 99.0, "close": 105.5}, [99.0, 105.5], make_exit_cfg()
        )

        self.assertEqual(updated["state"], "post_breakeven_partial_done")
        self.assertEqual(updated["current_stop_price"], 100.0)  # full entry, not the discounted 0.99x

    def test_below_partial_trigger_state_unchanged(self):
        pos = make_pos()

        updated, fills = portfolio.manage_position(
            pos, {"low": 99.0, "close": 102.0}, [99.0, 102.0], make_exit_cfg()
        )

        self.assertEqual(fills, [])
        self.assertEqual(updated["state"], "pre_breakeven")
        self.assertEqual(updated["qty"], 30)

    def test_partial_fully_closes_when_fraction_rounds_up_to_full_qty(self):
        pos = make_pos(qty=1)

        updated, fills = portfolio.manage_position(
            pos, {"low": 99.0, "close": 103.75}, [99.0, 103.75], make_exit_cfg()
        )

        self.assertIsNone(updated)
        self.assertEqual(fills[0]["qty"], 1)


class TestManagePositionBreakevenOnlyFallback(unittest.TestCase):
    """Defensive elif branch: only reachable if breakeven_trigger_R < partial_profit_trigger_R."""

    def test_breakeven_only_when_partial_trigger_is_higher(self):
        pos = make_pos()
        cfg = make_exit_cfg(partial_trigger_R=1.5, breakeven_trigger_R=1.0)

        updated, fills = portfolio.manage_position(pos, {"low": 99.0, "close": 105.0}, [99.0, 105.0], cfg)

        self.assertEqual(fills, [])
        self.assertEqual(updated["state"], "post_breakeven_no_partial")
        self.assertEqual(updated["current_stop_price"], 100.0)
        self.assertEqual(updated["qty"], 30)


class TestManagePositionTrailingStop(unittest.TestCase):
    def _lows_with_swing_low(self, low_value):
        # low_value must be lower than both neighboring pairs to register
        # as a swing low (matches cycle.compute_swing_lows' strict '<').
        return [110, 105, low_value, 105, 110]

    def test_ratchet_applied_when_candidate_above_current_stop(self):
        pos = make_pos(state="post_breakeven_partial_done", current_stop_price=99.0, qty=20)
        lows = self._lows_with_swing_low(100.5)  # candidate stop = 100.49 > 99.0

        updated, fills = portfolio.manage_position(pos, {"low": 105.0, "close": 110.0}, lows, make_exit_cfg())

        self.assertEqual(fills, [])
        self.assertAlmostEqual(updated["current_stop_price"], 100.49)

    def test_stop_never_ratchets_down(self):
        pos = make_pos(state="post_breakeven_partial_done", current_stop_price=101.0, qty=20)
        lows = self._lows_with_swing_low(100.5)  # candidate 100.49 < current 101.0

        updated, fills = portfolio.manage_position(pos, {"low": 105.0, "close": 110.0}, lows, make_exit_cfg())

        self.assertEqual(fills, [])
        self.assertEqual(updated["current_stop_price"], 101.0)  # unchanged

    def test_no_swing_low_yet_leaves_stop_unchanged(self):
        pos = make_pos(state="post_breakeven_no_partial", current_stop_price=100.0, qty=30)
        lows = [110, 109, 108, 107, 106]  # monotonic -- no swing low

        updated, fills = portfolio.manage_position(pos, {"low": 106.0, "close": 110.0}, lows, make_exit_cfg())

        self.assertEqual(fills, [])
        self.assertEqual(updated["current_stop_price"], 100.0)


class TestClosePosition(unittest.TestCase):
    def test_default_reason_is_force_close(self):
        fill = portfolio.close_position(make_pos(qty=20), 105.0)

        self.assertEqual(fill, {"qty": 20, "price": 105.0, "reason": "force_close"})

    def test_custom_reason(self):
        fill = portfolio.close_position(make_pos(qty=20), 105.0, reason="end_of_data")

        self.assertEqual(fill["reason"], "end_of_data")


if __name__ == "__main__":
    unittest.main()
