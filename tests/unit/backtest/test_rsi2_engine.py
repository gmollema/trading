"""Unit tests for trading_bot.backtest.rsi2_engine.

The arithmetic here is deliberately checkable by hand: MES is $5/point,
so a 200-point stop is exactly $1,000 of risk per contract and a 1%-risk
account needs exactly $100,000 per contract. Several tests sit on that
boundary on purpose -- it is where the lockout pathology lives.
"""

import unittest

from trading_bot.backtest import rsi2_engine as engine


def point_trade(entry_idx, exit_idx, entry_price, points, stop_price=None, date=None):
    """A minimal stand-in for a rsi2_signals trade dict; the engine reads
    only these keys."""
    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_date": date if date is not None else entry_idx,
        "exit_date": exit_idx,
        "entry_price": entry_price,
        "stop_price": entry_price - 200.0 if stop_price is None else stop_price,
        "points": points,
    }


def flat_bars(n, price=5000.0):
    return {"date": list(range(n)), "close": [price] * n}


# Costs for one MES contract, one round trip: 2 * $0.62 commission plus
# 2 * 1 tick * 0.25pt * $5 = $2.50 slippage.
COST_PER_CONTRACT = 2 * 0.62 + 2 * 1 * 0.25 * 5.0


class TestContractsForTrade(unittest.TestCase):
    def test_risk_based_size(self):
        # $500k at 1% = $5,000 of risk; a 200-point stop is $1,000 per
        # contract -> 5 contracts.
        self.assertEqual(engine.contracts_for_trade(500_000, 5000.0, 4800.0, engine.MES, 1.0, 100.0), 5)

    def test_floors_rather_than_rounding(self):
        # $549k at 1% = $5,490 -> 5.49 contracts, and there is no such
        # thing as half an MES.
        self.assertEqual(engine.contracts_for_trade(549_000, 5000.0, 4800.0, engine.MES, 1.0, 100.0), 5)

    def test_margin_cap_can_bind_before_risk(self):
        # Risk allows 10 contracts, but 25% of $1M is $250k of margin at
        # $2,400 each = 104 -> risk binds. Drop the cap to 1% ($10k) and
        # margin binds at 4.
        self.assertEqual(engine.contracts_for_trade(1_000_000, 5000.0, 4800.0, engine.MES, 1.0, 25.0), 10)
        self.assertEqual(engine.contracts_for_trade(1_000_000, 5000.0, 4800.0, engine.MES, 1.0, 1.0), 4)

    def test_returns_zero_when_one_contract_is_unaffordable(self):
        # $99,999 at 1% is $999.99 against $1,000 of risk per contract.
        self.assertEqual(engine.contracts_for_trade(99_999, 5000.0, 4800.0, engine.MES, 1.0, 100.0), 0)
        self.assertEqual(engine.contracts_for_trade(100_000, 5000.0, 4800.0, engine.MES, 1.0, 100.0), 1)

    def test_es_multiplier_makes_the_same_stop_ten_times_dearer(self):
        # The reason this module does not reuse portfolio.position_size:
        # risk per contract scales with the multiplier.
        self.assertEqual(engine.contracts_for_trade(1_000_000, 5000.0, 4800.0, engine.ES, 1.0, 100.0), 1)
        self.assertEqual(engine.contracts_for_trade(1_000_000, 5000.0, 4800.0, engine.MES, 1.0, 100.0), 10)

    def test_no_stop_falls_back_to_the_margin_cap(self):
        # No stop means no definable risk per contract, so only margin
        # limits the size: 25% of $100k = $25k / $2,400 = 10.
        self.assertEqual(engine.contracts_for_trade(100_000, 5000.0, None, engine.MES, 1.0, 25.0), 10)

    def test_non_positive_risk_distance_is_rejected(self):
        self.assertEqual(engine.contracts_for_trade(500_000, 5000.0, 5000.0, engine.MES, 1.0, 100.0), 0)
        self.assertEqual(engine.contracts_for_trade(500_000, 5000.0, 5200.0, engine.MES, 1.0, 100.0), 0)


class TestRoundCosts(unittest.TestCase):
    def test_both_legs_pay(self):
        commission, slippage = engine.round_costs(0.0, 3, engine.MES, 1.0)
        self.assertAlmostEqual(commission, 2 * 3 * 0.62)
        self.assertAlmostEqual(slippage, 2 * 3 * 1 * 0.25 * 5.0)

    def test_slippage_scales_with_ticks_and_multiplier(self):
        _, mes = engine.round_costs(0.0, 1, engine.MES, 2.0)
        _, es = engine.round_costs(0.0, 1, engine.ES, 2.0)
        self.assertAlmostEqual(mes, 2 * 2 * 0.25 * 5.0)
        self.assertAlmostEqual(es, mes * 10)


class TestRunRsi2FuturesBacktest(unittest.TestCase):
    def test_single_trade_dollar_arithmetic(self):
        trades = [point_trade(0, 2, 5000.0, 40.0)]
        result = engine.run_rsi2_futures_backtest(trades, flat_bars(3), 100_000, engine.MES, risk_pct=1.0)
        row = result["trades"][0]
        self.assertEqual(row["contracts"], 1)
        self.assertAlmostEqual(row["gross_pnl"], 40 * 5.0)
        self.assertAlmostEqual(row["net_pnl"], 200.0 - COST_PER_CONTRACT, places=2)
        self.assertAlmostEqual(result["final_equity"], 100_000 + 200.0 - COST_PER_CONTRACT, places=2)
        self.assertIsNone(result["locked_out_from"])

    def test_sizing_compounds_off_equity_at_each_entry(self):
        # $299k at 1% risk funds 2 contracts ($2,990 / $1,000 each).
        # Winning 200 points on both adds $2,000, crossing $300k, which
        # is where the third contract becomes affordable.
        trades = [point_trade(0, 1, 5000.0, 200.0), point_trade(2, 3, 5000.0, 10.0)]
        result = engine.run_rsi2_futures_backtest(trades, flat_bars(4), 299_000, engine.MES, risk_pct=1.0)
        self.assertEqual([t["contracts"] for t in result["trades"]], [2, 3])

    def test_unaffordable_trade_is_skipped_not_downsized(self):
        trades = [point_trade(0, 1, 5000.0, 40.0)]
        result = engine.run_rsi2_futures_backtest(trades, flat_bars(2), 50_000, engine.MES, risk_pct=1.0)
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["skipped_trades"], 1)
        self.assertEqual(result["final_equity"], 50_000)

    def test_a_single_loss_at_the_boundary_locks_the_account_out(self):
        """The absorbing state. At exactly $100k and 1% risk the account
        can afford exactly one contract; one losing trade drops it under
        the line and no later winner can ever be taken, however large."""
        trades = [
            point_trade(0, 1, 5000.0, -200.0),   # -$1,000, equity now < $100k
            point_trade(2, 3, 5000.0, 500.0),    # would have been +$2,500
            point_trade(4, 5, 5000.0, 500.0),
        ]
        result = engine.run_rsi2_futures_backtest(trades, flat_bars(6), 100_000, engine.MES, risk_pct=1.0)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["skipped_trades"], 2)
        self.assertEqual(result["locked_out_from"], 2)
        self.assertLess(result["final_equity"], 100_000)

    def test_a_skip_run_that_recovers_is_not_a_lockout(self):
        # Sized off equity, the middle trade is unaffordable only because
        # its stop is far away; the next one is affordable again, so the
        # account was never locked out.
        trades = [
            point_trade(0, 1, 5000.0, 100.0),
            point_trade(2, 3, 5000.0, 10.0, stop_price=1000.0),  # $20k risk/contract
            point_trade(4, 5, 5000.0, 10.0),
        ]
        result = engine.run_rsi2_futures_backtest(trades, flat_bars(6), 100_000, engine.MES, risk_pct=1.0)
        self.assertEqual(result["skipped_trades"], 1)
        self.assertIsNone(result["locked_out_from"])

    def test_equity_curve_marks_the_open_position_to_market(self):
        # Held over bars 0-1 with the index 100 points below the entry:
        # one contract is $500 underwater before the exit is realized.
        bars = {"date": [0, 1, 2], "close": [4900.0, 4900.0, 5050.0]}
        trades = [point_trade(0, 2, 5000.0, 50.0)]
        result = engine.run_rsi2_futures_backtest(trades, bars, 100_000, engine.MES, risk_pct=1.0)
        marked = [p["equity"] for p in result["equity_curve"]]
        self.assertAlmostEqual(marked[0], 100_000 - 500.0)
        self.assertAlmostEqual(marked[1], 100_000 - 500.0)
        # The exit bar is not marked, only realized.
        self.assertAlmostEqual(marked[-1], 100_000 + 250.0 - COST_PER_CONTRACT, places=2)

    def test_no_trades_leaves_capital_untouched(self):
        result = engine.run_rsi2_futures_backtest([], flat_bars(3), 100_000, engine.MES)
        self.assertEqual(result["final_equity"], 100_000)
        self.assertEqual(result["equity_curve"], [])
        self.assertIsNone(result["locked_out_from"])


class TestMaxDrawdownPct(unittest.TestCase):
    def test_peak_to_trough(self):
        curve = [{"equity": 100.0}, {"equity": 120.0}, {"equity": 90.0}, {"equity": 130.0}]
        self.assertAlmostEqual(engine.max_drawdown_pct(curve), 25.0)

    def test_monotone_curve_has_no_drawdown(self):
        self.assertEqual(engine.max_drawdown_pct([{"equity": 1.0}, {"equity": 2.0}]), 0.0)

    def test_empty_curve(self):
        self.assertEqual(engine.max_drawdown_pct([]), 0.0)


class TestCagrPct(unittest.TestCase):
    def test_doubling_over_two_years(self):
        self.assertAlmostEqual(engine.cagr_pct(100.0, 200.0, 2.0), 41.4214, places=3)

    def test_flat_is_zero(self):
        self.assertAlmostEqual(engine.cagr_pct(100.0, 100.0, 5.0), 0.0)

    def test_degenerate_inputs_return_zero_rather_than_raising(self):
        self.assertEqual(engine.cagr_pct(100.0, 0.0, 5.0), 0.0)
        self.assertEqual(engine.cagr_pct(100.0, -50.0, 5.0), 0.0)
        self.assertEqual(engine.cagr_pct(100.0, 200.0, 0.0), 0.0)
        self.assertEqual(engine.cagr_pct(0.0, 200.0, 5.0), 0.0)


if __name__ == "__main__":
    unittest.main()
