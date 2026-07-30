"""Unit tests for trading_bot.backtest.ut_bot_engine.run_ut_bot_backtest.

Reuses the hand-traced 8-bar fixture from test_ut_bot_signals.py's
TestFindUtBotLongTrades (see that file for the full bar-by-bar
derivation) so the expected entry/exit prices are already verified;
these tests focus on the portfolio-level layer on top: position sizing
and equity updates.
"""

import unittest

import pandas as pd

from trading_bot.backtest import portfolio
from trading_bot.backtest.ut_bot_engine import run_ut_bot_backtest, run_ut_bot_confirmed_backtest

# Same OHLC sequence as test_ut_bot_signals.py: entry @ 11 (idx 4, stop 6),
# exit @ 5.5 (idx 7, sell_signal). key_value=1.0/atr_period=1 for the same
# arithmetic-simplicity reason as that file.
H = [10, 11, 9, 8, 12, 11.5, 10, 8]
L = [8, 9, 6, 6, 7, 9, 7, 5]
C = [9, 10, 7, 7.5, 11, 9.5, 7.2, 5.5]

# Same confirmed-long fixture as test_ut_bot_signals.py's
# test_confirmed_long_round_trip: entry @ 9.10 (idx 6), exit @ 8.75 (idx 7,
# sell_signal), key_value=0.3/atr_period=1.
CONFIRMED_H = [10.0, 9.6, 9.2, 9.0, 8.9, 9.05, 9.15, 8.9]
CONFIRMED_L = [9.0, 8.8, 8.4, 8.2, 8.1, 8.90, 8.95, 8.6]
CONFIRMED_C = [9.5, 9.0, 8.6, 8.3, 8.15, 9.00, 9.10, 8.75]


def _bars():
    dates = list(pd.date_range("2024-01-02", periods=len(C), freq="h", tz="UTC"))
    return {"high": H, "low": L, "close": C, "date": dates}


def _confirmed_bars():
    dates = list(pd.date_range("2024-01-02", periods=len(CONFIRMED_C), freq="h", tz="UTC"))
    return {"high": CONFIRMED_H, "low": CONFIRMED_L, "close": CONFIRMED_C, "date": dates}


class TestRunUtBotBacktest(unittest.TestCase):
    def test_single_trade_sized_and_closed_correctly(self):
        result = run_ut_bot_backtest(_bars(), 100_000, symbol="GBPUSD", risk_pct=1.0, max_position_pct=100.0,
                                      key_value=1.0, atr_period=1)
        trades = result["trades"]

        self.assertEqual(len(trades), 2)
        buy, sell = trades
        self.assertEqual(buy["side"], "BUY")
        self.assertEqual(buy["symbol"], "GBPUSD")
        self.assertEqual(buy["fill_price"], 11)

        expected_size = portfolio.position_size(100_000, 1.0, 11, 6.0, 100.0)
        self.assertEqual(buy["size"], expected_size)

        self.assertEqual(sell["side"], "SELL")
        self.assertEqual(sell["fill_price"], 5.5)
        self.assertEqual(sell["size"], expected_size)
        self.assertEqual(sell["reason"], "sell_signal")

        expected_final_equity = 100_000 + (5.5 - 11) * expected_size
        self.assertAlmostEqual(result["equity_curve"][-1]["equity"], expected_final_equity, places=6)

    def test_zero_size_trade_is_skipped(self):
        # Tiny capital + a small risk_pct -> position_size floors to 0
        # whole units, so no trade should appear at all.
        result = run_ut_bot_backtest(_bars(), 1, symbol="GBPUSD", risk_pct=1.0, max_position_pct=100.0,
                                      key_value=1.0, atr_period=1)
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["equity_curve"], [])

    def test_fractional_sizing_allows_a_trade_whole_units_would_skip(self):
        whole = run_ut_bot_backtest(_bars(), 1, key_value=1.0, atr_period=1)
        self.assertEqual(whole["trades"], [])

        fractional = run_ut_bot_backtest(_bars(), 1, key_value=1.0, atr_period=1, allow_fractional_shares=True)
        self.assertEqual(len(fractional["trades"]), 2)
        self.assertGreater(fractional["trades"][0]["size"], 0)

    def test_empty_bars_returns_empty_result(self):
        result = run_ut_bot_backtest({"high": [], "low": [], "close": [], "date": []}, 100_000)
        self.assertEqual(result, {"trades": [], "equity_curve": []})

    def test_no_commission_modeled_by_default(self):
        # fx_commission_bps defaults to None -- identical to
        # test_single_trade_sized_and_closed_correctly's result, just
        # asserted explicitly so a future default change can't silently
        # start charging commission unnoticed.
        result = run_ut_bot_backtest(_bars(), 100_000, key_value=1.0, atr_period=1)
        expected_size = portfolio.position_size(100_000, 1.0, 11, 6.0, 100.0)
        expected_final_equity = 100_000 + (5.5 - 11) * expected_size
        self.assertAlmostEqual(result["equity_curve"][-1]["equity"], expected_final_equity, places=6)

    def test_commission_charged_once_per_fill_when_enabled(self):
        no_commission = run_ut_bot_backtest(_bars(), 100_000, key_value=1.0, atr_period=1)
        with_commission = run_ut_bot_backtest(
            _bars(), 100_000, key_value=1.0, atr_period=1, fx_commission_bps=0.20, fx_commission_min=2.0,
        )

        size = no_commission["trades"][0]["size"]
        expected_total_commission = portfolio.fx_commission(size * 11, 0.20, 2.0) + portfolio.fx_commission(
            size * 5.5, 0.20, 2.0
        )
        expected_equity = no_commission["equity_curve"][-1]["equity"] - expected_total_commission
        self.assertAlmostEqual(with_commission["equity_curve"][-1]["equity"], expected_equity, places=6)

    def test_no_spread_modeled_by_default(self):
        # spread_pips defaults to None -- fill prices are the raw signal
        # prices, identical to test_single_trade_sized_and_closed_correctly.
        result = run_ut_bot_backtest(_bars(), 100_000, key_value=1.0, atr_period=1)
        self.assertEqual(result["trades"][0]["fill_price"], 11)
        self.assertEqual(result["trades"][1]["fill_price"], 5.5)

    def test_spread_widens_entry_and_narrows_exit_fill_when_enabled(self):
        # symbol defaults to GBPUSD (a non-JPY pair): 2 pips -> half-spread
        # of 0.0001. BUY entry fills half a spread ABOVE the raw 11; SELL
        # exit fills half a spread BELOW the raw 5.5.
        result = run_ut_bot_backtest(_bars(), 100_000, key_value=1.0, atr_period=1, spread_pips=2.0)
        self.assertAlmostEqual(result["trades"][0]["fill_price"], 11.0001)
        self.assertAlmostEqual(result["trades"][1]["fill_price"], 5.4999)

        size = result["trades"][0]["size"]
        expected_equity = 100_000 + (5.4999 - 11.0001) * size
        self.assertAlmostEqual(result["equity_curve"][-1]["equity"], expected_equity, places=6)

    def test_spread_uses_jpy_pip_size_for_a_jpy_pair(self):
        # 2 pips on a JPY pair = 0.02 (not 0.0002) -> half-spread of 0.01.
        result = run_ut_bot_backtest(
            _bars(), 100_000, symbol="USDJPY", key_value=1.0, atr_period=1, spread_pips=2.0,
        )
        self.assertAlmostEqual(result["trades"][0]["fill_price"], 11.01)
        self.assertAlmostEqual(result["trades"][1]["fill_price"], 5.49)

    def test_no_regime_filter_by_default(self):
        # vol_filter_lookback defaults to None -- identical to
        # test_single_trade_sized_and_closed_correctly's result.
        result = run_ut_bot_backtest(_bars(), 100_000, key_value=1.0, atr_period=1)
        self.assertEqual(len(result["trades"]), 2)

    def test_regime_filter_wired_through_to_signal_layer(self):
        # lookback (1000) far exceeds this 8-bar fixture -- see
        # test_ut_bot_signals.TestFindUtBotLongTrades for why that
        # suppresses every entry. Confirms run_ut_bot_backtest actually
        # passes vol_filter_* down to find_ut_bot_long_trades rather than
        # silently ignoring them.
        result = run_ut_bot_backtest(
            _bars(), 100_000, key_value=1.0, atr_period=1,
            vol_filter_lookback=1000, vol_filter_atr_period=1,
        )
        self.assertEqual(result, {"trades": [], "equity_curve": []})


class TestRunUtBotConfirmedBacktestCommission(unittest.TestCase):
    def test_no_commission_modeled_by_default(self):
        result = run_ut_bot_confirmed_backtest(_confirmed_bars(), 100_000, key_value=0.3, atr_period=1)
        self.assertEqual(result["summary"]["total_trades"], 1)

    def test_round_trip_commission_netted_into_summary_and_equity(self):
        no_commission = run_ut_bot_confirmed_backtest(_confirmed_bars(), 100_000, key_value=0.3, atr_period=1)
        with_commission = run_ut_bot_confirmed_backtest(
            _confirmed_bars(), 100_000, key_value=0.3, atr_period=1, fx_commission_bps=0.20, fx_commission_min=2.0,
        )

        size = no_commission["trades"][0]["size"]
        expected_total_commission = portfolio.fx_commission(size * 9.10, 0.20, 2.0) + portfolio.fx_commission(
            size * 8.75, 0.20, 2.0
        )

        expected_gross_pnl = round(no_commission["summary"]["gross_pnl_usd"] - expected_total_commission, 2)
        self.assertAlmostEqual(with_commission["summary"]["gross_pnl_usd"], expected_gross_pnl, places=2)

    def test_no_spread_modeled_by_default(self):
        result = run_ut_bot_confirmed_backtest(_confirmed_bars(), 100_000, key_value=0.3, atr_period=1)
        self.assertEqual(result["trades"][0]["fill_price"], 9.10)
        self.assertEqual(result["trades"][1]["fill_price"], 8.75)

    def test_spread_applied_per_actual_fill_side_when_enabled(self):
        # Long round trip: entry is a BUY (fills above mid), exit is a
        # SELL (fills below mid) -- both moves hurt the long, same as a
        # real spread-crossing cost would. symbol defaults to GBPUSD: 2
        # pips -> half-spread of 0.0001.
        no_spread = run_ut_bot_confirmed_backtest(_confirmed_bars(), 100_000, key_value=0.3, atr_period=1)
        with_spread = run_ut_bot_confirmed_backtest(
            _confirmed_bars(), 100_000, key_value=0.3, atr_period=1, spread_pips=2.0,
        )

        self.assertAlmostEqual(with_spread["trades"][0]["fill_price"], 9.1001)
        self.assertAlmostEqual(with_spread["trades"][1]["fill_price"], 8.7499)

        size = no_spread["trades"][0]["size"]
        expected_gross_pnl = round(no_spread["summary"]["gross_pnl_usd"] - 0.0002 * size, 2)
        self.assertAlmostEqual(with_spread["summary"]["gross_pnl_usd"], expected_gross_pnl, places=2)


if __name__ == "__main__":
    unittest.main()
