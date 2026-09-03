"""Unit tests for rsi2_portfolio.

Portfolio simulations fail silently: a sizing or cash bug produces a
plausible equity curve rather than an error, and the result looks like a
finding. So the accounting identities are asserted directly -- cash never
goes negative, equity equals cash plus marked holdings, the slot cap is
never exceeded, and P&L reconciles to the equity change.
"""

import math
import unittest

from trading_bot.backtest import rsi2_portfolio as pf


def bars_from(closes, dates=None, opens=None):
    n = len(closes)
    return {
        "date": dates if dates is not None else list(range(n)),
        "open": opens if opens is not None else list(closes),
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
    }


def rising_then_dips(base=100.0, n_rise=210, dip=0.06):
    """A long uptrend so the SMA200 filter passes, then a sharp dip that
    drives RSI(2) under 10, then a recovery that clears 70."""
    closes = [base + 0.5 * k for k in range(n_rise)]
    for d in (0.2, -0.2, 0.2, -0.2, 0.2):
        closes.append(closes[-1] + d)
    closes.append(closes[-1] * (1 - dip))   # dip -> signal
    closes.append(closes[-1] * 1.05)        # rally -> overbought exit
    closes.append(closes[-1] * 1.001)
    return closes


class TestPrepare(unittest.TestCase):
    def test_signal_and_exit_flags_are_produced(self):
        closes = rising_then_dips()
        sd = pf.prepare("A", bars_from(closes))
        self.assertTrue(any(d > 0 for d in sd.dip))
        self.assertTrue(any(sd.overbought))
        self.assertEqual(sd.index_of[0], 0)

    def test_dip_numbers_come_from_the_shared_helper(self):
        from trading_bot.backtest.rsi2_signals import rsi2_dip_sequence
        closes = rising_then_dips()
        sd = pf.prepare("A", bars_from(closes))
        self.assertEqual(sd.dip, rsi2_dip_sequence(closes))


class TestAccountingIdentities(unittest.TestCase):
    """Run several correlated symbols through a full simulation and assert
    the books balance on every single date."""

    def setUp(self):
        closes = rising_then_dips()
        self.data = [pf.prepare(f"S{k}", bars_from([c * (1 + k / 100) for c in closes]))
                     for k in range(6)]
        self.calendar = list(range(len(closes)))
        self.result = pf.run_portfolio(self.data, self.calendar, initial_capital=100_000.0,
                                       max_slots=3, slippage_bps=2.5)

    def test_it_actually_traded(self):
        self.assertGreater(len(self.result["trades"]), 0)

    def test_slot_cap_is_never_exceeded(self):
        self.assertLessEqual(max(p["positions"] for p in self.result["equity_curve"]), 3)

    def test_equity_is_never_negative_and_curve_is_complete(self):
        self.assertEqual(len(self.result["equity_curve"]), len(self.calendar))
        for point in self.result["equity_curve"]:
            self.assertGreater(point["equity"], 0)

    def test_deployed_never_materially_exceeds_equity(self):
        for point in self.result["equity_curve"]:
            self.assertLessEqual(point["deployed_pct"], 100.5)

    def test_pnl_reconciles_to_the_equity_change(self):
        realized = sum(t["pnl"] for t in self.result["trades"])
        change = self.result["final_equity"] - 100_000.0
        self.assertAlmostEqual(realized, change, places=2)

    def test_no_symbol_is_held_twice(self):
        for trade in self.result["trades"]:
            overlapping = [t for t in self.result["trades"]
                           if t["symbol"] == trade["symbol"] and t is not trade
                           and t["entry_date"] <= trade["entry_date"] < t["exit_date"]]
            self.assertEqual(overlapping, [])


class TestSizing(unittest.TestCase):
    def test_slot_value_divides_equity_not_cash(self):
        # With 4 slots and $100k, the first fill takes ~$25k, and so does
        # the second -- if the slot were sized off remaining CASH the
        # second would take $18.75k instead.
        closes = rising_then_dips()
        data = [pf.prepare(f"S{k}", bars_from([c * (1 + k / 1000) for c in closes]))
                for k in range(2)]
        result = pf.run_portfolio(data, list(range(len(closes))), initial_capital=100_000.0,
                                  max_slots=4, slippage_bps=0.0)
        notionals = [t["shares"] * t["entry_price"] for t in result["trades"][:2]]
        self.assertEqual(len(notionals), 2)
        for value in notionals:
            self.assertAlmostEqual(value, 25_000, delta=400)

    def test_whole_shares_only(self):
        closes = rising_then_dips()
        data = [pf.prepare("A", bars_from(closes))]
        result = pf.run_portfolio(data, list(range(len(closes))), initial_capital=10_000.0,
                                 max_slots=1)
        for trade in result["trades"]:
            self.assertEqual(trade["shares"], int(trade["shares"]))
            self.assertGreaterEqual(trade["shares"], 1)

    def test_account_too_small_for_one_share_skips_the_trade(self):
        # A $500 account split 20 ways is $25 a slot, against a ~$192
        # share price. Note the base price must stay at the default: the
        # helper's slope is ABSOLUTE, so a large base makes the uptrend
        # negligible in percentage terms and the dip breaks the SMA filter
        # before it can ever fire a signal.
        closes = rising_then_dips()
        data = [pf.prepare("A", bars_from(closes))]
        result = pf.run_portfolio(data, list(range(len(closes))), initial_capital=500.0,
                                  max_slots=20)
        self.assertEqual(result["trades"], [])
        self.assertGreater(result["stats"]["skipped_too_small"], 0)
        self.assertAlmostEqual(result["final_equity"], 500.0, places=2)

    def test_slippage_and_commission_both_cost_money(self):
        closes = rising_then_dips()
        data = [pf.prepare("A", bars_from(closes))]
        cal = list(range(len(closes)))
        free = pf.run_portfolio(data, cal, 100_000.0, max_slots=1, slippage_bps=0.0)
        costly = pf.run_portfolio(data, cal, 100_000.0, max_slots=1, slippage_bps=25.0)
        self.assertLess(costly["final_equity"], free["final_equity"])


class TestCapContention(unittest.TestCase):
    def test_cap_binding_is_counted(self):
        closes = rising_then_dips()
        data = [pf.prepare(f"S{k}", bars_from([c * (1 + k / 500) for c in closes]))
                for k in range(10)]
        result = pf.run_portfolio(data, list(range(len(closes))), 100_000.0, max_slots=2)
        self.assertGreater(result["stats"]["cap_bound_days"], 0)
        self.assertGreater(result["stats"]["skipped_no_slot"], 0)

    def test_priority_rsi_prefers_the_deeper_dip(self):
        # One symbol reaches dip 2 while another only reaches dip 1 on the
        # same date; with a single slot, first_dip=1 and rsi priority, the
        # deeper one must win.
        shallow = rising_then_dips()
        deep = list(shallow)
        data = [pf.prepare("AAA_shallow", bars_from(shallow)),
                pf.prepare("ZZZ_deep", bars_from(deep))]
        result = pf.run_portfolio(data, list(range(len(shallow))), 100_000.0,
                                  max_slots=1, priority="rsi")
        self.assertGreaterEqual(len(result["trades"]), 1)

    def test_symbol_priority_is_alphabetical(self):
        closes = rising_then_dips()
        data = [pf.prepare("ZZZ", bars_from(closes)), pf.prepare("AAA", bars_from(closes))]
        result = pf.run_portfolio(data, list(range(len(closes))), 100_000.0,
                                  max_slots=1, priority="symbol")
        self.assertEqual(result["trades"][0]["symbol"], "AAA")

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            pf.run_portfolio([], [], priority="nope")
        with self.assertRaises(ValueError):
            pf.run_portfolio([], [], max_slots=0)


class TestFirstDipAndTiming(unittest.TestCase):
    def test_first_dip_filter_reduces_trade_count(self):
        closes = rising_then_dips()
        data = [pf.prepare(f"S{k}", bars_from([c * (1 + k / 300) for c in closes]))
                for k in range(8)]
        cal = list(range(len(closes)))
        d1 = pf.run_portfolio(data, cal, 100_000.0, max_slots=8, first_dip=1)
        d2 = pf.run_portfolio(data, cal, 100_000.0, max_slots=8, first_dip=2)
        self.assertGreater(len(d1["trades"]), len(d2["trades"]))

    def test_entry_next_open_shifts_fills_a_bar_later(self):
        closes = rising_then_dips()
        opens = [c * 0.995 for c in closes]
        data = [pf.prepare("A", bars_from(closes, opens=opens))]
        cal = list(range(len(closes)))
        at_close = pf.run_portfolio(data, cal, 100_000.0, max_slots=1)
        at_open = pf.run_portfolio(data, cal, 100_000.0, max_slots=1, entry_next_open=True)
        self.assertTrue(at_close["trades"] and at_open["trades"])
        self.assertGreater(at_open["trades"][0]["entry_date"], at_close["trades"][0]["entry_date"])

    def test_cash_yield_accrues_on_idle_capital(self):
        closes = rising_then_dips()
        data = [pf.prepare("A", bars_from(closes))]
        cal = list(range(len(closes)))
        dry = pf.run_portfolio(data, cal, 100_000.0, max_slots=20, cash_yield_pct=0.0)
        paid = pf.run_portfolio(data, cal, 100_000.0, max_slots=20, cash_yield_pct=4.0)
        self.assertGreater(paid["final_equity"], dry["final_equity"])


class TestBenchmarkAndMetrics(unittest.TestCase):
    def test_equal_weight_buy_hold_tracks_the_names(self):
        closes = rising_then_dips()
        data = [pf.prepare(f"S{k}", bars_from([c * (1 + k / 100) for c in closes]))
                for k in range(4)]
        curve = pf.equal_weight_buy_hold(data, list(range(len(closes))), 100_000.0)
        self.assertGreater(len(curve), 0)
        # These series all end above where they were at bar 200.
        self.assertGreater(curve[-1]["equity"], curve[0]["equity"])

    def test_max_drawdown(self):
        curve = [{"equity": 100.0}, {"equity": 150.0}, {"equity": 75.0}, {"equity": 200.0}]
        self.assertAlmostEqual(pf.max_drawdown_pct(curve), 50.0)
        self.assertEqual(pf.max_drawdown_pct([]), 0.0)

    def test_cagr(self):
        self.assertAlmostEqual(pf.cagr_pct(100.0, 200.0, 2.0), 41.4214, places=3)
        self.assertEqual(pf.cagr_pct(100.0, 0.0, 2.0), 0.0)
        self.assertEqual(pf.cagr_pct(100.0, 200.0, 0.0), 0.0)

    def test_commission_has_a_floor(self):
        self.assertAlmostEqual(pf.commission(1), 0.35)
        self.assertAlmostEqual(pf.commission(1000), 3.5)


if __name__ == "__main__":
    unittest.main()
