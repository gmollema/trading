"""Unit tests for trading_bot.cli.rsi20_dip_etf_cycle.

This file exists as much to IMPORT the module as to assert about it.
Entrypoints nothing else imports are exactly where a syntax error
survives a fully green suite -- that happened to rsi2_cycle on
2026-09-05 and only surfaced when the scheduled task failed.

The end-to-end tests drive main() against a fake broker rather than TWS.

This variant now trades the SAME instrument (SPY) it signals off, at 1x,
with a hard $25 position ceiling -- a rewrite from the retired 3x,
whole-share 3USL version. Because signal_symbol == trade_symbol, there is
only one bars series per scenario; the price used to size an order is
whatever the signal series' last close happens to be. `rising_then_dipping_at`
/ `rising_then_recovering_at` below apply a constant additive shift to the
pinned crossing patterns from the signal tests so a scenario can still pick
a convenient round price -- a uniform shift changes no RSI or SMA
comparison, since both the closes and their moving average shift together.
"""

import json
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trading_bot.cli import rsi20_dip_etf_cycle as cycle


def make_bars(closes, start=date(2024, 1, 1)):
    n = len(closes)
    dates = [start + timedelta(days=i) for i in range(n)]
    opens = [closes[0]] + [round(closes[i - 1] + 0.5, 4) for i in range(1, n)]
    return {"date": dates, "open": opens,
            "high": [max(o, c) + 0.25 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 0.25 for o, c in zip(opens, closes)],
            "close": list(closes)}


def rising_then_dipping():
    """Long rise so the 200-bar trend filter passes, then exactly enough
    drop bars that RSI(20) crosses DOWN through 60 on the LAST bar.

    The bar count is pinned, not approximate: crossed_down fires only on
    the first sub-60 bar, so a series that overshoots ends on a "no
    signal" bar and the test would assert nothing. Wilder RSI(20) after
    250 rising bars reads 100.00, then 76.00, 60.67, 50.05 -- the
    crossing is the third drop.
    """
    c = [100 + 1.0 * k for k in range(250)]
    for _ in range(3):
        c.append(c[-1] - 6.0)
    return c


def rising_then_recovering():
    """...and then exactly enough recovery that RSI(20) crosses UP
    through 65 on the last bar (64.02 -> 67.39 on the fourth rise)."""
    c = rising_then_dipping()
    for _ in range(4):
        c.append(c[-1] + 4.0)
    return c


def shifted_to(closes, target_last_close):
    """Translate a close series by a constant so it ends at
    target_last_close, leaving every bar-to-bar move -- and therefore
    RSI and the trend filter -- unchanged. Lets a test pick a convenient
    absolute price without hand-tuning a whole new crossing pattern."""
    shift = target_last_close - closes[-1]
    return [c + shift for c in closes]


def rising_then_dipping_at(target_last_close):
    return shifted_to(rising_then_dipping(), target_last_close)


def rising_then_recovering_at(target_last_close):
    return shifted_to(rising_then_recovering(), target_last_close)


class _Contract:
    def __init__(self, symbol, exchange="SMART", currency="USD"):
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency
        self.localSymbol = symbol


class _Status:
    status = "Filled"
    avgFillPrice = 42.5


class _Order:
    orderId = 7


class _Trade:
    def __init__(self, price=42.5):
        self.orderStatus = _Status()
        self.orderStatus.avgFillPrice = price
        self.order = _Order()


class FakeIB:
    """Stands in for ib_async's IB. Serves different bars per contract so
    a test can tell which contract the caller actually read."""

    def __init__(self, bars_by_symbol, equity=(500.0, "USD")):
        self.bars_by_symbol = bars_by_symbol
        self.equity = equity
        self.history_requests = []

    def reqHistoricalData(self, contract, **kwargs):
        self.history_requests.append(contract.symbol)
        bars = self.bars_by_symbol[contract.symbol]

        class _Bar:
            def __init__(self, d, o, h, low, c):
                self.date, self.open, self.high, self.low, self.close = d, o, h, low, c

        return [_Bar(bars["date"][i], bars["open"][i], bars["high"][i],
                     bars["low"][i], bars["close"][i])
                for i in range(len(bars["close"]))]


class FakeBroker:
    def __init__(self, bars_by_symbol, equity=(500.0, "USD"), fill_price=42.5):
        self.ib = FakeIB(bars_by_symbol, equity)
        self.equity = equity
        self.fill_price = fill_price
        self.orders = []
        self.disconnected = False

    def qualify_stock(self, symbol, exchange="SMART", currency="USD",
                      primary_exchange=""):
        return _Contract(symbol, exchange, currency)

    def place_stock_order(self, contract, side, quantity, outside_rth=False):
        self.orders.append({"symbol": contract.symbol, "side": side,
                            "quantity": quantity, "outside_rth": outside_rth})
        return _Trade(self.fill_price)

    def net_liquidation(self):
        return self.equity

    def disconnect(self):
        self.disconnected = True


class CycleHarness:
    """Redirects every path the cycle writes into a temp directory."""

    def __init__(self, broker, rules=None):
        self.broker = broker
        self.rules = rules or {}
        self._stack = []

    def __enter__(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        rules_path = root / "rules.json"
        rules_path.write_text(json.dumps(self.rules), encoding="utf-8")
        self.log_path = root / "cycle.log"
        self.positions_path = root / "positions.json"
        self.trades_path = root / "trades.csv"
        self.heartbeat_path = root / "heartbeat.json"

        patches = [
            patch.object(cycle, "CYCLE_LOG_PATH", self.log_path),
            patch.object(cycle.etf, "RULES_PATH", rules_path),
            patch.object(cycle.etf, "POSITIONS_PATH", self.positions_path),
            patch.object(cycle.etf, "TRADES_CSV_PATH", self.trades_path),
            patch.object(cycle.etf, "HEARTBEAT_PATH", self.heartbeat_path),
            patch.object(cycle, "load_dotenv", lambda *a, **k: None),
            patch("trading_bot.broker.ibkr_client.IBKRClient",
                  lambda *a, **k: self.broker),
        ]
        for p in patches:
            p.start()
            self._stack.append(p)
        return self

    def __exit__(self, *exc):
        for p in reversed(self._stack):
            p.stop()
        self.tmp.cleanup()
        return False

    def events(self):
        if not self.log_path.exists():
            return []
        return [json.loads(x) for x in
                self.log_path.read_text(encoding="utf-8").splitlines()]

    def event(self, name):
        return next((e for e in self.events() if e["event"] == name), None)


class TestParseArgs(unittest.TestCase):
    def test_defaults_are_safe(self):
        args = cycle.parse_args([])
        self.assertFalse(args.arm)
        self.assertFalse(args.ignore_window)
        self.assertFalse(args.check)

    def test_flags_parse(self):
        args = cycle.parse_args(["--arm", "--ignore-window", "--check"])
        self.assertTrue(args.arm)
        self.assertTrue(args.ignore_window)
        self.assertTrue(args.check)


class TestIsolationFromTheOtherBots(unittest.TestCase):
    def test_client_id_does_not_collide(self):
        """Sharing a client id makes TWS drop one of the connections."""
        from trading_bot.cli import rsi2_cycle, rsi20_dip_cycle
        self.assertNotEqual(cycle.DEFAULT_CLIENT_ID, rsi2_cycle.DEFAULT_CLIENT_ID)
        self.assertNotEqual(cycle.DEFAULT_CLIENT_ID, rsi20_dip_cycle.DEFAULT_CLIENT_ID)
        self.assertNotIn(cycle.DEFAULT_CLIENT_ID, (0, 4, 95))

    def test_log_path_is_its_own(self):
        from trading_bot.cli import rsi2_cycle, rsi20_dip_cycle
        self.assertNotEqual(cycle.CYCLE_LOG_PATH, rsi2_cycle.CYCLE_LOG_PATH)
        self.assertNotEqual(cycle.CYCLE_LOG_PATH, rsi20_dip_cycle.CYCLE_LOG_PATH)


class TestLogEvent(unittest.TestCase):
    def test_writes_one_json_line_and_creates_the_directory(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "cycle.log"
            cycle.log_event({"event": "decision", "action": "hold"}, path=path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["event"], "decision")

    def test_survives_no_console(self):
        """The scheduled task runs pythonw, where sys.stdout is None and
        print() raises. The file write must still happen, so it goes
        first."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cycle.log"
            with patch.object(sys, "stdout", None):
                cycle.log_event({"event": "headless"}, path=path)
            self.assertIn("headless", path.read_text(encoding="utf-8"))

    def test_an_unwritable_log_does_not_take_the_cycle_down(self):
        with TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory")
            cycle.log_event({"event": "decision"}, path=blocker / "cycle.log")


class TestRecordFill(unittest.TestCase):
    def test_row_shape_matches_the_csv_header(self):
        row = cycle.record_fill(_Trade(42.5), _Contract("SPY", "SMART"),
                                "BUY", 2, "rsi_dip", "2026-09-04")
        from trading_bot import rsi2_live
        self.assertEqual(set(row), set(rsi2_live.TRADES_CSV_HEADER))
        self.assertEqual(row["size"], 2)
        self.assertEqual(row["fill_price"], 42.5)

    def test_unfilled_order_records_zero_rather_than_guessing(self):
        row = cycle.record_fill(_Trade(0.0), _Contract("SPY", "SMART"),
                                "BUY", 2, "rsi_dip", "2026-09-04")
        self.assertEqual(row["fill_price"], 0.0)


class TestOutsideWindowShortCircuits(unittest.TestCase):
    def test_a_weekend_run_returns_without_connecting(self):
        """The gate must be checked BEFORE the broker import, so a
        mis-scheduled run costs nothing and cannot touch TWS."""
        with TemporaryDirectory() as tmp:
            with patch.object(cycle, "CYCLE_LOG_PATH", Path(tmp) / "c.log"):
                with patch.object(cycle.etf, "in_decision_window", return_value=False):
                    with patch.object(cycle, "load_dotenv",
                                      side_effect=AssertionError("must not get this far")):
                        self.assertEqual(cycle.main([]), 0)
                rows = [json.loads(x) for x in
                        (Path(tmp) / "c.log").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["event"], "outside_decision_window")

    def test_check_ignores_the_window(self):
        """--check is a manual diagnostic; refusing to run it out of
        hours would make it useless."""
        bars = {"SPY": make_bars(rising_then_dipping_at(40.0))}
        broker = FakeBroker(bars)
        with CycleHarness(broker) as h:
            with patch.object(cycle.etf, "in_decision_window", return_value=False):
                self.assertEqual(cycle.main(["--check"]), 0)
            self.assertIsNotNone(h.event("check"))


class TestBadRulesStopTheCycle(unittest.TestCase):
    def test_currency_mismatch_returns_one_without_ordering(self):
        broker = FakeBroker({})
        with CycleHarness(broker, {"capital_currency": "EUR"}) as h:
            self.assertEqual(cycle.main(["--ignore-window"]), 1)
            self.assertEqual(h.event("bad_rules")["event"], "bad_rules")
        self.assertEqual(broker.orders, [])


class TestSignalAndTradeAreTheSameInstrument(unittest.TestCase):
    def test_decision_reads_and_trades_spy(self):
        """Unlike the retired leveraged-ETP version -- where the signal
        had to come from the index proxy while orders went to a
        different, leveraged instrument -- this variant trades exactly
        what it signals off."""
        bars = {"SPY": make_bars(rising_then_dipping_at(40.0))}
        broker = FakeBroker(bars)
        with CycleHarness(broker) as h:
            self.assertEqual(cycle.main(["--ignore-window"]), 0)
            decision = h.event("decision")
        self.assertEqual(decision["action"], "buy")
        self.assertEqual(decision["signal_symbol"], "SPY")
        self.assertEqual(decision["trade_symbol"], "SPY")
        self.assertIn("SPY", broker.ib.history_requests)


class TestDryRunPlacesNothing(unittest.TestCase):
    def test_a_buy_signal_without_arm_sends_no_order(self):
        bars = {"SPY": make_bars(rising_then_dipping_at(40.0))}
        broker = FakeBroker(bars)
        with CycleHarness(broker) as h:
            self.assertEqual(cycle.main(["--ignore-window"]), 0)
            self.assertEqual(h.event("dry_run_no_order")["would"], "buy")
            self.assertFalse(h.positions_path.exists())
        self.assertEqual(broker.orders, [])


class TestArmedBuy(unittest.TestCase):
    """$500 configured capital, 5% allocation, $25 hard ceiling,
    fractional SPY -- the rewrite from whole-share 3USL sizing."""

    def _run(self, equity=(500.0, "USD"), price=100.0):
        bars = {"SPY": make_bars(rising_then_dipping_at(price))}
        broker = FakeBroker(bars, equity=equity, fill_price=price)
        return broker, CycleHarness(broker, {})

    def test_places_a_buy_on_spy_and_saves_the_position(self):
        broker, harness = self._run()
        with harness as h:
            self.assertEqual(cycle.main(["--ignore-window", "--arm"]), 0)
            saved = json.loads(h.positions_path.read_text(encoding="utf-8"))
        self.assertEqual(len(broker.orders), 1)
        order = broker.orders[0]
        self.assertEqual(order["symbol"], "SPY")
        self.assertEqual(order["side"], "BUY")
        # allocation = 500 * 0.05 = 25; budget = min(25, 25) * 0.97 = 24.25;
        # at 100.00 that divides evenly -> 0.2425 shares, no flooring loss.
        self.assertAlmostEqual(order["quantity"], 0.2425)
        self.assertAlmostEqual(saved[0]["shares"], 0.2425)
        self.assertEqual(saved[0]["symbol"], "SPY")

    def test_order_does_not_go_outside_regular_hours(self):
        broker, harness = self._run()
        with harness:
            cycle.main(["--ignore-window", "--arm"])
        self.assertFalse(broker.orders[0]["outside_rth"])

    def test_account_equity_below_the_allocation_caps_the_size(self):
        broker, harness = self._run(equity=(200.0, "USD"))
        with harness:
            cycle.main(["--ignore-window", "--arm"])
        # allocation = min(25, 200*0.05=10) = 10; budget = min(10, 25)*0.97
        # = 9.7; at 100.00 -> 0.097 shares, not 0.2425.
        self.assertAlmostEqual(broker.orders[0]["quantity"], 0.097)

    def test_logs_the_effective_leverage_after_rounding(self):
        broker, harness = self._run()
        with harness as h:
            cycle.main(["--ignore-window", "--arm"])
            decision = h.event("decision")
        # 24.25 of 25.00 allocated capital invested = 0.97 -- the sizing
        # buffer itself, since 100.00 divides the budget evenly. This is
        # notional / capital now, not a multiple of a configured leverage
        # (trade_leverage is forced to 1 for this variant).
        self.assertAlmostEqual(decision["effective_leverage"], 0.97, places=2)
        self.assertAlmostEqual(decision["idle_pct"], 3.0, places=1)

    def test_an_unaffordable_price_skips_the_buy(self):
        """Even fractional sizing has a floor: a price so high that the
        budget cannot afford even one size_increment still skips the buy,
        the fractional-era equivalent of the old whole-share case."""
        broker, harness = self._run(price=5_000_000.0)
        with harness as h:
            self.assertEqual(cycle.main(["--ignore-window", "--arm"]), 0)
            self.assertIsNotNone(h.event("buy_skipped_unfundable"))
        self.assertEqual(broker.orders, [])


class TestFractionalOrdersAreNowAccepted(unittest.TestCase):
    """The retired 3USL version could never place a fractional order
    (IBKR error 10243) and validated that at config time. This variant is
    built around fractional SPY -- the placed quantity is expected to be
    a non-integer, not rejected as one."""

    def test_the_placed_quantity_is_fractional(self):
        broker, harness = TestArmedBuy()._run()
        with harness:
            cycle.main(["--ignore-window", "--arm"])
        qty = broker.orders[0]["quantity"]
        self.assertNotEqual(qty, int(qty))


class TestLeverageIsRefusedAtConfigTime(unittest.TestCase):
    """The bot is intentionally unleveraged; a config asking for broker
    leverage must be caught before connecting, not silently traded at 1x
    anyway."""

    def test_a_leveraged_config_stops_the_cycle_before_connecting(self):
        broker = FakeBroker({})
        with CycleHarness(broker, {"trade_leverage": 3}) as h:
            self.assertEqual(cycle.main(["--ignore-window", "--arm"]), 1)
            self.assertIn("exactly 1", h.event("bad_rules")["error"])
        self.assertEqual(broker.orders, [])


class TestArmedSell(unittest.TestCase):
    def _bars(self):
        return {"SPY": make_bars(rising_then_recovering_at(40.0))}

    def test_sells_the_share_count_held_not_a_freshly_sized_one(self):
        """The price has moved since entry; re-sizing the exit would
        leave a residue or go short."""
        broker = FakeBroker(self._bars(), fill_price=40.0)
        with CycleHarness(broker) as h:
            h.positions_path.write_text(json.dumps([{
                "symbol": "SPY", "local_symbol": "SPY", "shares": 0.07,
                "entry_price": 38.0, "entry_date": "2026-01-01T09:30:00-05:00",
                "entry_reason": "rsi_dip", "signal_bar_date": "2025-12-31",
            }]), encoding="utf-8")
            self.assertEqual(cycle.main(["--ignore-window", "--arm"]), 0)
            remaining = json.loads(h.positions_path.read_text(encoding="utf-8"))
        self.assertEqual(len(broker.orders), 1)
        self.assertEqual(broker.orders[0]["side"], "SELL")
        self.assertAlmostEqual(broker.orders[0]["quantity"], 0.07)
        self.assertEqual(remaining, [])

    def test_a_fractional_holding_is_sold_in_full(self):
        """int() on the held size would floor 1.3572 to 1 and leave a
        residue the file calls closed."""
        broker = FakeBroker(self._bars(), fill_price=40.0)
        with CycleHarness(broker) as h:
            h.positions_path.write_text(json.dumps([{
                "symbol": "SPY", "local_symbol": "SPY", "shares": 1.3572,
                "entry_price": 357.33, "entry_date": "2026-01-01T09:30:00-05:00",
                "entry_reason": "rsi_dip", "signal_bar_date": "2025-12-31",
            }]), encoding="utf-8")
            self.assertEqual(cycle.main(["--ignore-window", "--arm"]), 0)
            remaining = json.loads(h.positions_path.read_text(encoding="utf-8"))
        self.assertEqual(broker.orders[0]["quantity"], 1.3572)
        self.assertEqual(remaining, [])

    def test_two_recorded_positions_stop_the_cycle(self):
        """One at a time is the strategy's own rule; two means the state
        file is wrong and guessing which to trade is worse than halting."""
        broker = FakeBroker(self._bars())
        with CycleHarness(broker) as h:
            h.positions_path.write_text(
                json.dumps([{"symbol": "SPY", "shares": 1},
                            {"symbol": "SPY", "shares": 2}]), encoding="utf-8")
            self.assertEqual(cycle.main(["--ignore-window", "--arm"]), 1)
            self.assertIsNotNone(h.event("unexpected_multiple_positions"))
        self.assertEqual(broker.orders, [])


class TestHold(unittest.TestCase):
    def test_no_signal_writes_a_heartbeat_and_no_order(self):
        flat = make_bars([100.0 + 0.01 * k for k in range(300)])
        bars = {"SPY": flat}
        broker = FakeBroker(bars)
        with CycleHarness(broker) as h:
            self.assertEqual(cycle.main(["--ignore-window", "--arm"]), 0)
            self.assertEqual(h.event("decision")["action"], "hold")
            self.assertTrue(h.heartbeat_path.exists())
        self.assertEqual(broker.orders, [])


class TestLastCompletedClose(unittest.TestCase):
    def test_raises_when_every_bar_is_todays(self):
        """Sizing off no bar at all would be sizing off nothing."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        now = datetime(2026, 9, 9, 9, 30, tzinfo=et)
        bars = make_bars([40.0], start=now.date())
        with self.assertRaises(RuntimeError):
            cycle.last_completed_close(bars, now)


class TestEquityIsSoftFailed(unittest.TestCase):
    def test_a_summary_error_still_trades_the_allocation(self):
        """Equity is a cap, not the sizing input, so a summary hiccup
        must not skip a trading day."""
        bars = {"SPY": make_bars(rising_then_dipping_at(100.0))}
        broker = FakeBroker(bars, fill_price=100.0)
        broker.net_liquidation = lambda: (_ for _ in ()).throw(RuntimeError("no summary"))
        with CycleHarness(broker) as h:
            self.assertEqual(cycle.main(["--ignore-window", "--arm"]), 0)
            self.assertIsNotNone(h.event("equity_unavailable"))
        # equity_unavailable falls back to the full $25 allocation, same
        # as the default-equity case: 0.2425 shares at 100.00.
        self.assertAlmostEqual(broker.orders[0]["quantity"], 0.2425)


if __name__ == "__main__":
    unittest.main()
