"""Unit tests for trading_bot.rsi2_live.

The centrepiece is TestBacktestParity, which replays a bar history one day
at a time through `decide` and asserts the resulting trades are identical
to what find_rsi2_scale_in_trades produces on the same bars with
next-open fills. That equivalence is the whole claim of the live bot: it
is supposed to be the backtested strategy, not a reimplementation that
resembles it. Everything else here guards a specific way the live path
could silently diverge -- above all `completed_bars`, since RSI(2) on a
partial bar is simply a different indicator.
"""

import csv
import json
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from trading_bot import rsi2_live
from trading_bot.backtest.rsi2_signals import find_rsi2_scale_in_trades

ET = ZoneInfo("America/New_York")


def build_closes():
    """Same geometry as the scale-in tests: a long trend so the SMA200
    filter passes, then oscillation, then three dips and a rally."""
    c = [100 + 1.5 * k for k in range(200)]
    for d in (0.5, -0.5, 0.5, -0.5, 0.5):
        c.append(c[-1] + d)
    c.append(c[-1] - 5)
    c.append(c[-1] + 0.5)
    c.append(c[-1] - 3)
    c.append(c[-1] + 0.5)
    c.append(c[-1] - 3)
    c.append(c[-1] + 40)
    c.append(c[-1] + 0.5)
    return c


CLOSES = build_closes()


def make_bars(closes, start=date(2024, 1, 1)):
    """Bars with one calendar day per bar and opens a shade off the prior
    close, so an open-fill is distinguishable from a close-fill."""
    n = len(closes)
    dates = [start + timedelta(days=i) for i in range(n)]
    opens = [closes[0]] + [round(closes[i - 1] + 0.1, 4) for i in range(1, n)]
    return {"date": dates, "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": list(closes)}


BARS = make_bars(CLOSES)
RULES = {**rsi2_live.DEFAULT_RULES, "sma_period": 200}


class TestLoadRules(unittest.TestCase):
    def _write(self, tmp, payload):
        p = Path(tmp) / "rsi2_rules.json"
        p.write_text(json.dumps(payload))
        return p

    def test_defaults_when_no_file(self):
        with TemporaryDirectory() as tmp:
            rules = rsi2_live.load_rules(Path(tmp) / "absent.json")
        self.assertEqual(rules["symbol"], "ES")
        self.assertEqual(rules["contracts"], 1)

    def test_file_overrides_defaults(self):
        with TemporaryDirectory() as tmp:
            rules = rsi2_live.load_rules(self._write(tmp, {"entry_level": 5}))
        self.assertEqual(rules["entry_level"], 5)
        self.assertEqual(rules["exit_level"], 70)

    def test_contracts_above_the_ceiling_is_refused(self):
        """The one typo that costs real money on a $50-a-point contract."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                rsi2_live.load_rules(self._write(tmp, {"contracts": 10, "max_contracts": 1}))

    def test_raising_the_ceiling_deliberately_is_allowed(self):
        with TemporaryDirectory() as tmp:
            rules = rsi2_live.load_rules(self._write(tmp, {"contracts": 3, "max_contracts": 5}))
        self.assertEqual(rules["contracts"], 3)

    def test_bad_levels_are_refused(self):
        with TemporaryDirectory() as tmp:
            for payload in ({"entry_level": 80, "exit_level": 70}, {"entry_level": 0},
                            {"exit_level": 120}, {"first_dip": 0},
                            {"contracts": 0}, {"min_days_to_expiry": 0}):
                with self.assertRaises(ValueError, msg=str(payload)):
                    rsi2_live.load_rules(self._write(tmp, payload))


class TestDecisionWindow(unittest.TestCase):
    def test_inside_the_window_on_a_weekday(self):
        self.assertTrue(rsi2_live.in_decision_window(datetime(2026, 9, 2, 9, 26, tzinfo=ET)))
        self.assertTrue(rsi2_live.in_decision_window(datetime(2026, 9, 2, 9, 44, tzinfo=ET)))

    def test_outside_the_window(self):
        for h, m in ((9, 20), (9, 50), (13, 0), (16, 30)):
            self.assertFalse(rsi2_live.in_decision_window(datetime(2026, 9, 2, h, m, tzinfo=ET)),
                             f"{h}:{m}")

    def test_weekend_is_always_out(self):
        # 2026-09-05 is a Saturday, 09-06 a Sunday.
        self.assertFalse(rsi2_live.in_decision_window(datetime(2026, 9, 5, 9, 30, tzinfo=ET)))
        self.assertFalse(rsi2_live.in_decision_window(datetime(2026, 9, 6, 9, 30, tzinfo=ET)))


class TestCompletedBars(unittest.TestCase):
    def test_todays_partial_bar_is_dropped(self):
        now = datetime(2024, 1, 1, 9, 30, tzinfo=ET)
        bars = make_bars([100.0, 101.0, 102.0], start=date(2023, 12, 30))
        # bars are 12-30, 12-31, 01-01; "now" is 01-01 so the last is partial
        out = rsi2_live.completed_bars(bars, now)
        self.assertEqual(len(out["close"]), 2)
        self.assertEqual(out["close"][-1], 101.0)

    def test_nothing_dropped_when_no_bar_is_dated_today(self):
        now = datetime(2024, 1, 5, 9, 30, tzinfo=ET)
        bars = make_bars([100.0, 101.0, 102.0], start=date(2024, 1, 1))
        self.assertEqual(len(rsi2_live.completed_bars(bars, now)["close"]), 3)

    def test_all_keys_survive_and_stay_aligned(self):
        now = datetime(2024, 1, 3, 9, 30, tzinfo=ET)
        out = rsi2_live.completed_bars(make_bars([1.0, 2.0, 3.0], start=date(2024, 1, 1)), now)
        self.assertEqual(set(out), {"date", "open", "high", "low", "close"})
        self.assertEqual({len(v) for v in out.values()}, {2})


class TestDecide(unittest.TestCase):
    def test_too_few_bars_holds(self):
        d = rsi2_live.decide(make_bars([100.0] * 10), None, RULES)
        self.assertEqual(d["action"], "hold")
        self.assertIn("need", d["reason"])

    def test_buys_on_the_dip_bar(self):
        # bar 205 is the first dip in this fixture
        bars = {k: v[:206] for k, v in BARS.items()}
        d = rsi2_live.decide(bars, None, RULES)
        self.assertEqual(d["action"], "buy")
        self.assertEqual(d["reason"], "rsi2_dip_1")
        self.assertEqual(d["contracts"], 1)
        self.assertLess(d["rsi"], 10)

    def test_holds_flat_when_no_signal(self):
        bars = {k: v[:205] for k, v in BARS.items()}
        d = rsi2_live.decide(bars, None, RULES)
        self.assertEqual(d["action"], "hold")
        self.assertEqual(d["reason"], "no_signal")

    def test_holds_a_position_while_not_overbought(self):
        bars = {k: v[:208] for k, v in BARS.items()}
        d = rsi2_live.decide(bars, {"contracts": 1}, RULES)
        self.assertEqual(d["action"], "hold")
        self.assertEqual(d["reason"], "holding")

    def test_sells_when_overbought(self):
        bars = {k: v[:211] for k, v in BARS.items()}  # bar 210 is the RSI-95 rally
        d = rsi2_live.decide(bars, {"contracts": 1}, RULES)
        self.assertEqual(d["action"], "sell")
        self.assertEqual(d["reason"], "rsi_exit")
        self.assertEqual(d["contracts"], 1)

    def test_exit_takes_priority_over_a_fresh_entry(self):
        """Mirrors the backtest's per-bar order: an overbought close ends
        the position before any new signal is considered."""
        bars = {k: v[:211] for k, v in BARS.items()}
        held = rsi2_live.decide(bars, {"contracts": 1}, RULES)
        flat = rsi2_live.decide(bars, None, RULES)
        self.assertEqual(held["action"], "sell")
        self.assertEqual(flat["action"], "hold")

    def test_first_dip_two_skips_the_first_dip(self):
        rules = {**RULES, "first_dip": 2}
        at_first = rsi2_live.decide({k: v[:206] for k, v in BARS.items()}, None, rules)
        at_second = rsi2_live.decide({k: v[:208] for k, v in BARS.items()}, None, rules)
        self.assertEqual(at_first["action"], "hold")
        self.assertEqual(at_second["action"], "buy")
        self.assertEqual(at_second["reason"], "rsi2_dip_2")

    def test_position_contract_count_is_used_for_the_exit(self):
        bars = {k: v[:211] for k, v in BARS.items()}
        d = rsi2_live.decide(bars, {"contracts": 3}, RULES)
        self.assertEqual(d["contracts"], 3)


class TestBacktestParity(unittest.TestCase):
    """Replay the history one day at a time through `decide` and check the
    trades match find_rsi2_scale_in_trades with next-open fills."""

    def _replay(self, rules):
        trades = []
        position = None
        for i in range(len(BARS["close"]) - 1):
            visible = {k: v[:i + 1] for k, v in BARS.items()}
            d = rsi2_live.decide(visible, position, rules)
            fill_idx = i + 1  # the order fills at the next bar's open
            if d["action"] == "buy":
                position = {"entry_idx": fill_idx, "contracts": d["contracts"]}
            elif d["action"] == "sell":
                trades.append((position["entry_idx"], fill_idx))
                position = None
        if position is not None:
            trades.append((position["entry_idx"], None))
        return trades

    def _backtest(self, rules):
        out = find_rsi2_scale_in_trades(
            BARS, rsi_period=rules["rsi_period"], entry_level=rules["entry_level"],
            exit_level=rules["exit_level"], sma_period=rules["sma_period"],
            max_positions=1, first_dip=rules["first_dip"],
            entry_timing="next_open", exit_timing="next_open")
        return [(t["entry_idx"], None if t["reason"] == "end_of_data" else t["exit_idx"])
                for t in out]

    def test_parity_on_the_fixture(self):
        live, back = self._replay(RULES), self._backtest(RULES)
        self.assertTrue(back, "fixture produced no backtest trades")
        self.assertEqual(live, back)

    def test_parity_with_first_dip_two(self):
        rules = {**RULES, "first_dip": 2}
        self.assertEqual(self._replay(rules), self._backtest(rules))

    def test_parity_on_a_long_noisy_series(self):
        global BARS
        closes, price, state = [], 1000.0, 424242
        for _ in range(2500):
            state = (1103515245 * state + 12345) % (2 ** 31)
            price *= 1 + ((state / (2 ** 31)) - 0.49) * 0.02
            closes.append(price)
        saved = BARS
        try:
            BARS = make_bars(closes)
            live, back = self._replay(RULES), self._backtest(RULES)
            self.assertGreater(len(back), 15)
            self.assertEqual(live, back)
        finally:
            BARS = saved


class TestExpiryAction(unittest.TestCase):
    def test_inside_the_window_rolls_out(self):
        pos = {"expiry": "20260918"}
        self.assertEqual(rsi2_live.expiry_action(pos, date(2026, 9, 10), 10), "roll_out")

    def test_outside_the_window_does_nothing(self):
        pos = {"expiry": "20261218"}
        self.assertEqual(rsi2_live.expiry_action(pos, date(2026, 9, 10), 10), "")

    def test_missing_or_unparseable_expiry_does_nothing(self):
        self.assertEqual(rsi2_live.expiry_action({}, date(2026, 9, 10), 10), "")
        self.assertEqual(rsi2_live.expiry_action({"expiry": "nope"}, date(2026, 9, 10), 10), "")


class TestStateIO(unittest.TestCase):
    def test_positions_round_trip(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "pos.json"
            rsi2_live.save_positions([{"symbol": "ES", "contracts": 1}], p)
            self.assertEqual(rsi2_live.load_positions(p), [{"symbol": "ES", "contracts": 1}])

    def test_missing_file_reads_as_flat(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(rsi2_live.load_positions(Path(tmp) / "absent.json"), [])

    def test_corrupt_file_reads_as_flat_rather_than_raising(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "pos.json"
            p.write_text("{not json")
            self.assertEqual(rsi2_live.load_positions(p), [])

    def test_non_list_payload_reads_as_flat(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "pos.json"
            p.write_text('{"symbol": "ES"}')
            self.assertEqual(rsi2_live.load_positions(p), [])

    def test_save_leaves_no_temp_file_behind(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "pos.json"
            rsi2_live.save_positions([], p)
            self.assertEqual([f.name for f in Path(tmp).iterdir()], ["pos.json"])

    def test_trade_log_writes_a_header_once(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "trades.csv"
            row = {k: "x" for k in rsi2_live.TRADES_CSV_HEADER}
            rsi2_live.append_trade(row, p)
            rsi2_live.append_trade(row, p)
            with p.open() as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows[0], rsi2_live.TRADES_CSV_HEADER)
        self.assertEqual(len(rows), 3)

    def test_trade_log_ignores_extra_keys(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "trades.csv"
            row = {k: "x" for k in rsi2_live.TRADES_CSV_HEADER}
            rsi2_live.append_trade({**row, "unexpected": "y"}, p)
            self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()
