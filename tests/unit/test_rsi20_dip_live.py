"""Unit tests for trading_bot.rsi20_dip_live.

The fixture geometry is the same as test_rsi20_dip_signals': a long rise
so the 200-bar trend filter has a value and passes, then a sharp drop so
RSI(20) crosses down through 60, then a recovery so it crosses back up
through 65. Bar indices are pinned in TestDecideTiming.
"""

import json
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from trading_bot import rsi20_dip_live as live
from trading_bot.backtest.rsi2_signals import wilder_rsi

ET = ZoneInfo("America/New_York")


def build_closes():
    c = [100 + 1.0 * k for k in range(250)]
    for _ in range(6):
        c.append(c[-1] - 6.0)
    for _ in range(24):
        c.append(c[-1] + 4.0)
    return c


def make_bars(closes, start=date(2024, 1, 1)):
    n = len(closes)
    dates = [start + timedelta(days=i) for i in range(n)]
    opens = [closes[0]] + [round(closes[i - 1] + 0.5, 4) for i in range(1, n)]
    return {"date": dates, "open": opens,
            "high": [max(o, c) + 0.25 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 0.25 for o, c in zip(opens, closes)],
            "close": list(closes)}


CLOSES = build_closes()
BARS = make_bars(CLOSES)
RULES = dict(live.DEFAULT_RULES)


class TestLoadRules(unittest.TestCase):
    def _write(self, tmp, payload):
        p = Path(tmp) / "rules.json"
        p.write_text(json.dumps(payload))
        return p

    def test_defaults_when_no_file(self):
        with TemporaryDirectory() as tmp:
            rules = live.load_rules(Path(tmp) / "absent.json")
        self.assertEqual(rules["symbol"], "MES")
        self.assertEqual(rules["entry_level"], 60.0)
        self.assertEqual(rules["exit_level"], 65.0)

    def test_file_overrides_defaults(self):
        with TemporaryDirectory() as tmp:
            rules = live.load_rules(self._write(tmp, {"entry_level": 55}))
        self.assertEqual(rules["entry_level"], 55)
        self.assertEqual(rules["exit_level"], 65.0)

    def test_equal_levels_are_refused(self):
        """Not a sizing typo but a LOGIC error: equal levels exit on the
        mirror of the entry, which is how the original reconstruction
        turned into a churning machine."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                live.load_rules(self._write(tmp, {"entry_level": 60, "exit_level": 60}))

    def test_inverted_levels_are_refused(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                live.load_rules(self._write(tmp, {"entry_level": 70, "exit_level": 60}))

    def test_contracts_above_the_ceiling_is_refused(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                live.load_rules(self._write(tmp, {"contracts": 10, "max_contracts": 1}))

    def test_other_bad_values_are_refused(self):
        with TemporaryDirectory() as tmp:
            for payload in ({"contracts": 0}, {"rsi_period": 0}, {"ma_period": 0},
                            {"offset_bars": -1}, {"min_days_to_expiry": 0},
                            {"entry_level": 0}, {"exit_level": 120}):
                with self.assertRaises(ValueError, msg=str(payload)):
                    live.load_rules(self._write(tmp, payload))


class TestBarsNeeded(unittest.TestCase):
    def test_accounts_for_the_offset(self):
        """The trend filter reads the SMA as it stood offset_bars ago, so
        the offset has to be added to the MA period, not ignored."""
        self.assertEqual(live.bars_needed({"ma_period": 200, "offset_bars": 3,
                                           "rsi_period": 20}), 204)

    def test_zero_offset_needs_one_fewer_than_three(self):
        with_offset = live.bars_needed({"ma_period": 200, "offset_bars": 3, "rsi_period": 20})
        without = live.bars_needed({"ma_period": 200, "offset_bars": 0, "rsi_period": 20})
        self.assertEqual(with_offset - without, 3)

    def test_a_long_rsi_can_dominate(self):
        self.assertEqual(live.bars_needed({"ma_period": 10, "offset_bars": 0,
                                           "rsi_period": 50}), 52)


class TestDecideTiming(unittest.TestCase):
    """RSI(20) on this fixture reads 60.67 at bar 251 and 50.05 at bar
    252, so the crossing down through 60 completes on bar 252. It reads
    63.49 at 263 and 65.97 at 264, so the crossing up through 65
    completes on 264. decide() sees bars 0..i and acts on the LAST one,
    so a history of 253 bars (indices 0..252) must say buy."""

    def setUp(self):
        self.rsi = wilder_rsi(CLOSES, 20)

    def _upto(self, count):
        return {k: v[:count] for k, v in BARS.items()}

    def test_rsi_brackets_the_entry_crossing(self):
        self.assertGreaterEqual(self.rsi[251], 60)
        self.assertLess(self.rsi[252], 60)

    def test_buys_when_the_last_bar_completed_the_crossing(self):
        d = live.decide(self._upto(253), None, RULES)
        self.assertEqual(d["action"], "buy")
        self.assertEqual(d["reason"], "rsi_dip")
        self.assertEqual(d["contracts"], 1)
        self.assertTrue(d["trend_ok"])

    def test_holds_the_bar_before_the_crossing_completes(self):
        d = live.decide(self._upto(252), None, RULES)
        self.assertEqual(d["action"], "hold")
        self.assertEqual(d["reason"], "no_signal")

    def test_does_not_re_enter_on_the_following_sub_level_bars(self):
        """Bars 253-255 are still below 60. A state test rather than a
        crossing test would buy again on each."""
        for count in (254, 255, 256):
            d = live.decide(self._upto(count), None, RULES)
            self.assertEqual(d["action"], "hold", msg=f"bars={count}")

    def test_holds_a_position_until_the_exit_crossing(self):
        d = live.decide(self._upto(264), {"contracts": 1}, RULES)
        self.assertEqual(d["action"], "hold")
        self.assertEqual(d["reason"], "holding")

    def test_sells_when_the_last_bar_completed_the_exit_crossing(self):
        d = live.decide(self._upto(265), {"contracts": 1}, RULES)
        self.assertEqual(d["action"], "sell")
        self.assertEqual(d["reason"], "rsi_exit")
        self.assertEqual(d["contracts"], 1)

    def test_short_history_reports_bars_needed(self):
        d = live.decide(self._upto(150), None, RULES)
        self.assertEqual(d["action"], "hold")
        self.assertIn("need 204 bars", d["reason"])

    def test_signal_bar_date_is_the_last_completed_bar(self):
        bars = self._upto(253)
        d = live.decide(bars, None, RULES)
        self.assertEqual(d["signal_bar_date"], str(bars["date"][-1]))


class TestTrendBreakExit(unittest.TestCase):
    def test_a_close_below_the_unshifted_ma_sells(self):
        closes = [100 + 1.0 * k for k in range(250)] + [
            250 - 20.0 * k for k in range(1, 13)]
        bars = make_bars(closes)
        d = live.decide(bars, {"contracts": 1}, RULES)
        self.assertEqual(d["action"], "sell")
        self.assertEqual(d["reason"], "trend_break")

    def test_the_rsi_exit_is_tested_before_the_trend_break(self):
        """Order matters only for the logged reason, but the reason has to
        match the backtest's or the two trade logs cannot be compared."""
        d = live.decide({k: v[:265] for k, v in BARS.items()}, {"contracts": 1}, RULES)
        self.assertEqual(d["reason"], "rsi_exit")


class TestExitPrecedence(unittest.TestCase):
    def test_a_veto_never_traps_an_open_position(self):
        """Whatever else happens, a held position must still be closeable
        -- an exit path that can be blocked is worse than no filter."""
        held = live.decide({k: v[:265] for k, v in BARS.items()}, {"contracts": 1}, RULES)
        self.assertEqual(held["action"], "sell")

    def test_position_contract_count_is_honoured_on_exit(self):
        rules = {**RULES, "contracts": 1, "max_contracts": 3}
        d = live.decide({k: v[:265] for k, v in BARS.items()}, {"contracts": 3}, rules)
        self.assertEqual(d["contracts"], 3)


class TestDecisionWindow(unittest.TestCase):
    """Shared with rsi2_live on purpose; asserted here so a change there
    cannot silently widen this bot's window."""

    def test_inside_the_window_on_a_weekday(self):
        self.assertTrue(live.in_decision_window(datetime(2026, 9, 8, 9, 30, tzinfo=ET)))

    def test_outside_the_window(self):
        self.assertFalse(live.in_decision_window(datetime(2026, 9, 8, 12, 0, tzinfo=ET)))

    def test_weekends_are_refused(self):
        self.assertFalse(live.in_decision_window(datetime(2026, 9, 5, 9, 30, tzinfo=ET)))


class TestStateIsSeparateFromRsi2(unittest.TestCase):
    """Two bots on one machine must not share state files, or enabling one
    would silently reach into the other's positions."""

    def test_paths_do_not_collide_with_rsi2(self):
        from trading_bot import rsi2_live
        pairs = [
            (live.RULES_PATH, rsi2_live.RSI2_RULES_PATH),
            (live.POSITIONS_PATH, rsi2_live.RSI2_POSITIONS_PATH),
            (live.TRADES_CSV_PATH, rsi2_live.RSI2_TRADES_CSV_PATH),
            (live.HEARTBEAT_PATH, rsi2_live.RSI2_HEARTBEAT_PATH),
        ]
        for mine, theirs in pairs:
            self.assertNotEqual(mine, theirs)


class TestExpiryAction(unittest.TestCase):
    def test_inside_the_window_rolls_out(self):
        self.assertEqual(
            live.expiry_action({"expiry": "20260920"}, date(2026, 9, 15), 10), "roll_out")

    def test_outside_the_window_does_nothing(self):
        self.assertEqual(
            live.expiry_action({"expiry": "20261220"}, date(2026, 9, 15), 10), "")

    def test_missing_or_unparseable_expiry_does_nothing(self):
        self.assertEqual(live.expiry_action({}, date(2026, 9, 15), 10), "")
        self.assertEqual(
            live.expiry_action({"expiry": "not-a-date"}, date(2026, 9, 15), 10), "")


if __name__ == "__main__":
    unittest.main()
