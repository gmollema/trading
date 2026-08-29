"""Unit tests for trading_bot.cli.smc_fill_audit.

The stop leg is the one with a subtlety worth pinning: stopped_out records
the fill but not the level it was resting at, so the level has to be
tracked forward through every event that moves it. Get that wrong and the
tool reports a plausible number against the wrong baseline -- which is
worse than reporting nothing.
"""

import json
import tempfile
import unittest
from pathlib import Path

from trading_bot.cli import smc_fill_audit as audit


def _write(events: list[dict]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "smc-safety-check-log.json"
    tmp.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return tmp


class ReadEventsTest(unittest.TestCase):
    def test_reads_json_lines(self):
        path = _write([{"event": "a"}, {"event": "b"}])
        self.assertEqual([e["event"] for e in audit.read_events(path)], ["a", "b"])

    def test_a_truncated_final_line_is_skipped_not_fatal(self):
        """The bot may be mid-write; refusing to report on 200 good events
        because of one partial one would be the wrong trade."""
        path = _write([{"event": "a"}])
        path.write_text(path.read_text() + '{"event": "b", "sym')
        self.assertEqual([e["event"] for e in audit.read_events(path)], ["a"])

    def test_missing_file_is_empty(self):
        self.assertEqual(audit.read_events(Path("does_not_exist.json")), [])


class MeasureTest(unittest.TestCase):
    def test_entry_filled_above_its_signal_is_adverse(self):
        events = [{"event": "entry_opened", "symbol": "AAA",
                   "signal_price": 100.0, "entry_price": 100.5, "stop": 95.0}]
        [sample] = audit.measure(events)["entry"]
        self.assertAlmostEqual(sample["bps"], 50.0)

    def test_tp1_filled_below_its_level_is_adverse(self):
        events = [{"event": "tp1_done", "symbol": "AAA",
                   "tp1_level": 100.0, "fill_price": 99.5}]
        [sample] = audit.measure(events)["tp1"]
        self.assertAlmostEqual(sample["bps"], 50.0)

    def test_a_favourable_fill_reads_negative(self):
        """Adverse-positive throughout, so the legs compare directly."""
        events = [{"event": "entry_opened", "symbol": "AAA",
                   "signal_price": 100.0, "entry_price": 99.5, "stop": 95.0}]
        [sample] = audit.measure(events)["entry"]
        self.assertAlmostEqual(sample["bps"], -50.0)

    def test_stop_level_comes_from_the_entry(self):
        events = [
            {"event": "entry_opened", "symbol": "AAA",
             "signal_price": 100.0, "entry_price": 100.0, "stop": 95.0},
            {"event": "stopped_out", "symbol": "AAA", "fill_price": 94.5},
        ]
        [sample] = audit.measure(events)["stop"]
        self.assertEqual(sample["level"], 95.0)
        self.assertAlmostEqual(sample["bps"], 52.6, places=1)

    def test_tp1_moves_the_level_the_stop_is_measured_against(self):
        """Against the ENTRY stop this would read as a huge favourable
        fill; against the breakeven stop TP1 actually moved it to, it is
        the small adverse one it really was."""
        events = [
            {"event": "entry_opened", "symbol": "AAA",
             "signal_price": 100.0, "entry_price": 100.0, "stop": 95.0},
            {"event": "tp1_done", "symbol": "AAA", "new_stop": 100.0},
            {"event": "stopped_out", "symbol": "AAA", "fill_price": 99.9},
        ]
        [sample] = audit.measure(events)["stop"]
        self.assertEqual(sample["level"], 100.0)
        self.assertAlmostEqual(sample["bps"], 10.0)

    def test_a_replaced_stop_moves_the_level_too(self):
        events = [
            {"event": "entry_opened", "symbol": "AAA",
             "signal_price": 100.0, "entry_price": 100.0, "stop": 95.0},
            {"event": "stop_replaced", "symbol": "AAA", "stop_price": 97.0},
            {"event": "stopped_out", "symbol": "AAA", "fill_price": 97.0},
        ]
        [sample] = audit.measure(events)["stop"]
        self.assertEqual(sample["level"], 97.0)

    def test_levels_do_not_leak_between_trades_on_one_symbol(self):
        """The same symbol is traded again later; the second stop-out must
        not be scored against the first trade's level."""
        events = [
            {"event": "entry_opened", "symbol": "AAA",
             "signal_price": 100.0, "entry_price": 100.0, "stop": 95.0},
            {"event": "stopped_out", "symbol": "AAA", "fill_price": 95.0},
            {"event": "stopped_out", "symbol": "AAA", "fill_price": 80.0},
        ]
        self.assertEqual(len(audit.measure(events)["stop"]), 1)

    def test_entries_without_the_newer_fields_are_ignored(self):
        """Every entry logged before 2026-08-28 lacks signal_price. They
        are not zero-slippage fills, they are unmeasured ones."""
        events = [{"event": "entry_opened", "symbol": "AAA", "entry_price": 100.0, "stop": 95.0}]
        self.assertEqual(audit.measure(events)["entry"], [])


class VerdictTest(unittest.TestCase):
    def test_no_samples(self):
        self.assertEqual(audit.verdict(None, 2.0, 8), "no samples yet")

    def test_refuses_to_call_it_on_too_few_fills(self):
        """Replacing an assumption with three samples is not evidence."""
        stat = {"n": 3, "median": 40.0, "mean": 40.0, "min": 40.0, "max": 40.0}
        self.assertIn("too few", audit.verdict(stat, 2.0, 8))

    def test_a_rate_near_the_assumption_holds(self):
        stat = {"n": 10, "median": 2.7, "mean": 2.6, "min": 0.0, "max": 5.9}
        self.assertIn("basis holds", audit.verdict(stat, 2.0, 8))

    def test_a_moderately_high_rate_asks_for_a_re_run(self):
        stat = {"n": 10, "median": 8.0, "mean": 8.0, "min": 4.0, "max": 12.0}
        self.assertIn("re-run", audit.verdict(stat, 2.0, 8))

    def test_a_rate_like_the_measured_tp1_legs_condemns_the_figures(self):
        """48-79 bps is what the early TP1 market orders actually gave up."""
        stat = {"n": 10, "median": 60.0, "mean": 60.0, "min": 48.0, "max": 79.0}
        self.assertIn("do not survive", audit.verdict(stat, 2.0, 8))


class NotifyLineTest(unittest.TestCase):
    """The push has to carry its own verdict: it is read on a phone weeks
    later, by someone with none of this context in front of them."""

    BREAKEVEN = audit.BREAKEVEN_ENTRY_BPS

    def test_no_samples_says_keep_waiting(self):
        title, body = audit.notify_line(None, self.BREAKEVEN, 8)
        self.assertIn("not enough fills", title)
        self.assertIn("0 entry fills", body)

    def test_too_few_samples_does_not_call_it(self):
        stat = {"n": 3, "median": 40.0}
        title, body = audit.notify_line(stat, self.BREAKEVEN, 8)
        self.assertIn("not enough fills", title)
        self.assertNotIn("40", body)  # no verdict leaks out of a sample this thin

    def test_under_the_threshold_reads_as_viable_but_says_how_thin(self):
        title, body = audit.notify_line({"n": 12, "median": 2.4}, self.BREAKEVEN, 8)
        self.assertIn("clears break-even", title)
        self.assertIn("marginal", body)

    def test_over_the_threshold_says_what_to_do(self):
        title, body = audit.notify_line({"n": 12, "median": 9.1}, self.BREAKEVEN, 8)
        self.assertIn("exceeds break-even", title)
        self.assertIn("losing money", body)

    def test_the_boundary_counts_as_clearing_it(self):
        title, _ = audit.notify_line({"n": 12, "median": self.BREAKEVEN}, self.BREAKEVEN, 8)
        self.assertIn("clears break-even", title)

    def test_every_message_names_the_threshold_it_judged_against(self):
        for stat in (None, {"n": 3, "median": 1.0}, {"n": 12, "median": 1.0}, {"n": 12, "median": 99.0}):
            title, body = audit.notify_line(stat, self.BREAKEVEN, 8)
            self.assertTrue(str(self.BREAKEVEN) in body or "need 8" in body, title)

    def test_the_threshold_matches_the_sweep_that_produced_it(self):
        """A stale constant here would look authoritative while being
        wrong -- worse than having no threshold at all."""
        self.assertEqual(audit.BREAKEVEN_ENTRY_BPS, 3.5)
