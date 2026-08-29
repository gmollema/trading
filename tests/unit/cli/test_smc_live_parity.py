"""Unit tests for trading_bot.cli.smc_live_parity.

The sweep needs the full bar cache; what is testable in isolation is that
the steps are genuinely cumulative and read the live bot's own config
keys. A step that silently dropped a constraint would still produce a
table, just one that answers a different question.
"""

import unittest

from trading_bot.cli import smc_live_parity as p

RULES = {"time_filter": {"earliest_entry_et": "10:05", "latest_entry_et": "15:30",
                         "force_close_et": "15:51"}}
WATCHLIST = {"2026-01-02": {"AAA"}}


class ParityStepsTest(unittest.TestCase):
    def _steps(self):
        return {s["name"]: s["kwargs"] for s in p.parity_steps(RULES, WATCHLIST)}

    def test_the_first_step_is_the_old_harness_basis(self):
        """No constraints at all -- what every SMC figure in this repo was
        scored on before 2026-08-29."""
        self.assertEqual(self._steps()["harness"], {})

    def test_each_step_keeps_everything_the_previous_one_added(self):
        steps = p.parity_steps(RULES, WATCHLIST)
        for earlier, later in zip(steps, steps[1:]):
            for key, value in earlier["kwargs"].items():
                self.assertEqual(later["kwargs"][key], value, f"{later['name']} dropped {key}")
            self.assertGreater(len(later["kwargs"]), len(earlier["kwargs"]), later["name"])

    def test_the_window_comes_from_the_rules_time_filter(self):
        self.assertEqual(self._steps()["+entry_window"]["entry_window_et"], ("10:05", "15:30"))

    def test_the_last_step_carries_all_three_constraints(self):
        final = self._steps()["+watchlist"]
        self.assertTrue(final["force_close_same_day"])
        self.assertEqual(final["entry_window_et"], ("10:05", "15:30"))
        self.assertIs(final["daily_watchlist"], WATCHLIST)

    def test_a_missing_time_filter_degrades_to_unbounded(self):
        """Open-ended bounds, not a crash: entry_window_mask reads None as
        'no bound on that side'."""
        steps = {s["name"]: s["kwargs"] for s in p.parity_steps({}, WATCHLIST)}
        self.assertEqual(steps["+entry_window"]["entry_window_et"], (None, None))
