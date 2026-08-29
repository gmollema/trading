"""Unit tests for trading_bot.cli.smc_entry_window.

The sweep needs the full bar cache. What is testable in isolation is the
cohort bucketing -- which has to bucket by when the BOT ACTS, not when
the bar opened, or the table points at the wrong half-hour -- and the
candidate window table, where mislabelling a window as reachable would
turn a diagnostic row into a proposal.
"""

import unittest

import pandas as pd

from trading_bot.cli import smc_entry_window as w


def _candidate(acted_at: str, fills: list[tuple[float, str]], entry_price: float = 100.0):
    """(entry_date, symbol, trade) as build_smc_candidates emits it."""
    return (
        pd.Timestamp(acted_at, tz="America/New_York"),
        "AAA",
        {"entry_price": entry_price,
         "fills": [{"qty_fraction": q, "price": p, "reason": r} for p, r, q in
                   [(f[0], f[1], 1.0 / len(fills)) for f in fills]]},
    )


class CohortTableTest(unittest.TestCase):
    def test_buckets_by_when_the_bot_acts_not_when_the_bar_opened(self):
        """A bar timestamped 09:55 is acted on at 10:02, so it belongs to
        the 10:00 bucket. Bucketing on the raw timestamp would credit the
        09:30 half-hour with a trade it never saw."""
        table = w.cohort_table([_candidate("2026-01-02 09:55", [(101.0, "tp1")])])
        self.assertEqual(list(table.index), ["10:00"])

    def test_counts_and_outcomes_per_bucket(self):
        table = w.cohort_table([
            _candidate("2026-01-02 11:00", [(102.0, "tp1")]),
            _candidate("2026-01-02 11:05", [(98.0, "stop")]),
        ])
        self.assertEqual(table.loc["11:00", "signals"], 2)
        self.assertEqual(table.loc["11:00", "win_pct"], 50.0)
        self.assertEqual(table.loc["11:00", "tp1_pct"], 50.0)
        self.assertEqual(table.loc["11:00", "stopped_pct"], 50.0)

    def test_total_is_the_mean_times_the_count(self):
        """What the bucket contributes in aggregate -- a great mean on
        three signals is not worth a window change."""
        table = w.cohort_table([
            _candidate("2026-01-02 11:00", [(102.0, "tp1")]),
            _candidate("2026-01-02 11:05", [(104.0, "tp1")]),
        ])
        row = table.loc["11:00"]
        self.assertAlmostEqual(row["total_ret_pct"], row["signals"] * row["mean_ret_pct"], places=6)

    def test_empty_input_is_an_empty_frame(self):
        self.assertTrue(w.cohort_table([]).empty)


class CandidateWindowsTest(unittest.TestCase):
    def test_the_current_window_is_among_the_candidates(self):
        """Otherwise there is nothing to compare a change against."""
        names = {c["name"] for c in w.CANDIDATE_WINDOWS}
        self.assertIn("1005-1530", names)

    def test_windows_the_schedule_cannot_serve_are_flagged(self):
        """The watchlist does not exist until the 09:40 prefilter, so
        anything starting before it is diagnostic, not a proposal."""
        by_name = {c["name"]: c for c in w.CANDIDATE_WINDOWS}
        self.assertTrue(by_name["none"]["unreachable"])
        self.assertTrue(by_name["0930-1530"]["unreachable"])
        self.assertFalse(by_name["0945-1530"]["unreachable"])

    def test_every_candidate_is_a_pair_of_bounds(self):
        for candidate in w.CANDIDATE_WINDOWS:
            self.assertEqual(len(candidate["window"]), 2, candidate["name"])

    def test_names_are_unique(self):
        names = [c["name"] for c in w.CANDIDATE_WINDOWS]
        self.assertEqual(len(names), len(set(names)))
