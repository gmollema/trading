"""Unit tests for trading_bot.smc_live (pure live-plumbing helpers for the
SMC paper-trading cycle)."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_bot import smc_live
from trading_bot.backtest import portfolio

RULES = {
    "time_filter": {"earliest_entry_et": "10:05", "latest_entry_et": "15:30", "force_close_et": "15:51"},
    "risk": {
        "max_risk_per_trade_pct": 1.0,
        "max_position_size_pct_of_portfolio": 10.0,
        "max_concurrent_positions": 2,
    },
    "reactive_derisk": {"window": 2, "pf_threshold": 0.8, "size_mult": 0.3},
}


def _dt(hh, mm, weekday_date="2024-01-02"):  # a Tuesday
    return datetime.fromisoformat(f"{weekday_date} {hh:02d}:{mm:02d}:00")


class TestGetMarketStatus(unittest.TestCase):
    def test_weekend(self):
        self.assertEqual(smc_live.get_market_status(_dt(11, 0, "2024-01-06"), RULES), "weekend")

    def test_windows(self):
        self.assertEqual(smc_live.get_market_status(_dt(9, 59), RULES), "too_early")
        self.assertEqual(smc_live.get_market_status(_dt(10, 2), RULES), "manage_only")
        self.assertEqual(smc_live.get_market_status(_dt(10, 5), RULES), "ok")
        self.assertEqual(smc_live.get_market_status(_dt(15, 29), RULES), "ok")
        self.assertEqual(smc_live.get_market_status(_dt(15, 30), RULES), "manage_only")
        self.assertEqual(smc_live.get_market_status(_dt(15, 51), RULES), "force_close")
        self.assertEqual(smc_live.get_market_status(_dt(16, 0), RULES), "force_close")
        self.assertEqual(smc_live.get_market_status(_dt(16, 1), RULES), "closed")


class TestReadWatchlist(unittest.TestCase):
    def test_parses_tickers_and_skips_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smc_watchlist.txt"
            path.write_text("# header\nNVDA  # uptrend\n\nBRK B\n#AAPL\n")
            self.assertEqual(smc_live.read_watchlist(path), ["NVDA", "BRK B"])

    def test_missing_file_is_empty(self):
        self.assertEqual(smc_live.read_watchlist(Path("does_not_exist_watchlist.txt")), [])


class TestBarsFrameToDict(unittest.TestCase):
    def test_roundtrip_shape(self):
        idx = pd.date_range("2024-01-02 09:30", periods=3, freq="5min", tz="America/New_York")
        frame = pd.DataFrame(
            {"Open": [1.0, 2.0, 3.0], "High": [1.5, 2.5, 3.5], "Low": [0.5, 1.5, 2.5], "Close": [1.2, 2.2, 3.2]},
            index=idx,
        )
        bars = smc_live.bars_frame_to_dict(frame)
        self.assertEqual(bars["open"], [1.0, 2.0, 3.0])
        self.assertEqual(bars["high"], [1.5, 2.5, 3.5])
        self.assertEqual(bars["low"], [0.5, 1.5, 2.5])
        self.assertEqual(bars["close"], [1.2, 2.2, 3.2])
        self.assertEqual(list(bars["date"]), list(idx))


def _write_trades_csv(path: Path, rows: list[tuple[str, str, str, int, float]]) -> None:
    lines = [",".join(smc_live.TRADES_CSV_HEADER)]
    for ts, symbol, side, size, price in rows:
        lines.append(f"{ts},{symbol},{side},{size},{price},1,Filled,test")
    path.write_text("\n".join(lines) + "\n")


class TestReactiveSizeMultiplier(unittest.TestCase):
    def test_missing_file_leaves_size_unaffected(self):
        self.assertEqual(smc_live.reactive_size_multiplier(Path("nope.csv"), 2, 0.8, 0.3), 1.0)

    def test_window_not_full_leaves_size_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smc_trades.csv"
            _write_trades_csv(path, [
                ("2024-01-02T15:00:00+00:00", "AAA", "BUY", 10, 100.0),
                ("2024-01-02T16:00:00+00:00", "AAA", "SELL", 10, 90.0),  # 1 losing round trip < window=2
            ])
            self.assertEqual(smc_live.reactive_size_multiplier(path, 2, 0.8, 0.3), 1.0)

    def test_trailing_losses_scale_size_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smc_trades.csv"
            _write_trades_csv(path, [
                ("2024-01-02T15:00:00+00:00", "AAA", "BUY", 10, 100.0),
                ("2024-01-02T16:00:00+00:00", "AAA", "SELL", 10, 90.0),
                ("2024-01-03T15:00:00+00:00", "BBB", "BUY", 10, 50.0),
                ("2024-01-03T16:00:00+00:00", "BBB", "SELL", 10, 45.0),
            ])
            self.assertEqual(smc_live.reactive_size_multiplier(path, 2, 0.8, 0.3), 0.3)

    def test_trailing_wins_leave_size_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smc_trades.csv"
            _write_trades_csv(path, [
                ("2024-01-02T15:00:00+00:00", "AAA", "BUY", 10, 100.0),
                ("2024-01-02T16:00:00+00:00", "AAA", "SELL", 10, 110.0),
                ("2024-01-03T15:00:00+00:00", "BBB", "BUY", 10, 50.0),
                ("2024-01-03T16:00:00+00:00", "BBB", "SELL", 10, 55.0),
            ])
            self.assertEqual(smc_live.reactive_size_multiplier(path, 2, 0.8, 0.3), 1.0)


class TestEntrySize(unittest.TestCase):
    def test_matches_backtest_sizing_when_derisk_inactive(self):
        expected = portfolio.position_size(100_000, 1.0, 11.0, 8.0, 10.0)
        size = smc_live.entry_size(100_000, 11.0, 8.0, RULES, trades_csv_path=Path("nope.csv"))
        self.assertEqual(size, expected)

    def test_derisk_scales_entry_down_after_trailing_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smc_trades.csv"
            _write_trades_csv(path, [
                ("2024-01-02T15:00:00+00:00", "AAA", "BUY", 10, 100.0),
                ("2024-01-02T16:00:00+00:00", "AAA", "SELL", 10, 90.0),
                ("2024-01-03T15:00:00+00:00", "BBB", "BUY", 10, 50.0),
                ("2024-01-03T16:00:00+00:00", "BBB", "SELL", 10, 45.0),
            ])
            baseline = portfolio.position_size(100_000, 1.0, 11.0, 8.0, 10.0)
            size = smc_live.entry_size(100_000, 11.0, 8.0, RULES, trades_csv_path=path)
            self.assertEqual(size, int(baseline * 0.3))


if __name__ == "__main__":
    unittest.main()


class TestEntryRules(unittest.TestCase):
    """The entry spec is read from config by both the live bot and the
    backtests, so it has exactly one reading."""

    def test_missing_block_keeps_the_original_spec(self):
        self.assertEqual(
            smc_live.entry_rules({}),
            {"fill": "level", "require_ob_reclaim": False},
        )

    def test_partial_block_is_filled_in(self):
        self.assertEqual(
            smc_live.entry_rules({"entry": {"fill": "next_open"}}),
            {"fill": "next_open", "require_ob_reclaim": False},
        )

    def test_values_are_read_through(self):
        spec = smc_live.entry_rules({"entry": {"fill": "next_high", "require_ob_reclaim": True}})
        self.assertEqual(spec, {"fill": "next_high", "require_ob_reclaim": True})

    def test_unknown_fill_is_rejected(self):
        """Only the backtests consume `fill`, so a typo would otherwise go
        unnoticed until a run misdescribed what the live bot had done."""
        with self.assertRaises(ValueError):
            smc_live.entry_rules({"entry": {"fill": "ob_high"}})

    def test_the_shipped_rules_file_parses(self):
        self.assertIn("fill", smc_live.entry_rules(smc_live.load_smc_rules()))


class TestExitRules(unittest.TestCase):
    def test_missing_block_keeps_the_original_spec(self):
        self.assertEqual(
            smc_live.exit_rules({}),
            {"fill": "level", "tp1_resting_limit": False},
        )

    def test_partial_block_is_filled_in(self):
        self.assertEqual(
            smc_live.exit_rules({"exit": {"fill": "next_open"}}),
            {"fill": "next_open", "tp1_resting_limit": False},
        )

    def test_unknown_fill_is_rejected(self):
        with self.assertRaises(ValueError):
            smc_live.exit_rules({"exit": {"fill": "pivot_close"}})

    def test_the_shipped_rules_file_parses(self):
        self.assertIn("fill", smc_live.exit_rules(smc_live.load_smc_rules()))

    def test_entry_and_exit_specs_are_read_independently(self):
        rules = {"entry": {"fill": "next_open"}, "exit": {"fill": "level"}}
        self.assertEqual(smc_live.entry_rules(rules)["fill"], "next_open")
        self.assertEqual(smc_live.exit_rules(rules)["fill"], "level")
