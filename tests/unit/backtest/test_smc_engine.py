"""Unit tests for trading_bot.backtest.smc_engine.run_smc_backtest.

Reuses the same hand-traced bar sequence from test_smc_signals.py's
lifecycle test (see that file for the full derivation) so the expected
entry/exit prices are already verified; these tests focus on the
portfolio-level layer on top: position sizing, equity updates, and the
max_concurrent_positions cap.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_bot.backtest import portfolio
from trading_bot.backtest.smc_engine import run_smc_backtest

# Same OHLC sequence as test_smc_signals.py's test_full_trade_lifecycle:
# entry @ 11 (idx 8), stop @ 8 (idx 5's low), no TP1, full exit @ 17 (idx 11,
# confirmed/filled at idx 13). See that file for the bar-by-bar trace.
LIFECYCLE_ROWS = [
    (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
    (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19), (19, 19, 10, 13),
    (13, 16, 13, 15), (15, 19, 15, 18), (18, 25, 17, 17), (17, 20, 15, 16), (16, 18, 13, 14),
]


def _write_intraday_csv(path: Path, rows: list[tuple[float, float, float, float]], start="2024-01-02 10:00:00"):
    lines = ["date,open,high,low,close,volume"]
    ts = pd.Timestamp(start)
    for o, h, low, c in rows:
        lines.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')}-05:00,{o},{h},{low},{c},1000")
        ts += pd.Timedelta(minutes=5)
    path.write_text("\n".join(lines) + "\n")


class TestRunSmcBacktestSingleSymbol(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.intraday_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_no_cached_data_returns_empty_result(self):
        result = run_smc_backtest(["NOPE"], 100_000, intraday_dir=self.intraday_dir)
        self.assertEqual(result, {"trades": [], "equity_curve": []})

    def test_single_trade_sized_and_closed_correctly(self):
        _write_intraday_csv(self.intraday_dir / "TEST.csv", LIFECYCLE_ROWS)

        result = run_smc_backtest(["TEST"], 100_000, risk_pct=1.0, max_position_pct=10.0, intraday_dir=self.intraday_dir)
        trades = result["trades"]

        self.assertEqual(len(trades), 2)
        buy, sell = trades
        self.assertEqual(buy["side"], "BUY")
        self.assertEqual(buy["fill_price"], 11)
        expected_size = portfolio.position_size(100_000, 1.0, 11, 8, 10.0)
        self.assertEqual(buy["size"], expected_size)

        self.assertEqual(sell["side"], "SELL")
        self.assertEqual(sell["fill_price"], 17)
        self.assertEqual(sell["size"], expected_size)
        self.assertEqual(sell["reason"], "new_high_exit")

        expected_final_equity = 100_000 + (17 - 11) * expected_size
        self.assertAlmostEqual(result["equity_curve"][-1]["equity"], expected_final_equity, places=2)

    def test_commission_charged_once_per_fill_when_enabled(self):
        _write_intraday_csv(self.intraday_dir / "TEST.csv", LIFECYCLE_ROWS)

        no_commission = run_smc_backtest(
            ["TEST"], 100_000, risk_pct=1.0, max_position_pct=10.0, intraday_dir=self.intraday_dir,
        )
        with_commission = run_smc_backtest(
            ["TEST"], 100_000, risk_pct=1.0, max_position_pct=10.0, intraday_dir=self.intraday_dir,
            commission_per_share=0.005, commission_min=1.0,
        )

        size = no_commission["trades"][0]["size"]
        expected_total_commission = 2 * portfolio.commission(size, 0.005, 1.0)  # one BUY fill, one SELL fill
        expected_equity = no_commission["equity_curve"][-1]["equity"] - expected_total_commission
        self.assertAlmostEqual(with_commission["equity_curve"][-1]["equity"], expected_equity, places=6)

    def test_fractional_shares_take_a_trade_a_tiny_account_would_otherwise_skip(self):
        _write_intraday_csv(self.intraday_dir / "TEST.csv", LIFECYCLE_ROWS)

        # $10 account, 1% risk, 10% cap, entry@11/stop@8 -> both risk-based
        # and cap-based sizing round to 0 whole shares, so the default
        # (whole-share) backtest takes no trade at all.
        whole_shares = run_smc_backtest(
            ["TEST"], 10, risk_pct=1.0, max_position_pct=10.0, intraday_dir=self.intraday_dir,
        )
        self.assertEqual(whole_shares["trades"], [])

        fractional = run_smc_backtest(
            ["TEST"], 10, risk_pct=1.0, max_position_pct=10.0, intraday_dir=self.intraday_dir,
            allow_fractional_shares=True,
        )
        self.assertEqual(len(fractional["trades"]), 2)
        buy, sell = fractional["trades"]
        expected_size = portfolio.position_size(10, 1.0, 11, 8, 10.0, allow_fractional=True)
        self.assertAlmostEqual(buy["size"], expected_size, places=5)
        self.assertAlmostEqual(sell["size"], expected_size, places=5)
        self.assertGreater(buy["size"], 0)
        self.assertLess(buy["size"], 1)  # confirms this really is sub-1-share sizing

    def test_fractional_fills_billed_via_ibkr_fractional_schedule_not_whole_share_one(self):
        """With allow_fractional_shares, commission must come from
        fractional_commission (1% of notional, $0.01 min) -- NOT
        commission() (the whole-share $/share + $1 min formula), which
        would be wildly wrong for a sub-1-share fill (e.g. $1 flat on a
        position worth a few cents)."""
        _write_intraday_csv(self.intraday_dir / "TEST.csv", LIFECYCLE_ROWS)

        no_commission = run_smc_backtest(
            ["TEST"], 10, risk_pct=1.0, max_position_pct=10.0, intraday_dir=self.intraday_dir,
            allow_fractional_shares=True,
        )
        with_commission = run_smc_backtest(
            ["TEST"], 10, risk_pct=1.0, max_position_pct=10.0, intraday_dir=self.intraday_dir,
            allow_fractional_shares=True, commission_per_share=0.005, commission_min=1.0,
        )

        size = no_commission["trades"][0]["size"]
        expected_total_commission = (
            portfolio.fractional_commission(size, 11)  # BUY fill @ entry price 11
            + portfolio.fractional_commission(size, 17)  # SELL fill @ exit price 17
        )
        expected_equity = no_commission["equity_curve"][-1]["equity"] - expected_total_commission
        self.assertAlmostEqual(with_commission["equity_curve"][-1]["equity"], expected_equity, places=6)


# Same fixture as test_smc_signals.py's test_stop_out_before_any_exit_signal:
# entry @ 11 (idx 8), stopped out @ 8 (idx 9) -- a clean, deterministic loss.
STOP_OUT_ROWS = [
    (10, 10, 9, 10), (10, 10, 9, 10), (12, 12, 11, 12), (10, 10, 9, 10), (10, 10, 9, 10),
    (11, 11, 8, 9), (9, 17, 12, 16), (16, 20, 16, 19),
    (19, 19, 10, 13),  # entry @ 11
    (13, 13, 5, 6),    # low(5) <= stop(8) -> stopped out @ 8
]


class TestReactiveDerisk(unittest.TestCase):
    """Two symbols (SYM_L1, SYM_L2) each take one losing trade and fully
    close well before a third (SYM_TEST) enters -- with
    reactive_derisk_window=2, the trailing profit factor over those two
    losses is 0 (all losses, no gains), below any threshold >= 0, so
    SYM_TEST's entry size should be scaled down."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.intraday_dir = Path(self._tmpdir.name)
        _write_intraday_csv(self.intraday_dir / "SYM_L1.csv", STOP_OUT_ROWS, start="2024-01-02 09:30:00")
        _write_intraday_csv(self.intraday_dir / "SYM_L2.csv", STOP_OUT_ROWS, start="2024-01-02 10:30:00")
        _write_intraday_csv(self.intraday_dir / "SYM_TEST.csv", STOP_OUT_ROWS, start="2024-01-02 12:00:00")

    def tearDown(self):
        self._tmpdir.cleanup()

    def _buy_size(self, result, symbol):
        buys = [t for t in result["trades"] if t["side"] == "BUY" and t["symbol"] == symbol]
        self.assertEqual(len(buys), 1)
        return buys[0]["size"]

    def test_size_scaled_down_after_trailing_losses_breach_threshold(self):
        """SYM_L1 and SYM_L2's own sizing is identical with the feature on
        or off (the window isn't full until right at SYM_TEST's entry), so
        the equity path up to SYM_TEST is the same in both runs -- only
        SYM_TEST's size should differ, by exactly the configured
        multiplier."""
        baseline = run_smc_backtest(
            ["SYM_L1", "SYM_L2", "SYM_TEST"], 100_000, risk_pct=1.0, max_position_pct=10.0,
            intraday_dir=self.intraday_dir,
        )
        derisked = run_smc_backtest(
            ["SYM_L1", "SYM_L2", "SYM_TEST"], 100_000, risk_pct=1.0, max_position_pct=10.0,
            intraday_dir=self.intraday_dir,
            reactive_derisk_window=2, reactive_derisk_pf_threshold=1.0, reactive_derisk_size_mult=0.5,
        )
        self.assertEqual(self._buy_size(derisked, "SYM_L1"), self._buy_size(baseline, "SYM_L1"))
        self.assertEqual(self._buy_size(derisked, "SYM_L2"), self._buy_size(baseline, "SYM_L2"))
        self.assertEqual(self._buy_size(derisked, "SYM_TEST"), int(self._buy_size(baseline, "SYM_TEST") * 0.5))

    def test_window_not_yet_full_leaves_size_unaffected(self):
        """Only SYM_L1 has closed by the time SYM_L2 enters -- with
        window=2, one closed trade isn't enough history yet, so SYM_L2's
        own size should match the feature-disabled baseline exactly."""
        baseline = run_smc_backtest(
            ["SYM_L1", "SYM_L2"], 100_000, risk_pct=1.0, max_position_pct=10.0, intraday_dir=self.intraday_dir,
        )
        derisked = run_smc_backtest(
            ["SYM_L1", "SYM_L2"], 100_000, risk_pct=1.0, max_position_pct=10.0,
            intraday_dir=self.intraday_dir,
            reactive_derisk_window=2, reactive_derisk_pf_threshold=1.0, reactive_derisk_size_mult=0.5,
        )
        self.assertEqual(self._buy_size(derisked, "SYM_L2"), self._buy_size(baseline, "SYM_L2"))


class TestRunSmcBacktestConcurrencyCap(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.intraday_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_only_one_symbol_enters_when_capped(self):
        # Both symbols have the IDENTICAL setup (same entry bar/date), so
        # with cap=1 only the alphabetically-first tie-break winner enters.
        _write_intraday_csv(self.intraday_dir / "AAA.csv", LIFECYCLE_ROWS)
        _write_intraday_csv(self.intraday_dir / "ZZZ.csv", LIFECYCLE_ROWS)

        result = run_smc_backtest(
            ["AAA", "ZZZ"], 100_000, max_concurrent_positions=1, intraday_dir=self.intraday_dir
        )

        buys = [t for t in result["trades"] if t["side"] == "BUY"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0]["symbol"], "AAA")


if __name__ == "__main__":
    unittest.main()
