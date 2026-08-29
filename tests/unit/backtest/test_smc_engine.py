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
from trading_bot.backtest.smc_engine import (
    build_smc_candidates,
    daily_watchlist_by_date,
    entry_window_mask,
    run_smc_backtest,
)

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


def _write_daily_csv(path: Path, closes: list[float], volumes: list[float], start="2024-01-02"):
    lines = ["Date,Open,High,Low,Close,Volume,Dividends,Stock Splits"]
    ts = pd.Timestamp(start)
    for close, volume in zip(closes, volumes):
        lines.append(f"{ts.strftime('%Y-%m-%d')} 00:00:00-05:00,{close},{close},{close},{close},{volume},0.0,0.0")
        ts += pd.Timedelta(days=1)
    path.write_text("\n".join(lines) + "\n")


class TestDailyWatchlistByDate(unittest.TestCase):
    """Rebuilds what cli/smc_prefilter.py would have written each morning:
    prior close above SMA200 and at/above min_price, ranked by 20-day
    average dollar volume, capped at max_watchlist_size."""

    SMA = 5
    DV = 3

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daily_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _build(self, **kwargs):
        return daily_watchlist_by_date(
            sorted(p.stem for p in self.daily_dir.glob("*.csv")),
            daily_dir=self.daily_dir, sma_window=self.SMA, dollar_volume_window=self.DV, **kwargs,
        )

    def test_only_names_above_their_sma_qualify(self):
        _write_daily_csv(self.daily_dir / "UP.csv", [10, 11, 12, 13, 14, 15, 16], [100] * 7)
        _write_daily_csv(self.daily_dir / "DOWN.csv", [16, 15, 14, 13, 12, 11, 10], [100] * 7)
        watchlist = self._build()
        qualifying = {d: names for d, names in watchlist.items() if names}
        self.assertTrue(qualifying)
        for names in qualifying.values():
            self.assertIn("UP", names)
            self.assertNotIn("DOWN", names)

    def test_ranked_by_dollar_volume_and_capped(self):
        for name, volume in (("THIN", 10), ("MID", 100), ("THICK", 1000)):
            _write_daily_csv(self.daily_dir / f"{name}.csv", [10, 11, 12, 13, 14, 15, 16], [volume] * 7)
        watchlist = self._build(max_size=2)
        picked = [names for names in watchlist.values() if names]
        self.assertTrue(picked)
        for names in picked:
            self.assertEqual(names, {"THICK", "MID"})

    def test_min_price_screens_on_the_prior_close(self):
        _write_daily_csv(self.daily_dir / "PENNY.csv", [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], [100] * 7)
        self.assertTrue(any(self._build(min_price=0.0).values()))
        self.assertFalse(any(self._build(min_price=5.0).values()))

    def test_a_date_is_decided_before_it_opens(self):
        """Anchored to prior-day data, like the 09:40 ET screen. A name
        that only crosses its SMA on the last day must not appear on that
        day's list -- that would be trading on a close not yet printed."""
        _write_daily_csv(self.daily_dir / "LATE.csv", [10, 9, 8, 7, 6, 5, 99], [100] * 7)
        for names in self._build().values():
            self.assertNotIn("LATE", names)

    def test_dates_with_no_qualifier_are_present_but_empty(self):
        """So a caller can tell an empty universe from a date outside the
        data -- the difference between "no trades allowed" and "no data"."""
        _write_daily_csv(self.daily_dir / "DOWN.csv", [16, 15, 14, 13, 12, 11, 10], [100] * 7)
        watchlist = self._build()
        self.assertTrue(watchlist)
        self.assertTrue(all(names == set() for names in watchlist.values()))


class TestEntryWindowMask(unittest.TestCase):
    def _dates(self, times: list[str]):
        return pd.Series(pd.to_datetime([f"2024-01-02 {t}:00-05:00" for t in times], utc=True))

    def test_bounds_are_inclusive(self):
        dates = self._dates(["09:35", "10:05", "12:00", "15:30", "15:45"])
        self.assertEqual(
            entry_window_mask(dates, "10:05", "15:30"),
            [False, True, True, True, False],
        )

    def test_open_ended_bounds(self):
        dates = self._dates(["09:35", "12:00", "15:45"])
        self.assertEqual(entry_window_mask(dates, None, "15:30"), [True, True, False])
        self.assertEqual(entry_window_mask(dates, "10:05", None), [False, True, True])
        self.assertEqual(entry_window_mask(dates, None, None), [True, True, True])

    def test_compares_in_et_not_utc(self):
        """Cached bars are stored in UTC; 14:00 UTC is 09:00 ET, before
        the window, and reading the raw hour would let it through."""
        dates = pd.Series(pd.to_datetime(["2024-01-02 14:00:00+00:00"], utc=True))
        self.assertEqual(entry_window_mask(dates, "10:05", "15:30"), [False])


class TestBuildCandidatesUniverse(unittest.TestCase):
    """The watchlist gate is on the ENTRY date, matching the live bot,
    which manages an open position whether or not the symbol is still on
    today's list."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.intraday_dir = Path(self._tmpdir.name)
        _write_intraday_csv(self.intraday_dir / "AAA.csv", LIFECYCLE_ROWS)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _candidates(self, **kwargs):
        return build_smc_candidates(["AAA"], intraday_dir=self.intraday_dir, **kwargs)

    def test_no_watchlist_keeps_everything(self):
        self.assertEqual(len(self._candidates()), 1)

    def test_symbol_on_that_days_list_is_kept(self):
        entry_day = self._candidates()[0][0].date()
        self.assertEqual(len(self._candidates(daily_watchlist={entry_day: {"AAA"}})), 1)

    def test_symbol_off_that_days_list_is_dropped(self):
        entry_day = self._candidates()[0][0].date()
        self.assertEqual(self._candidates(daily_watchlist={entry_day: {"BBB"}}), [])

    def test_a_date_missing_from_the_watchlist_trades_nothing(self):
        self.assertEqual(self._candidates(daily_watchlist={}), [])
