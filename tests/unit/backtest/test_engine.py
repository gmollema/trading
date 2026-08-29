"""End-to-end unit tests for trading_bot.backtest.engine.run_backtest.

Builds a tiny synthetic ticker fixture (200+ days of flat daily history so
SMA200 is defined, plus a handful of 5-min bars on the "entry day") that is
crafted to produce exactly one clean entry signal followed by a stop-out,
and verifies the resulting trades + equity curve. A second scenario checks
that max_concurrent_positions actually caps concurrent entries, breaking
ties by gap size (largest first).

All bar timestamps are placed at/after 10:05 ET (rules.json's
earliest_entry_et) -- bars before 10:00 ET are "too_early" per
cycle.get_market_status and would never be considered for entry regardless
of what the filters say.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_bot.backtest import engine
from trading_bot.backtest.engine import run_backtest

DAILY_HISTORY_LEN = 200  # rows needed for a defined SMA200


def make_rules(max_concurrent_positions=5):
    return {
        # The engine shares cycle.get_market_status with the live bot, so
        # it needs the same time_filter the live bot reads. These are the
        # shipped values; before 2026-08-29 they were hardcoded inside the
        # gate and this block would not have been needed -- which was the
        # bug, not the convenience.
        "time_filter": {
            "earliest_entry_et": "10:05",
            "latest_entry_et": "15:30",
            "force_close_et": "15:51",
        },
        "daily_filters": {
            "D1_above_prior_day_high": True,
            "D2_prior_close_above_sma200": True,
            "D3_min_gap_pct_from_prior_close": 3.0,
        },
        "intraday_filters": {
            "I2_above_today_hod": True,
            "I3_rvol_min": 2.0,
            "I3_rvol_lookback_days": 14,
        },
        "exit": {
            "partial_profit_trigger_R": 0.75,
            "partial_profit_fraction": 1 / 3,
            "breakeven_trigger_R": 1.0,
        },
        "risk": {
            "max_risk_per_trade_pct": 1.0,
            "max_position_size_pct_of_portfolio": 10,
            "max_concurrent_positions": max_concurrent_positions,
        },
    }


def _daily_dates(n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d 00:00:00-05:00") for d in pd.date_range("2024-01-01", periods=n, freq="D")]


def _write_daily_csv(path: Path, dates: list[str], closes: list[float]):
    lines = ["Date,Open,High,Low,Close,Volume,Dividends,Stock Splits"]
    for d, c in zip(dates, closes):
        lines.append(f"{d},{c},{c + 1},{c - 1},{c},1000,0.0,0.0")
    path.write_text("\n".join(lines) + "\n")


def _write_intraday_csv(path: Path, rows: list[tuple[str, float, float, float, float, int]]):
    lines = ["date,open,high,low,close,volume"]
    for ts, o, h, low, c, v in rows:
        lines.append(f"{ts},{o},{h},{low},{c},{v}")
    path.write_text("\n".join(lines) + "\n")


def _build_ticker_fixture(daily_dir: Path, intraday_dir: Path, ticker: str, breakout_close: float):
    """Write a daily + intraday CSV pair for `ticker` such that:
      - SMA200/prior-day context is defined and D1/D2 pass.
      - A 10:05 bar has rvol=1.0 (fails I3, no entry).
      - A 10:10 bar breaks out to `breakout_close` with rvol=2.0 and its own
        high == close (so I2's "at/above today's HOD" holds), triggering
        entry provided `breakout_close` clears the D3 gap threshold.
      - A 10:15 bar's low breaches the LOD-1% initial stop, closing the
        position via a stop-out.

    Returns the initial_stop price the entry should compute (so callers can
    assert the exact stop fill price/size without recomputing it inline).
    """
    n = DAILY_HISTORY_LEN + 1  # +1 "today" placeholder row for the merge key
    closes = [50.0 + i * 0.01 for i in range(DAILY_HISTORY_LEN)] + [52.0]
    dates = _daily_dates(n)
    _write_daily_csv(daily_dir / f"{ticker}.csv", dates, closes)

    prior_day = dates[DAILY_HISTORY_LEN - 1][:10]
    entry_day = dates[DAILY_HISTORY_LEN][:10]

    today_lod = 49.5  # min low across the 10:05/10:10 bars below
    initial_stop = today_lod * 0.99

    _write_intraday_csv(
        intraday_dir / f"{ticker}.csv",
        [
            (f"{prior_day} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
            (f"{prior_day} 10:10:00-04:00", 50, 50.5, 49.5, 50.0, 100),
            (f"{entry_day} 10:05:00-04:00", 50, 50.5, today_lod, 50.0, 100),
            (f"{entry_day} 10:10:00-04:00", 50, breakout_close, 49.8, breakout_close, 300),
            (f"{entry_day} 10:15:00-04:00", breakout_close, breakout_close + 0.2, 48.0, 49.0, 100),
        ],
    )
    return initial_stop


def _build_open_ended_fixture(daily_dir: Path, intraday_dir: Path, ticker: str, breakout_close: float, last_close: float):
    """Like _build_ticker_fixture, but the data simply ENDS shortly after
    entry (no stop, no force-close, no partial trigger) -- for exercising
    run_backtest's end-of-data safety net, which force-closes any position
    still open when a symbol's cached history runs out."""
    n = DAILY_HISTORY_LEN + 1
    closes = [50.0 + i * 0.01 for i in range(DAILY_HISTORY_LEN)] + [52.0]
    dates = _daily_dates(n)
    _write_daily_csv(daily_dir / f"{ticker}.csv", dates, closes)

    prior_day = dates[DAILY_HISTORY_LEN - 1][:10]
    entry_day = dates[DAILY_HISTORY_LEN][:10]

    _write_intraday_csv(
        intraday_dir / f"{ticker}.csv",
        [
            (f"{prior_day} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
            (f"{prior_day} 10:10:00-04:00", 50, 50.5, 49.5, 50.0, 100),
            (f"{entry_day} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
            (f"{entry_day} 10:10:00-04:00", 50, breakout_close, 49.8, breakout_close, 300),
            # Last bar in the whole dataset: a modest move that neither
            # breaches the stop nor reaches the 0.75R partial trigger, so
            # the position is still open when the data simply runs out.
            (f"{entry_day} 10:15:00-04:00", breakout_close, last_close + 0.1, breakout_close - 0.2, last_close, 50),
        ],
    )


class TestRunBacktestEndOfDataSafetyNet(unittest.TestCase):
    """Regression test: run_backtest's end-of-data safety net must update
    equity_curve too, not just `trades` -- equity_curve is built inside the
    main day-by-day loop, but the safety net that force-closes anything
    still open runs in a SEPARATE loop afterward, so it must append its own
    equity_curve entries or the reported final equity silently omits any
    P&L realized only by that safety net (which happens whenever
    force_close_daily=False leaves positions open past the cached data)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daily_dir = Path(self._tmpdir.name) / "daily"
        self.intraday_dir = Path(self._tmpdir.name) / "intraday_5m"
        self.daily_dir.mkdir()
        self.intraday_dir.mkdir()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_equity_curve_reflects_safety_net_close(self):
        _build_open_ended_fixture(
            self.daily_dir, self.intraday_dir, "TEST", breakout_close=53.8, last_close=54.5
        )

        result = run_backtest(
            ["TEST"],
            make_rules(),
            100_000,
            daily_dir=self.daily_dir,
            intraday_dir=self.intraday_dir,
            force_close_daily=False,
        )
        trades = result["trades"]

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0]["side"], "BUY")
        sell = trades[1]
        self.assertEqual(sell["side"], "SELL")
        self.assertEqual(sell["reason"], "end_of_data")

        expected_pnl = (sell["fill_price"] - trades[0]["fill_price"]) * sell["size"]
        expected_final_equity = 100_000 + expected_pnl

        self.assertAlmostEqual(result["equity_curve"][-1]["equity"], expected_final_equity, places=2)
        # The safety-net close must be a real, nonzero move -- otherwise
        # this test can't distinguish "bug fixed" from "nothing to fix".
        self.assertNotAlmostEqual(expected_pnl, 0.0, places=2)

    def test_overnight_snapshot_recorded_when_position_held_past_eod(self):
        n = DAILY_HISTORY_LEN + 1
        closes = [50.0 + i * 0.01 for i in range(DAILY_HISTORY_LEN)] + [52.0]
        dates = _daily_dates(n)
        _write_daily_csv(self.daily_dir / "TEST.csv", dates, closes)

        prior_day = dates[DAILY_HISTORY_LEN - 1][:10]
        day1 = dates[DAILY_HISTORY_LEN][:10]
        day2 = (pd.Timestamp(day1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        _write_intraday_csv(
            self.intraday_dir / "TEST.csv",
            [
                (f"{prior_day} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{prior_day} 10:10:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{day1} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{day1} 10:10:00-04:00", 50, 53.8, 49.8, 53.8, 300),  # entry
                (f"{day1} 10:15:00-04:00", 53.8, 54.0, 53.5, 53.9, 50),  # no trigger, position stays open
                (f"{day2} 10:05:00-04:00", 53.9, 54.2, 53.6, 54.0, 100),  # next day: should be snapshotted
            ],
        )

        result = run_backtest(
            ["TEST"],
            make_rules(),
            100_000,
            daily_dir=self.daily_dir,
            intraday_dir=self.intraday_dir,
            force_close_daily=False,
        )

        snapshots = result["overnight_stop_snapshots"]
        self.assertEqual(len(snapshots), 1)
        snap = snapshots[0]
        self.assertEqual(snap["symbol"], "TEST")
        self.assertEqual(snap["date"], day2)
        self.assertEqual(snap["entry_price"], 53.8)
        self.assertAlmostEqual(snap["stop_price"], 49.5 * 0.99)  # unchanged initial LOD-1% stop

    def test_overnight_size_reduction_sells_a_fraction_at_prior_close(self):
        n = DAILY_HISTORY_LEN + 1
        closes = [50.0 + i * 0.01 for i in range(DAILY_HISTORY_LEN)] + [52.0]
        dates = _daily_dates(n)
        _write_daily_csv(self.daily_dir / "TEST.csv", dates, closes)

        prior_day = dates[DAILY_HISTORY_LEN - 1][:10]
        day1 = dates[DAILY_HISTORY_LEN][:10]
        day2 = (pd.Timestamp(day1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        _write_intraday_csv(
            self.intraday_dir / "TEST.csv",
            [
                (f"{prior_day} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{prior_day} 10:10:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{day1} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{day1} 10:10:00-04:00", 50, 53.8, 49.8, 53.8, 300),  # entry, size=185
                (f"{day1} 10:15:00-04:00", 53.8, 54.0, 53.5, 53.9, 50),  # last bar of day1 -- reduction priced here
                (f"{day2} 10:05:00-04:00", 53.9, 54.2, 53.6, 54.0, 100),
            ],
        )

        result = run_backtest(
            ["TEST"],
            make_rules(),
            100_000,
            daily_dir=self.daily_dir,
            intraday_dir=self.intraday_dir,
            force_close_daily=False,
            overnight_size_reduction_pct=0.5,
        )
        trades = result["trades"]

        # BUY, the overnight size reduction, then the end-of-data safety
        # net closing the remaining 93 shares (this fixture's data simply
        # ends on day2 with nothing to trigger a normal exit).
        self.assertEqual(len(trades), 3)
        buy, reduction, final_close = trades
        self.assertEqual(buy["side"], "BUY")
        self.assertEqual(buy["size"], 185)

        self.assertEqual(reduction["side"], "SELL")
        self.assertEqual(reduction["reason"], "overnight_size_reduction")
        self.assertEqual(reduction["size"], 92)  # floor(185 * 0.5)
        self.assertEqual(reduction["fill_price"], 53.9)  # day1's last bar close, not day2's

        self.assertEqual(final_close["size"], 93)  # 185 - 92, closed via end_of_data

        # The overnight snapshot must reflect the qty AFTER reduction.
        snap = result["overnight_stop_snapshots"][0]
        self.assertEqual(snap["qty"], 93)  # 185 - 92

    def test_max_hold_days_force_closes_on_first_bar_past_the_cap(self):
        n = DAILY_HISTORY_LEN + 1
        closes = [50.0 + i * 0.01 for i in range(DAILY_HISTORY_LEN)] + [52.0]
        dates = _daily_dates(n)
        _write_daily_csv(self.daily_dir / "TEST.csv", dates, closes)

        prior_day = dates[DAILY_HISTORY_LEN - 1][:10]
        day1 = dates[DAILY_HISTORY_LEN][:10]
        day2 = (pd.Timestamp(day1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        _write_intraday_csv(
            self.intraday_dir / "TEST.csv",
            [
                (f"{prior_day} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{prior_day} 10:10:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{day1} 10:05:00-04:00", 50, 50.5, 49.5, 50.0, 100),
                (f"{day1} 10:10:00-04:00", 50, 53.8, 49.8, 53.8, 300),  # entry
                (f"{day1} 10:15:00-04:00", 53.8, 54.0, 53.5, 53.9, 50),  # no trigger, position stays open
                (f"{day2} 10:05:00-04:00", 53.9, 54.2, 53.6, 54.1, 100),  # 1 day later -> should force-close here
            ],
        )

        result = run_backtest(
            ["TEST"],
            make_rules(),
            100_000,
            daily_dir=self.daily_dir,
            intraday_dir=self.intraday_dir,
            force_close_daily=False,
            max_hold_days=1,
        )
        trades = result["trades"]

        self.assertEqual(len(trades), 2)
        sell = trades[1]
        self.assertEqual(sell["reason"], "max_hold_reached")
        self.assertEqual(sell["fill_price"], 54.1)  # closed at the day2 10:05 bar's close
        # No overnight snapshot should remain unresolved past the cap either.
        self.assertEqual(result["overnight_stop_snapshots"], [
            {"date": day2, "symbol": "TEST", "stop_price": 49.5 * 0.99, "entry_price": 53.8, "qty": 185}
        ])


class TestRunBacktestSingleTrade(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daily_dir = Path(self._tmpdir.name) / "daily"
        self.intraday_dir = Path(self._tmpdir.name) / "intraday_5m"
        self.daily_dir.mkdir()
        self.intraday_dir.mkdir()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_no_cached_data_returns_empty_result(self):
        result = run_backtest(
            ["NOPE"], make_rules(), 100_000, daily_dir=self.daily_dir, intraday_dir=self.intraday_dir
        )

        self.assertEqual(result, {"trades": [], "equity_curve": [], "overnight_stop_snapshots": []})

    def test_entry_then_stop_out_produces_expected_trades(self):
        initial_stop = _build_ticker_fixture(self.daily_dir, self.intraday_dir, "TEST", breakout_close=53.8)

        result = run_backtest(
            ["TEST"], make_rules(), 100_000, daily_dir=self.daily_dir, intraday_dir=self.intraday_dir
        )
        trades = result["trades"]

        self.assertEqual(len(trades), 2)

        buy = trades[0]
        self.assertEqual(buy["symbol"], "TEST")
        self.assertEqual(buy["side"], "BUY")
        self.assertEqual(buy["fill_price"], 53.8)

        r_per_share = 53.8 - initial_stop
        expected_size = min(
            int(100_000 * 0.01 / r_per_share),  # size_by_risk (floor via int-division on positive floats)
            int(100_000 * 0.10 / 53.8),  # size_by_cap
        )
        self.assertEqual(buy["size"], expected_size)

        sell = trades[1]
        self.assertEqual(sell["symbol"], "TEST")
        self.assertEqual(sell["side"], "SELL")
        self.assertAlmostEqual(sell["fill_price"], round(initial_stop, 4))
        self.assertEqual(sell["size"], expected_size)

        expected_final_equity = 100_000 + (initial_stop - 53.8) * expected_size
        self.assertAlmostEqual(result["equity_curve"][-1]["equity"], expected_final_equity, places=2)

    def test_gap_below_threshold_never_enters(self):
        # prior_day_close ~= 51.99 -> a 1% "breakout" close of 52.51 is far
        # short of the 3% D3 gap threshold, so no trade should ever open.
        _build_ticker_fixture(self.daily_dir, self.intraday_dir, "TEST", breakout_close=52.51)

        result = run_backtest(
            ["TEST"], make_rules(), 100_000, daily_dir=self.daily_dir, intraday_dir=self.intraday_dir
        )

        self.assertEqual(result["trades"], [])


class TestRunBacktestConcurrencyCap(unittest.TestCase):
    """Two symbols both signal entry on the same tick; max_concurrent_positions=1
    must admit only one -- the larger gap, per the engine's tie-break order."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daily_dir = Path(self._tmpdir.name) / "daily"
        self.intraday_dir = Path(self._tmpdir.name) / "intraday_5m"
        self.daily_dir.mkdir()
        self.intraday_dir.mkdir()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_only_larger_gap_symbol_enters_when_capped(self):
        _build_ticker_fixture(self.daily_dir, self.intraday_dir, "SMALLGAP", breakout_close=53.8)
        _build_ticker_fixture(self.daily_dir, self.intraday_dir, "BIGGAP", breakout_close=55.0)

        result = run_backtest(
            ["SMALLGAP", "BIGGAP"],
            make_rules(max_concurrent_positions=1),
            100_000,
            daily_dir=self.daily_dir,
            intraday_dir=self.intraday_dir,
        )

        buys = [t for t in result["trades"] if t["side"] == "BUY"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0]["symbol"], "BIGGAP")


if __name__ == "__main__":
    unittest.main()


class TestSlippageNormalisation(unittest.TestCase):
    def test_none_is_frictionless(self):
        self.assertEqual(set(engine._normalize_slippage(None).values()), {0.0})

    def test_scalar_applies_to_every_leg(self):
        self.assertEqual(set(engine._normalize_slippage(5.0).values()), {5.0})

    def test_partial_dict_leaves_the_rest_at_zero(self):
        slip = engine._normalize_slippage({"entry": 10.0})
        self.assertEqual(slip["entry"], 10.0)
        self.assertEqual(slip["stop"], 0.0)

    def test_unknown_reason_is_rejected(self):
        """A typo would otherwise read as "no slippage on that leg"."""
        with self.assertRaises(ValueError):
            engine._normalize_slippage({"tp1": 10.0})  # an SMC reason, not a gap-and-go one
