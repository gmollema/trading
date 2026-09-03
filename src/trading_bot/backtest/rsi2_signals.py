"""Per-symbol signal generation for the Connors/Alvarez 2-period RSI mean
reversion strategy, plus the "first profitable close with a day delay"
exit modification demonstrated in the YouTube video "New Improved! 2
Period RSI Trading Strategy" (gvzCDqjccLs, Aug 2021).

Rules, transcribed from the video with the timestamp each one comes from:

  Instrument / timeframe  S&P 500, daily bars. The video tests OANDA's
      SPX500 CFD and reports results in INDEX POINTS, not dollars
      ("the results you'll see are going to be in points rather than
      dollars", 0:55). This module likewise returns point P&L and models
      no position sizing or costs -- see cli/rsi2_backtest.py.
  Direction (1:33)        Long only.
  Trend filter (1:56)     The signal bar's close must be above its own
      200-period SMA. Simple vs exponential "doesn't make much
      difference"; simple is what the video uses.
  Entry signal (2:41)     RSI(2) crosses down below 10. Connors documents
      5, which trades less often; the video uses 10.
  Entry fill (2:52)       "then we buy on the next open" -- the bar AFTER
      the signal bar, at its open.
  Stop (3:04)             A flat 200 index points below the entry. Connors
      himself documents no stop (they hurt mean-reversion performance);
      the 200 is the video author's own addition.
  Baseline exit (3:50)    RSI(2) closes above 70, exiting on the next bar.
  Modified exit (4:34)    Larry Williams' "bailout" / first profitable
      close: at each daily close, if the trade is in profit AT ALL, exit;
      otherwise hold, and let the stop or a later profitable close end it.
  Day delay (8:51)        The video's actual contribution. Do not start
      applying the first-profitable-close test until the trade has been
      open for `min_hold_days` bars. `min_hold_days=1` means "no delay
      whatsoever" (9:36), i.e. the entry bar's own close already counts.
      Sweeping 1..20 on 2008-2019 data, the video picks 12.
  Position count (6:32)   One at a time. The video explicitly shows RSI
      dipping below 10 again mid-trade and says "we stayed in the
      original trade" -- in-trade signals are ignored, not stacked.

Two things the video leaves genuinely underspecified, both exposed here as
parameters rather than guessed at silently:

  exit_timing  The baseline RSI exit is stated as next-bar ("we're
      exiting on the next bar", 6:09), but the first-profitable-close
      exit is described only as looking at the close and getting out,
      which in a bar-close backtest can mean either filling at that
      close or at the next open. Williams' bailout is classically
      at-the-close, so "close" is the default; "next_open" is the
      conservative reading, so cli/rsi2_backtest.py reports both.
  SMA alignment  `sma[i]` here includes bar i's own close, because the
      video's filter is a statement about the signal bar itself and the
      fill only happens on the following open -- so no lookahead is
      involved. Note this differs from backtest/data.compute_daily_context,
      whose SMA is deliberately shifted a day to mirror what the live
      gap-and-go bot can see mid-session.
"""

from __future__ import annotations

DEFAULT_RSI_PERIOD = 2
DEFAULT_ENTRY_LEVEL = 10.0
DEFAULT_EXIT_LEVEL = 70.0
DEFAULT_SMA_PERIOD = 200
DEFAULT_STOP_POINTS = 200.0
# The video's optimized value; 1 = the undelayed first-profitable-close.
DEFAULT_MIN_HOLD_DAYS = 12

EXIT_MODE_RSI = "rsi"
EXIT_MODE_FIRST_PROFITABLE_CLOSE = "first_profitable_close"


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def wilder_rsi(closes: list[float], period: int = DEFAULT_RSI_PERIOD) -> list[float | None]:
    """Wilder's RSI, the standard definition used by TradeStation's RSI and
    by Connors' own work -- an RMA (Wilder-smoothed average) of gains and
    losses, NOT a simple mean of them.

    Seeds at bar `period` with the simple mean of the first `period` price
    changes, then recurses avg[i] = (avg[i-1]*(period-1) + x[i]) / period.
    Bars 0..period-1 report None: RSI genuinely does not exist yet there,
    and returning None rather than a placeholder is what stops those bars
    from manufacturing a spurious "crossed below 10" on the first real
    value.

    Degenerate averages follow the usual convention: no losses in the
    window is RSI 100, no gains is RSI 0, and a completely flat window
    (neither gains nor losses) is 50.
    """
    n = len(closes)
    if n <= period:
        return [None] * n

    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)

    out: list[float | None] = [None] * n
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    out[period] = _rsi_from_averages(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i] = _rsi_from_averages(avg_gain, avg_loss)
    return out


def simple_moving_average(values: list[float], period: int) -> list[float | None]:
    """Trailing SMA INCLUDING the current bar; None for the first
    period-1 bars. See the module docstring on SMA alignment."""
    n = len(values)
    out: list[float | None] = [None] * n
    window_sum = 0.0
    for i in range(n):
        window_sum += values[i]
        if i >= period:
            window_sum -= values[i - period]
        if i >= period - 1:
            out[i] = window_sum / period
    return out


def rsi2_entry_signals(
    closes: list[float],
    rsi_period: int = DEFAULT_RSI_PERIOD,
    entry_level: float = DEFAULT_ENTRY_LEVEL,
    sma_period: int = DEFAULT_SMA_PERIOD,
) -> list[bool]:
    """True on each bar whose CLOSE completes an entry setup: RSI(period)
    crossed down through `entry_level` on this bar while the close sits
    above its `sma_period` SMA. The fill belongs on the next bar's open.

    "Crosses down below" is a strict crossing (prior >= level, current <
    level), not merely "is below" -- a run of consecutive sub-10 closes
    produces one signal, on the first of them, which is what makes the
    video's trade count (76 baseline trades over 12 years) reachable at
    all.
    """
    rsi = wilder_rsi(closes, rsi_period)
    sma = simple_moving_average(closes, sma_period)
    return [
        i > 0
        and rsi[i] is not None
        and rsi[i - 1] is not None
        and sma[i] is not None
        and closes[i] > sma[i]
        and rsi[i - 1] >= entry_level
        and rsi[i] < entry_level
        for i in range(len(closes))
    ]


def find_rsi2_long_trades(
    bars: dict,
    rsi_period: int = DEFAULT_RSI_PERIOD,
    entry_level: float = DEFAULT_ENTRY_LEVEL,
    exit_level: float = DEFAULT_EXIT_LEVEL,
    sma_period: int = DEFAULT_SMA_PERIOD,
    stop_points: float | None = DEFAULT_STOP_POINTS,
    stop_pct: float | None = None,
    exit_mode: str = EXIT_MODE_FIRST_PROFITABLE_CLOSE,
    min_hold_days: int = DEFAULT_MIN_HOLD_DAYS,
    exit_timing: str = "close",
) -> list[dict]:
    """Long-only walk over one symbol's daily bars. One position at a time.

    Args:
        bars: {"date","open","high","low","close"} -- equal-length lists,
            chronologically sorted.
        stop_points: flat point distance below entry, the video's 200.
            None disables the stop entirely (Connors' documented original).
        stop_pct: alternative percentage stop, e.g. 3.0 for 3% below
            entry. Mutually exclusive with stop_points. Not in the video,
            but the flat 200-point stop means something wildly different
            at SPX 750 (2009) than at SPX 7600 (2026) -- 27% vs 2.6% --
            so a percentage stop is the only way to hold the risk taken
            per trade roughly constant across a multi-decade test.
        exit_mode: EXIT_MODE_RSI for the Connors baseline (out when RSI
            closes above `exit_level`), or
            EXIT_MODE_FIRST_PROFITABLE_CLOSE for the video's modification.
        min_hold_days: first-profitable-close mode only. Bars the trade
            must have been open, counting the entry bar as day 1, before
            the profit test starts applying. 1 = no delay.
        exit_timing: "close" fills a signalled exit at that bar's close,
            "next_open" at the following bar's open. Stops always fill
            intrabar regardless.

    Returns:
        list of {"entry_idx","entry_date","entry_price","stop_price",
        "exit_idx","exit_date","exit_price","bars_held","points",
        "mae_points","reason"} -- one dict per completed round trip.
        `points` is the raw per-unit point move, exit minus entry. A
        position still open at the end of the data exits at the last
        bar's close with reason "end_of_data".

        `mae_points` is maximum adverse excursion: how far below the
        entry the position ever traded while open, as a positive number
        (0 if it never went underwater). This is the only honest risk
        figure this strategy produces. Its win rate is not one: a
        first-profitable-close exit can only ever CLOSE a winner, so with
        the stop disabled the win rate is 100% by construction and the
        entire risk of the strategy sits in this column instead.
    """
    if stop_points is not None and stop_pct is not None:
        raise ValueError("pass stop_points or stop_pct, not both")
    if exit_mode not in (EXIT_MODE_RSI, EXIT_MODE_FIRST_PROFITABLE_CLOSE):
        raise ValueError(f"unknown exit_mode {exit_mode!r}")
    if exit_timing not in ("close", "next_open"):
        raise ValueError(f"unknown exit_timing {exit_timing!r}")

    dates, opens = bars["date"], bars["open"]
    lows, closes = bars["low"], bars["close"]
    n = len(closes)
    if n == 0:
        return []

    rsi = wilder_rsi(closes, rsi_period)
    signals = rsi2_entry_signals(closes, rsi_period, entry_level, sma_period)

    trades: list[dict] = []
    open_trade: dict | None = None
    pending_entry = False
    pending_exit: str | None = None

    def _close_trade(i: int, exit_price: float, reason: str, filled_at_open: bool = False) -> None:
        entry_idx = open_trade["entry_idx"]
        # The entry bar's own low counts (we were long from its open), but
        # an exit filled at bar i's OPEN means bar i's low happened after
        # we were already out, so it must not.
        last_held = i - 1 if (filled_at_open and i > entry_idx) else i
        worst_low = min(lows[entry_idx : last_held + 1])
        trades.append({
            **open_trade,
            "exit_idx": i,
            "exit_date": dates[i],
            "exit_price": exit_price,
            "bars_held": i - entry_idx + 1,
            "points": exit_price - open_trade["entry_price"],
            "mae_points": max(0.0, open_trade["entry_price"] - worst_low),
            "reason": reason,
        })

    for i in range(n):
        # 1. An exit signalled on the previous bar's close fills here, at
        # the open. If the market gapped straight through the stop the
        # resting stop is what fills, at that same open price -- so only
        # the recorded reason differs.
        if open_trade is not None and pending_exit is not None:
            stop_price = open_trade["stop_price"]
            gapped_through = stop_price is not None and opens[i] <= stop_price
            _close_trade(i, opens[i], "stop_loss" if gapped_through else pending_exit, filled_at_open=True)
            open_trade = None
            pending_exit = None

        # 2. An entry signalled on the previous bar's close fills at this
        # bar's open. Must precede the exit checks below: with
        # min_hold_days=1 a trade can legitimately open and close on the
        # same bar.
        if open_trade is None and pending_entry:
            entry_price = opens[i]
            if stop_points is not None:
                stop_price = entry_price - stop_points
            elif stop_pct is not None:
                stop_price = entry_price * (1 - stop_pct / 100.0)
            else:
                stop_price = None
            open_trade = {
                "entry_idx": i,
                "entry_date": dates[i],
                "entry_price": entry_price,
                "stop_price": stop_price,
            }
        pending_entry = False

        # 3. Manage an open position. The stop is a resting order and so
        # fills intrabar, ahead of any close-based exit rule -- if both
        # the stop and the exit condition are met on one bar, the stop is
        # the honest fill, since we cannot know from a daily bar that the
        # low came after the close.
        if open_trade is not None:
            stop_price = open_trade["stop_price"]
            if stop_price is not None and lows[i] <= stop_price:
                # A gap-down open below the stop fills at the open, not
                # at the stop price -- there is no liquidity in between.
                _close_trade(i, min(opens[i], stop_price), "stop_loss")
                open_trade = None
            else:
                if exit_mode == EXIT_MODE_RSI:
                    should_exit = rsi[i] is not None and rsi[i] > exit_level
                    reason = "rsi_exit"
                else:
                    days_held = i - open_trade["entry_idx"] + 1
                    should_exit = days_held >= min_hold_days and closes[i] > open_trade["entry_price"]
                    reason = "first_profitable_close"
                if should_exit:
                    if exit_timing == "close":
                        _close_trade(i, closes[i], reason)
                        open_trade = None
                    else:
                        pending_exit = reason

        # 4. Only a flat book takes a new signal (see "Position count" in
        # the module docstring). A trade closed at this bar's close does
        # leave the book flat, so its own bar's signal is eligible.
        if open_trade is None and signals[i]:
            pending_entry = True

    if open_trade is not None:
        _close_trade(n - 1, closes[n - 1], "end_of_data")

    return trades
