"""Per-symbol signal generation for the "UT Bot Alerts" ATR trailing-stop
system, translated from the user-supplied Pine Script v5 indicator
("GM - UT Bot Alerts with LinReg Candles").

The LinReg Candles half of that script is COSMETIC ONLY (a smoothed
candle overlay + a plotted signal line) -- nothing in the script's own
`buy`/`sell` alert conditions references it, so it is intentionally not
implemented here. This module reproduces exactly the part that IS the
tradeable signal: the ATR trailing-stop line and its crossover/crossunder
with price.

Pine -> Python mapping (verified line by line against the source):
  xATR   = ta.atr(c)                         -- Wilder's ATR (RMA of true
                                                 range), NOT a simple
                                                 average of true range.
  nLoss  = a * xATR
  xATRTrailingStop:  a bar-by-bar recursive trailing stop --
      if src > prevStop and prevSrc > prevStop: max(prevStop, src-nLoss)
      elif src < prevStop and prevSrc < prevStop: min(prevStop, src+nLoss)
      elif src > prevStop: src - nLoss   (trend flips up)
      else: src + nLoss                  (trend flips down or first bar)
    `nz(x, 0)` (Pine's null-coalesce) means every comparison against the
    not-yet-existing previous bar uses 0, which is what makes the very
    first bar's stop resolve via the "trend flips up" branch whenever
    price is positive (always true for an FX rate or a stock price).
  emaVal = ta.ema(src, 1)  -- EMA of period 1 is mathematically identical
    to its own input (alpha = 2/(1+1) = 1), so this is just `src` again;
    it exists in the original script only so ta.crossover has a "series"
    argument, not because it does any smoothing.
  buy  = src > stop and crossover(src, stop)   == crossover(src, stop)
  sell = src < stop and crossunder(src, stop)  == crossunder(src, stop)
    (the extra comparisons are redundant with what crossover/crossunder
    already require, kept here only as a comment for traceability.)

`h` (Heikin-Ashi source) and the LinReg smoothing inputs are not
implemented -- this module always operates on raw OHLC closes, matching
the script's default (h=false).
"""

from __future__ import annotations

DEFAULT_KEY_VALUE = 2.0
DEFAULT_ATR_PERIOD = 1


def true_range(high: float, low: float, prev_close: float | None) -> float:
    """Wilder's true range: max(high-low, |high-prevClose|, |low-prevClose|),
    falling back to high-low on the first bar (no previous close)."""
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def wilder_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    """Pine's ta.atr(period): RMA (Wilder-smoothed) true range.

    RMA seeds at the simple mean of the first `period` true-range values
    (bars 0..period-1 all report that seed, matching Pine's behavior of
    only having a "closed" ATR value once enough bars exist), then
    recurses as rma[i] = (rma[i-1] * (period-1) + tr[i]) / period.
    period=1 degenerates to rma[i] = tr[i] exactly, the script's default.
    """
    n = len(closes)
    trs = [true_range(highs[i], lows[i], closes[i - 1] if i > 0 else None) for i in range(n)]
    if n == 0:
        return []

    atr = [0.0] * n
    seed_end = min(period, n)
    seed = sum(trs[:seed_end]) / seed_end
    for i in range(seed_end):
        atr[i] = seed
    for i in range(seed_end, n):
        atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period
    return atr


def atr_trailing_stop(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    key_value: float = DEFAULT_KEY_VALUE,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> list[float]:
    """The recursive xATRTrailingStop line, one value per bar."""
    n = len(closes)
    if n == 0:
        return []
    atr = wilder_atr(highs, lows, closes, atr_period)

    stop = [0.0] * n
    prev_stop = 0.0
    prev_close = 0.0  # nz(src[1], 0) has no meaning before bar 0; Pine's
    # nz(xATRTrailingStop[1], 0) is what actually drives bar 0's branch,
    # so prev_close only matters from bar 1 onward.
    for i in range(n):
        src = closes[i]
        n_loss = key_value * atr[i]
        prior_src = closes[i - 1] if i > 0 else prev_close

        if src > prev_stop and prior_src > prev_stop:
            new_stop = max(prev_stop, src - n_loss)
        elif src < prev_stop and prior_src < prev_stop:
            new_stop = min(prev_stop, src + n_loss)
        elif src > prev_stop:
            new_stop = src - n_loss
        else:
            new_stop = src + n_loss

        stop[i] = new_stop
        prev_stop = new_stop

    return stop


def crossover(series: list[float], reference: list[float]) -> list[bool]:
    """Pine's ta.crossover(series, reference): series[1] <= reference[1]
    and series[0] > reference[0]. False on bar 0 (no previous bar)."""
    n = len(series)
    return [i > 0 and series[i - 1] <= reference[i - 1] and series[i] > reference[i] for i in range(n)]


def crossunder(series: list[float], reference: list[float]) -> list[bool]:
    n = len(series)
    return [i > 0 and series[i - 1] >= reference[i - 1] and series[i] < reference[i] for i in range(n)]


DEFAULT_VOL_FILTER_ATR_PERIOD = 14
DEFAULT_VOL_FILTER_LOOKBACK = 500
DEFAULT_VOL_FILTER_MAX_RATIO = 1.5


def _trailing_average(values: list[float], lookback: int) -> list[float | None]:
    """Simple trailing mean over the last `lookback` values, None until
    enough history exists (bars 0..lookback-2) -- a plain sliding-window
    sum, not pandas, to match this module's no-dependencies style."""
    n = len(values)
    result: list[float | None] = [None] * n
    window_sum = 0.0
    for i in range(n):
        window_sum += values[i]
        if i >= lookback:
            window_sum -= values[i - lookback]
        if i >= lookback - 1:
            result[i] = window_sum / lookback
    return result


def volatility_regime_ok(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    atr_period: int = DEFAULT_VOL_FILTER_ATR_PERIOD,
    lookback: int = DEFAULT_VOL_FILTER_LOOKBACK,
    max_ratio: float = DEFAULT_VOL_FILTER_MAX_RATIO,
) -> list[bool]:
    """True at bar i when current ATR(atr_period)-as-a-fraction-of-price is
    within `max_ratio` of its own trailing `lookback`-bar average -- i.e.
    volatility hasn't spiked to an abnormal regime relative to this
    symbol's own recent history. False for the first `lookback` bars (no
    trailing average yet) and during a detected spike.

    Motivated by USDJPY's long-only backtest: a walk-forward pass found
    one bad stretch (Dec 2024-May 2025, the BOJ carry-trade-unwind period)
    where ATR(14)-as-%-of-price ran ~50% above its usual level (0.20% vs
    a 0.135% baseline) while a sustained downtrend kept stopping out long
    entries. A naive "only go long above a long-term SMA" trend filter
    was tried first and made things WORSE overall (it also cut genuinely
    profitable trades in unrelated, calmer windows) -- this volatility-
    ratio filter is more surgical: on the same walk-forward folds it left
    6 of 7 windows' trades completely untouched while cutting the bad
    fold's loss roughly in half and improving total P&L across all folds
    (deliberately independent of `atr_period` -- the strategy's own
    trailing-stop parameter, which can be as short as 1 -- since ATR(14)
    is a steadier, more standard volatility yardstick for this purpose)."""
    n = len(closes)
    atr = wilder_atr(highs, lows, closes, atr_period)
    atr_pct = [a / c if c else 0.0 for a, c in zip(atr, closes)]
    trailing_avg = _trailing_average(atr_pct, lookback)
    return [
        avg is not None and avg > 0 and atr_pct[i] <= avg * max_ratio
        for i, avg in enumerate(trailing_avg)
    ]


def find_ut_bot_long_trades(
    bars: dict,
    key_value: float = DEFAULT_KEY_VALUE,
    atr_period: int = DEFAULT_ATR_PERIOD,
    vol_filter_lookback: int | None = None,
    vol_filter_max_ratio: float = DEFAULT_VOL_FILTER_MAX_RATIO,
    vol_filter_atr_period: int = DEFAULT_VOL_FILTER_ATR_PERIOD,
) -> list[dict]:
    """Long-only walk over one symbol's bars: enter on a buy crossover
    while flat, exit on a sell crossunder while in a position. Fills at
    the signal bar's own close (Pine alerts fire at bar close; no
    repainting/lookahead). At most one open position at a time.

    Args:
        bars: {"high": [...], "low": [...], "close": [...], "date": [...]}
            -- equal-length lists, chronologically sorted.
        vol_filter_lookback: see volatility_regime_ok. Left None (the
            default), no regime filter is applied at all -- every result
            already reported for this strategy is on an unfiltered basis.
            When set, a buy crossover is skipped (not just delayed -- a
            fresh crossover is required to try again) whenever
            volatility_regime_ok is False for that bar.

    Returns:
        list of {"entry_idx", "entry_date", "entry_price", "stop_at_entry",
        "exit_idx", "exit_date", "exit_price", "reason"} -- one dict per
        completed round trip. A position still open at the end of the
        data closes at the last bar's close with reason "end_of_data".
    """
    highs, lows, closes, dates = bars["high"], bars["low"], bars["close"], bars["date"]
    n = len(closes)
    if n == 0:
        return []

    stop = atr_trailing_stop(highs, lows, closes, key_value, atr_period)
    buy_signal = crossover(closes, stop)
    sell_signal = crossunder(closes, stop)
    regime_ok = (
        volatility_regime_ok(highs, lows, closes, vol_filter_atr_period, vol_filter_lookback, vol_filter_max_ratio)
        if vol_filter_lookback is not None
        else [True] * n
    )

    trades: list[dict] = []
    open_trade: dict | None = None

    for i in range(n):
        if open_trade is None:
            if buy_signal[i] and regime_ok[i]:
                open_trade = {
                    "entry_idx": i,
                    "entry_date": dates[i],
                    "entry_price": closes[i],
                    "stop_at_entry": stop[i],
                }
        else:
            if sell_signal[i]:
                trades.append({
                    **open_trade,
                    "exit_idx": i, "exit_date": dates[i], "exit_price": closes[i], "reason": "sell_signal",
                })
                open_trade = None

    if open_trade is not None:
        trades.append({
            **open_trade,
            "exit_idx": n - 1, "exit_date": dates[n - 1], "exit_price": closes[n - 1], "reason": "end_of_data",
        })

    return trades


def find_ut_bot_confirmed_trades(
    bars: dict,
    key_value: float = DEFAULT_KEY_VALUE,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> list[dict]:
    """Long+short variant with a 1-bar confirmation filter on ENTRIES only
    (user-specified addition on top of the raw UT Bot rule, not part of
    the original Pine Script):

    Entry (long): bar i has a raw buy crossover AND the stop line is
        rising (stop[i] > stop[i-1]) -- this arms a pending long. If bar
        i+1's close is STILL above stop[i+1], the long enters at that
        bar's close. If it isn't, the setup is discarded outright (no
        re-checking later bars -- a fresh crossover is required to
        re-arm).
    Entry (short): mirror -- raw sell crossunder AND the stop line
        falling (stop[i] < stop[i-1]), confirmed if bar i+1's close is
        still below stop[i+1].
    Exit: the ORIGINAL, unconfirmed rule -- a long closes on the next
        crossunder, a short closes on the next crossover. No
        confirmation delay on exits.

    At most one open position (long or short) at a time. An exit's
    crossing event can simultaneously satisfy the opposite side's arming
    condition, so a flip can arm a new pending entry on the very same bar
    a position closes.

    Returns list of {"side" ("long"/"short"), "entry_idx", "entry_date",
    "entry_price", "stop_at_entry", "exit_idx", "exit_date", "exit_price",
    "reason"}.
    """
    highs, lows, closes, dates = bars["high"], bars["low"], bars["close"], bars["date"]
    n = len(closes)
    if n < 2:
        return []

    stop = atr_trailing_stop(highs, lows, closes, key_value, atr_period)
    buy_cross = crossover(closes, stop)
    sell_cross = crossunder(closes, stop)

    trades: list[dict] = []
    open_trade: dict | None = None
    pending: dict | None = None  # {"side", "armed_idx"}

    for i in range(n):
        # 1. Exit an open position with the plain (unconfirmed) rule.
        if open_trade is not None:
            if open_trade["side"] == "long" and sell_cross[i]:
                trades.append({**open_trade, "exit_idx": i, "exit_date": dates[i],
                               "exit_price": closes[i], "reason": "sell_signal"})
                open_trade = None
            elif open_trade["side"] == "short" and buy_cross[i]:
                trades.append({**open_trade, "exit_idx": i, "exit_date": dates[i],
                               "exit_price": closes[i], "reason": "buy_signal"})
                open_trade = None

        # 2. Resolve a pending confirmation due this bar (armed at i-1).
        if open_trade is None and pending is not None and pending["armed_idx"] == i - 1:
            if pending["side"] == "long" and closes[i] > stop[i]:
                open_trade = {"side": "long", "entry_idx": i, "entry_date": dates[i],
                              "entry_price": closes[i], "stop_at_entry": stop[i]}
            elif pending["side"] == "short" and closes[i] < stop[i]:
                open_trade = {"side": "short", "entry_idx": i, "entry_date": dates[i],
                              "entry_price": closes[i], "stop_at_entry": stop[i]}
            pending = None  # consumed either way (confirmed or invalidated)

        # 3. If flat with nothing pending, a fresh cross + sloping stop
        # arms a new pending entry for the NEXT bar to confirm.
        if open_trade is None and pending is None and i > 0:
            if buy_cross[i] and stop[i] > stop[i - 1]:
                pending = {"side": "long", "armed_idx": i}
            elif sell_cross[i] and stop[i] < stop[i - 1]:
                pending = {"side": "short", "armed_idx": i}

    if open_trade is not None:
        trades.append({**open_trade, "exit_idx": n - 1, "exit_date": dates[n - 1],
                        "exit_price": closes[n - 1], "reason": "end_of_data"})

    return trades
