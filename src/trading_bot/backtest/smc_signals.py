"""Per-symbol signal generation for the SMC ("Smart Money Concepts") Level 1
strategy, as specified by the user's source write-up:

  1. Identify trend via ChoCh (change of character) -> Higher Low -> BoS
     (break of structure). Everything here is for LONGS in an uptrend.
  2. A bullish order block (OB) is the last bearish candle before price
     breaks structure (ChoCh or BoS), validated by:
       - Imbalance/FVG: a 3-candle gap (candle[i].high < candle[i+2].low)
         starting at the OB candle.
       - Time: must be retested within 33 bars of formation (the "golden
         window").
       - Unmitigated: only the FIRST retest counts.
  3. Entry on the topside of the OB (its high) when price retests it.
  4. Stop below the OB's wick low.
  5. TP1 at the first bearish OB (resistance) above entry -- close 25%,
     move stop to breakeven.
  6. Full exit on the next new (post-entry) swing high.

This is a stateful, per-symbol, single forward pass -- unlike the gap-
breakout strategy's per-bar boolean filters, an SMC order block's validity
depends on multi-bar structure and its own trade lifecycle, so signal
generation and trade simulation are done together in one walk through the
bars. Each symbol is fully independent (no cross-symbol dependency), so
this can run before any portfolio-level (multi-symbol) event loop -- see
smc_engine.py.

INTERPRETATION NOTES (the source write-up is a conceptual breakdown, not a
precise algorithm -- these choices are documented here, not hidden):
  - Swing high/low pivots use the same "2 bars before AND after" definition
    already used elsewhere in this codebase (cycle.compute_swing_lows),
    mirrored for highs. This carries the same ~2-bar confirmation lag.
  - The OB candle is found by walking backward from a structure-break bar
    to the nearest bearish candle, capped at OB_SEARCH_LOOKBACK bars.
  - Entry fill is assumed exact at the OB's high (a resting limit order),
    not modeling slippage on a fast break-and-return.
  - "Full exit on a new high" is interpreted as the first CONFIRMED swing
    high after entry (same pivot definition/lag as above), filled at that
    pivot bar's close.
"""

from __future__ import annotations

OB_SEARCH_LOOKBACK = 10
DEFAULT_TIME_WINDOW_BARS = 33
DEFAULT_TP1_FRACTION = 0.25


def candle_direction(open_: float, close: float) -> str:
    """"bearish" iff close < open; ties count as bullish (arbitrary but
    documented -- a doji is rare and its direction barely matters here)."""
    return "bearish" if close < open_ else "bullish"


DEFAULT_SWING_WINDOW = 2


def find_swing_highs(highs: list[float], window: int = DEFAULT_SWING_WINDOW) -> list[tuple[int, float]]:
    """(index, value) pairs where a bar's high exceeds the `window` bars
    before AND after it (strict) -- mirrors cycle.compute_swing_lows's
    pivot definition (window=2), for highs, keeping the index (needed for
    OB bookkeeping). A wider window filters out small, noisy pivots in
    favor of more significant structural swings -- important on 5-min
    bars, where window=2 fires on nearly every local wiggle."""
    swings = []
    for i in range(window, len(highs) - window):
        before = highs[i - window:i]
        after = highs[i + 1:i + 1 + window]
        if highs[i] > max(before) and highs[i] > max(after):
            swings.append((i, highs[i]))
    return swings


def find_swing_lows(lows: list[float], window: int = DEFAULT_SWING_WINDOW) -> list[tuple[int, float]]:
    """(index, value) pairs, mirroring find_swing_highs for lows."""
    swings = []
    for i in range(window, len(lows) - window):
        before = lows[i - window:i]
        after = lows[i + 1:i + 1 + window]
        if lows[i] < min(before) and lows[i] < min(after):
            swings.append((i, lows[i]))
    return swings


def has_fvg(highs: list[float], lows: list[float], start_idx: int) -> bool:
    """Bullish fair value gap starting at `start_idx`: the first candle's
    high doesn't reach the third candle's low, leaving an unfilled gap."""
    if start_idx + 2 >= len(highs):
        return False
    return highs[start_idx] < lows[start_idx + 2]


def _find_ob_candle(opens: list[float], closes: list[float], break_idx: int) -> int | None:
    """Walk backward from a structure-break bar to the nearest bearish
    candle (the order block), capped at OB_SEARCH_LOOKBACK bars back."""
    for i in range(break_idx, max(break_idx - OB_SEARCH_LOOKBACK, -1), -1):
        if candle_direction(opens[i], closes[i]) == "bearish":
            return i
    return None


def find_smc_long_trades(
    bars: dict,
    time_window_bars: int = DEFAULT_TIME_WINDOW_BARS,
    tp1_fraction: float = DEFAULT_TP1_FRACTION,
    swing_window: int = DEFAULT_SWING_WINDOW,
    require_confirmed_trend: bool = False,
    force_close_same_day: bool = False,
) -> list[dict]:
    """Scan one symbol's chronological 5-min bars and return every long
    trade this Level 1 SMC strategy would have taken, fully simulated
    (entry, stop, TP1, full-exit or end-of-data), independent of any other
    symbol or portfolio-level state (sizing/caps are applied later by
    smc_engine.py).

    Args:
        bars: {"open": [...], "high": [...], "low": [...], "close": [...],
            "date": [...]} -- equal-length lists, one entry per bar,
            already sorted chronologically (see smc_engine.py for how this
            is built from the cached CSVs).
        require_confirmed_trend: a trend-quality experiment, NOT part of
            the literal Level 1 spec (which explicitly forms an OB at the
            ChoCh itself). When True, skip the OB formed at the very break
            that FIRST flips trend to "up" (the raw ChoCh, unconfirmed) and
            only trade OBs formed at later continuation breaks (BoS) once
            the uptrend has already proven itself -- trading the source
            write-up's own observation that a ChoCh can be indistinguishable
            from an "inducement" (a fakeout) without more confirmation.
        force_close_same_day: NOT part of the literal Level 1 spec -- an
            overnight-gap-risk experiment, same idea as engine.py's
            force_close_daily for the gap-breakout strategy. When True,
            force-exit at that bar's close whenever the current bar is the
            last bar of its calendar day and a position is still open
            (reason "same_day_force_close"), instead of letting it ride
            into the next session and risk gapping through its stop
            before the next open.

    Returns:
        list of dicts: {entry_idx, entry_date, entry_price, stop_price,
        ob_idx, tp1_price (or None), fills: [{"idx","date","price","qty_
        fraction","reason"}, ...]} -- qty_fraction sums to 1.0 across a
        trade's fills (0.25 at TP1 then 0.75 at exit, or 1.0 in one fill
        for a stop-out / no-TP1 exit).
    """
    opens, highs, lows, closes, dates = bars["open"], bars["high"], bars["low"], bars["close"], bars["date"]
    n = len(closes)
    if n < 10:
        return []

    swing_highs = find_swing_highs(highs, swing_window)
    swing_lows = find_swing_lows(lows, swing_window)

    trend = "none"
    last_swing_high = None
    last_swing_low = None
    swing_high_ptr = 0  # next un-consumed swing_highs index
    swing_low_ptr = 0

    # Pending, unmitigated bullish OBs awaiting their first retest:
    # {"ob_idx", "ob_high", "ob_low", "expiry_idx"}
    pending_bull_obs: list[dict] = []
    # Confirmed bearish OBs (resistance zones for TP1), most recent last:
    # {"ob_idx", "ob_high", "ob_low"}
    bearish_obs: list[dict] = []

    trades: list[dict] = []
    open_trade = None  # at most one open position per symbol at a time

    for i in range(n):
        # --- advance confirmed swing points up to bar i (swing_window-bar lag) ---
        while swing_high_ptr < len(swing_highs) and swing_highs[swing_high_ptr][0] <= i - swing_window:
            last_swing_high = swing_highs[swing_high_ptr][1]
            swing_high_ptr += 1
        while swing_low_ptr < len(swing_lows) and swing_lows[swing_low_ptr][0] <= i - swing_window:
            last_swing_low = swing_lows[swing_low_ptr][1]
            swing_low_ptr += 1

        # --- bullish structure break (ChoCh or BoS) -> candidate bullish OB ---
        if last_swing_high is not None and closes[i] > last_swing_high:
            was_already_up = trend == "up"
            trend = "up"
            take_this_ob = (not require_confirmed_trend) or was_already_up
            if take_this_ob:
                ob_idx = _find_ob_candle(opens, closes, i)
                if ob_idx is not None and has_fvg(highs, lows, ob_idx):
                    pending_bull_obs.append({
                        "ob_idx": ob_idx, "ob_high": highs[ob_idx], "ob_low": lows[ob_idx],
                        "expiry_idx": ob_idx + time_window_bars,
                    })
            last_swing_high = None  # look for the next swing high to define the next BoS

        # --- bearish structure break -> candidate bearish OB (resistance only) ---
        if last_swing_low is not None and closes[i] < last_swing_low:
            ob_idx = _find_ob_candle(opens, closes, i)  # nearest bearish candle by direction search
            # A bearish OB is the last BULLISH candle before a downside break.
            bear_ob_idx = None
            for j in range(i, max(i - OB_SEARCH_LOOKBACK, -1), -1):
                if candle_direction(opens[j], closes[j]) == "bullish":
                    bear_ob_idx = j
                    break
            if bear_ob_idx is not None:
                bearish_obs.append({"ob_idx": bear_ob_idx, "ob_high": highs[bear_ob_idx], "ob_low": lows[bear_ob_idx]})
            last_swing_low = None

        # --- drop expired, unmitigated bullish OBs ---
        pending_bull_obs = [ob for ob in pending_bull_obs if ob["expiry_idx"] >= i]

        # --- manage an already-open trade ---
        if open_trade is not None:
            pos = open_trade
            if lows[i] <= pos["stop_price"]:
                pos["fills"].append({
                    "idx": i, "date": dates[i], "price": pos["stop_price"],
                    "qty_fraction": pos["remaining_fraction"], "reason": "stop",
                })
                trades.append(pos)
                open_trade = None
            else:
                if not pos["tp1_done"] and pos["tp1_price"] is not None and highs[i] >= pos["tp1_price"]:
                    pos["fills"].append({
                        "idx": i, "date": dates[i], "price": pos["tp1_price"],
                        "qty_fraction": tp1_fraction, "reason": "tp1",
                    })
                    pos["remaining_fraction"] = round(pos["remaining_fraction"] - tp1_fraction, 6)
                    pos["stop_price"] = pos["entry_price"]  # move to breakeven
                    pos["tp1_done"] = True

                # full exit on the first confirmed swing high after entry
                if pos["tp1_done"] or pos["tp1_price"] is None:
                    for sh_idx, _ in swing_highs:
                        if pos["entry_idx"] < sh_idx <= i and sh_idx + swing_window <= i:
                            pos["fills"].append({
                                "idx": sh_idx, "date": dates[sh_idx], "price": closes[sh_idx],
                                "qty_fraction": pos["remaining_fraction"], "reason": "new_high_exit",
                            })
                            trades.append(pos)
                            open_trade = None
                            break

            if (
                force_close_same_day
                and open_trade is not None
                and (i == n - 1 or dates[i].date() != dates[i + 1].date())
            ):
                pos["fills"].append({
                    "idx": i, "date": dates[i], "price": closes[i],
                    "qty_fraction": pos["remaining_fraction"], "reason": "same_day_force_close",
                })
                trades.append(pos)
                open_trade = None

        # --- look for a new entry (only when flat) ---
        # With force_close_same_day, skip entries on the last bar of a day:
        # there'd be no later bar left that day to force-close it at, which
        # would otherwise let exactly this edge case ride overnight anyway.
        entry_window_open = not (
            force_close_same_day and (i == n - 1 or dates[i].date() != dates[i + 1].date())
        )
        if open_trade is None and pending_bull_obs and entry_window_open:
            for ob in pending_bull_obs:
                if ob["ob_idx"] < i and lows[i] <= ob["ob_high"]:
                    entry_price = ob["ob_high"]
                    stop_price = ob["ob_low"]
                    if entry_price <= stop_price:
                        continue  # degenerate OB, skip
                    tp1_price = None
                    resistances_above = [b["ob_low"] for b in bearish_obs if b["ob_low"] > entry_price]
                    if resistances_above:
                        tp1_price = min(resistances_above)

                    open_trade = {
                        "entry_idx": i, "entry_date": dates[i], "entry_price": entry_price,
                        "stop_price": stop_price, "ob_idx": ob["ob_idx"], "tp1_price": tp1_price,
                        "tp1_done": False, "remaining_fraction": 1.0, "fills": [],
                    }
                    pending_bull_obs.remove(ob)  # mitigated -- first retest consumed it
                    break

    # End-of-data safety net: close any still-open trade at the last bar.
    if open_trade is not None:
        open_trade["fills"].append({
            "idx": n - 1, "date": dates[n - 1], "price": closes[n - 1],
            "qty_fraction": open_trade["remaining_fraction"], "reason": "end_of_data",
        })
        trades.append(open_trade)

    return trades


# ---------------------------------------------------------------------------
# Live-polling adapters (used by cli/smc_cycle.py)
#
# The live bot re-runs the SAME find_smc_long_trades pass over recent bar
# history each cycle instead of tracking OB state incrementally -- one
# source of truth for the signal logic, no live-vs-backtest drift.
# ---------------------------------------------------------------------------

def latest_entry_signal(
    bars: dict,
    time_window_bars: int = DEFAULT_TIME_WINDOW_BARS,
    tp1_fraction: float = DEFAULT_TP1_FRACTION,
    swing_window: int = DEFAULT_SWING_WINDOW,
    require_confirmed_trend: bool = False,
) -> dict | None:
    """If this symbol's SMC entry would trigger on the LAST bar of `bars`,
    return that trade dict (entry_price = OB high, stop_price = OB low,
    tp1_price or None); else None.

    Deliberately runs with force_close_same_day=False: that flag skips
    entries on the final bar of a day, and to a mid-session recompute the
    latest fetched bar ALWAYS looks like the final bar -- it would
    suppress every live entry. Same-day discipline is enforced by the
    live bot's own EOD force-close instead, and the entry-window gate
    (no entries near the close) covers the true last-bar-of-day case.
    """
    n = len(bars["close"])
    if n == 0:
        return None
    for trade in find_smc_long_trades(
        bars, time_window_bars, tp1_fraction, swing_window, require_confirmed_trend,
        force_close_same_day=False,
    ):
        if trade["entry_idx"] == n - 1:
            return trade
    return None


def confirmed_new_high_exit(highs: list[float], entry_idx: int, swing_window: int = DEFAULT_SWING_WINDOW) -> bool:
    """True once a swing high has formed strictly AFTER entry_idx and been
    confirmed (i.e. swing_window bars have printed after it) -- the same
    `entry_idx < sh_idx <= i and sh_idx + swing_window <= i` condition the
    backtest's full-exit branch uses, evaluated at i = the latest bar."""
    last_idx = len(highs) - 1
    for sh_idx, _ in find_swing_highs(highs, swing_window):
        if sh_idx > entry_idx and sh_idx + swing_window <= last_idx:
            return True
    return False
