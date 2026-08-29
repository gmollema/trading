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
  - Entry fill: see ENTRY_FILLS below. The original spec filled at the
    OB's high the instant a bar's low touched it, which no order type can
    actually achieve (smc_fill_model, 2026-08-28); the reachable specs
    fill on the bar AFTER the signal bar closes.
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


# Fill-side slippage, in basis points, keyed by the fill's `reason`.
#
# The backtest triggers on a bar level but must not FILL at it: every live
# exit is a market order sent after the level is touched, and the entry is
# a market order sent after the OB high is retested. Modelling zero
# slippage makes two trade classes look better than they can be -- the
# breakeven stop (which fills at exactly entry, so the runner leg is
# exactly $0 across every such trade) and the TP1 partial (which fills at
# exactly tp1_price, where the live bot's market order came in 49 bps
# lower on the first real fill we have). Defaults are 0.0 so existing
# callers -- including smc_live.py -- are byte-for-byte unchanged.
# Fraction of the entry-to-initial-stop distance the stop covers once TP1
# fills. 1.0 == move to breakeven, the literal Level 1 spec.
DEFAULT_POST_TP1_STOP_FRACTION = 1.0

SLIPPAGE_REASONS = ("entry", "stop", "tp1", "new_high_exit", "same_day_force_close", "end_of_data")
DEFAULT_SLIPPAGE_BPS = {r: 0.0 for r in SLIPPAGE_REASONS}


def _normalize_slippage_bps(slippage_bps: dict | float | int | None) -> dict:
    """Accept None (no slippage), a scalar applied to every leg, or a partial
    dict keyed by fill reason. Unknown keys are rejected rather than silently
    ignored -- a typo'd reason would otherwise read as "no slippage there"."""
    if slippage_bps is None:
        return dict(DEFAULT_SLIPPAGE_BPS)
    if isinstance(slippage_bps, (int, float)):
        return {r: float(slippage_bps) for r in SLIPPAGE_REASONS}
    unknown = set(slippage_bps) - set(SLIPPAGE_REASONS)
    if unknown:
        raise ValueError(f"unknown slippage_bps keys: {sorted(unknown)}; expected {list(SLIPPAGE_REASONS)}")
    return {r: float(slippage_bps.get(r, 0.0)) for r in SLIPPAGE_REASONS}


def _slipped(price: float, bps: float, side: str) -> float:
    """Move `price` adversely by `bps`: a buy fills higher, a sell lower."""
    if not bps:
        return price
    factor = 1.0 + bps / 10_000.0 if side == "buy" else 1.0 - bps / 10_000.0
    return price * factor


# Where the entry actually fills, once a retest has been detected.
#
# "level" is the original spec: fill at the OB's high the instant a bar's
# low touches it. Nothing achieves that. A limit resting there fills 31-42%
# of the time and almost only when the setup is failing (smc_fill_model),
# and a market order sent once the touch is known pays 48-79 bps on the
# comparable leg against a 17.3 bps median stop distance. The level is not
# a price this strategy can trade at; it is only the trigger.
#
# The other two are reachable by construction: the signal is decided on a
# CLOSED bar and the fill happens on the next one, which is what a market
# order sent at the close actually gets. They bracket where in that bar the
# fill lands rather than claiming one number:
#
#   next_open: the first print of the fill bar -- what an order sent the
#       instant the signal bar closes should get.
#   next_high: the worst price the fill bar ever offered. A hard upper
#       bound on what any fill inside that bar could have cost, however
#       late the order went in.
#
# The live cycle currently fires ~2 minutes into the fill bar, so its true
# fill sits between the two, nearer next_open the earlier the cycle runs.
ENTRY_FILLS = ("level", "next_open", "next_high")
DEFAULT_ENTRY_FILL = "level"


def _same_session(dates: list, a: int, b: int) -> bool:
    """Whether bars a and b fall on the same calendar day.

    Bars carry pandas Timestamps in every real caller. The unit fixtures
    use plain integers as date stand-ins, which have no .date(); those
    describe one continuous session, so treat them as such rather than
    forcing every fixture to grow real datetimes.
    """
    da, db = dates[a], dates[b]
    if hasattr(da, "date") and hasattr(db, "date"):
        return da.date() == db.date()
    return True


def _is_last_bar_of_day(dates: list, i: int, n: int) -> bool:
    return i == n - 1 or not _same_session(dates, i, i + 1)


def _entry_fill_bar(dates: list, i: int, n: int, entry_fill: str, force_close_same_day: bool) -> int | None:
    """Index of the bar the entry fills on, or None if it cannot fill.

    A reachable fill needs a bar after the signal bar, in the same session
    -- nothing rests overnight, and an order sent at 15:55 is not live at
    the next open. Under force_close_same_day the FILL bar (not the signal
    bar) is what must not be a day's last, since that is the bar the
    position would have to be closed on.
    """
    fill_idx = i if entry_fill == "level" else i + 1
    if fill_idx >= n:
        return None
    if fill_idx != i and not _same_session(dates, i, fill_idx):
        return None
    if force_close_same_day and _is_last_bar_of_day(dates, fill_idx, n):
        return None
    return fill_idx


def find_smc_long_trades(
    bars: dict,
    time_window_bars: int = DEFAULT_TIME_WINDOW_BARS,
    tp1_fraction: float = DEFAULT_TP1_FRACTION,
    swing_window: int = DEFAULT_SWING_WINDOW,
    require_confirmed_trend: bool = False,
    force_close_same_day: bool = False,
    slippage_bps: dict | float | None = None,
    post_tp1_stop_fraction: float = DEFAULT_POST_TP1_STOP_FRACTION,
    exit_fully_at_tp1: bool = False,
    entry_fill: str = DEFAULT_ENTRY_FILL,
    require_ob_reclaim: bool = False,
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
        slippage_bps: adverse fill slippage in basis points -- None or 0
            for the frictionless fills this backtest historically assumed,
            a scalar for one rate across every leg, or a dict keyed by
            fill reason (see SLIPPAGE_REASONS) to price the legs
            separately, which is usually what you want: a resting stop
            and a market-order TP1 do not slip alike. Levels still
            TRIGGER exactly as before; only the recorded fill price
            moves, adversely (buys higher, sells lower). Note this makes
            the post-TP1 breakeven stop a small LOSS rather than a
            guaranteed scratch, which is the realistic outcome.
        post_tp1_stop_fraction: how far the stop travels from
            initial_stop_price toward entry_price once TP1 fills, as a
            fraction of that distance. 1.0 (the default, and the only
            behavior available before this was a parameter) is the
            literal Level 1 spec's move to breakeven; 0.0 leaves the
            original stop untouched and lets the runner keep its full
            room; values above 1.0 push the stop past entry to lock in
            profit at the cost of stopping out sooner. Worth sweeping
            because breakeven is not free: it converts a large share of
            trades into scratches that still occupy one of the portfolio's
            concurrent-position slots.
        entry_fill: where the entry fills once a retest is detected -- one
            of ENTRY_FILLS. "level" (the default, and the only behavior
            available before this was a parameter) fills at the OB high on
            the touching bar, which is what every figure in this repo
            before 2026-08-29 assumed and what nothing can actually
            execute. "next_open" and "next_high" decide the signal on the
            CLOSED touching bar and fill on the next one, which a market
            order sent at that close does achieve; they bracket where
            inside the fill bar the order lands. Under both, `entry_idx`
            and `entry_date` refer to the FILL bar, while `signal_idx` and
            `signal_price` record the bar and level that triggered it.
        require_ob_reclaim: only take the retest if the touching bar also
            CLOSES back above the OB high -- the setup rejecting the level
            rather than sinking through it. This is a signal filter, not a
            fill model: it selects a subset of the same retests, so its
            results stay directly comparable. smc_fill_model measured the
            two cohorts on 17,354 signals and they are barely the same
            strategy -- bars closing back above the level returned +0.49%
            mean at a 62% win rate, bars closing below returned -0.05% at
            18%. Either way the first retest MITIGATES the order block, so
            a rejected one is consumed rather than left pending for a
            second attempt; that keeps the "only the FIRST retest counts"
            rule intact instead of quietly inventing a wait-for-a-better-
            retest strategy.
        exit_fully_at_tp1: sell the ENTIRE position at tp1_price instead of
            tp1_fraction of it, ending the trade there -- no runner, no
            breakeven stop, no new-high exit. This is the exit policy that
            post_tp1_stop_fraction >= 1.5 degenerated into once the
            above-market stop was clamped, expressed directly so it can be
            measured on its own fills (at tp1_price on the touching bar)
            rather than inferred from a stop resting at some earlier bar's
            close. Mutually exclusive with post_tp1_stop_fraction, which
            has nothing left to place.

    Returns:
        list of dicts: {entry_idx, entry_date, entry_price, signal_idx,
        signal_price, stop_price, initial_stop_price, ob_idx, tp1_price
        (or None), fills: [{"idx",
        "date","price","qty_fraction","reason"}, ...]} -- qty_fraction sums
        to 1.0 across a trade's fills (0.25 at TP1 then 0.75 at exit, or
        1.0 in one fill for a stop-out / no-TP1 exit). `stop_price` is
        live exit-management state -- it moves to breakeven once TP1
        fills, so it does NOT describe the entry-time risk distance for a
        trade that reaches TP1. `initial_stop_price` is set once at entry
        and never mutated; anything computing entry-time risk (e.g.
        position sizing) must use `initial_stop_price`, not `stop_price`.
    """
    if entry_fill not in ENTRY_FILLS:
        raise ValueError(f"unknown entry_fill: {entry_fill!r}; expected one of {list(ENTRY_FILLS)}")
    opens, highs, lows, closes, dates = bars["open"], bars["high"], bars["low"], bars["close"], bars["date"]
    slip = _normalize_slippage_bps(slippage_bps)
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
                    "idx": i, "date": dates[i],
                    "price": _slipped(pos["stop_price"], slip["stop"], "sell"),
                    "qty_fraction": pos["remaining_fraction"], "reason": "stop",
                })
                trades.append(pos)
                open_trade = None
            else:
                if not pos["tp1_done"] and pos["tp1_price"] is not None and highs[i] >= pos["tp1_price"]:
                    exit_qty = pos["remaining_fraction"] if exit_fully_at_tp1 else tp1_fraction
                    pos["fills"].append({
                        "idx": i, "date": dates[i],
                        "price": _slipped(pos["tp1_price"], slip["tp1"], "sell"),
                        "qty_fraction": exit_qty, "reason": "tp1",
                    })
                    pos["remaining_fraction"] = round(pos["remaining_fraction"] - exit_qty, 6)
                    pos["tp1_done"] = True
                    if exit_fully_at_tp1:
                        # Whole position gone at TP1: no runner, so no stop to
                        # move and nothing for the exit rules below to manage.
                        trades.append(pos)
                        open_trade = None
                    else:
                        # Default 1.0 lands exactly on entry_price (breakeven).
                        new_stop = pos["initial_stop_price"] + post_tp1_stop_fraction * (
                            pos["entry_price"] - pos["initial_stop_price"]
                        )
                        # A stop cannot rest ABOVE the last traded price: live, that
                        # order is rejected or fires instantly at market. Without
                        # this clamp, post_tp1_stop_fraction > 1 books exits at
                        # prices the bars never offered, and the metric improves
                        # monotonically on fills that cannot happen.
                        pos["stop_price"] = min(new_stop, closes[i])

                # full exit on the first confirmed swing high after entry
                if open_trade is not None and (pos["tp1_done"] or pos["tp1_price"] is None):
                    for sh_idx, _ in swing_highs:
                        if pos["entry_idx"] < sh_idx <= i and sh_idx + swing_window <= i:
                            pos["fills"].append({
                                "idx": sh_idx, "date": dates[sh_idx],
                                "price": _slipped(closes[sh_idx], slip["new_high_exit"], "sell"),
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
                    "idx": i, "date": dates[i],
                    "price": _slipped(closes[i], slip["same_day_force_close"], "sell"),
                    "qty_fraction": pos["remaining_fraction"], "reason": "same_day_force_close",
                })
                trades.append(pos)
                open_trade = None

        # --- look for a new entry (only when flat) ---
        # Bar i is the SIGNAL bar: its low reached the OB's high. Where
        # that turns into a fill is entry_fill's job -- for the reachable
        # specs the position does not exist until the next bar, so it is
        # created here with entry_idx pointing there and the exit
        # management above only picks it up from that bar onward.
        # Iterating a copy: a retest consumes its order block whether or
        # not it produces a trade, and the rejection is about that block
        # rather than about the bar -- a second, lower OB touched by the
        # same bar still gets its own test.
        if open_trade is None and pending_bull_obs:
            for ob in list(pending_bull_obs):
                if ob["ob_idx"] >= i or lows[i] > ob["ob_high"]:
                    continue

                if require_ob_reclaim and closes[i] <= ob["ob_high"]:
                    # Retested and rejected: still mitigated, just not traded.
                    pending_bull_obs.remove(ob)
                    continue

                fill_idx = _entry_fill_bar(dates, i, n, entry_fill, force_close_same_day)
                if fill_idx is None:
                    # No reachable fill bar, and that is a property of the
                    # bar, not of this block -- so it holds for every other
                    # pending OB too. Leave them all pending.
                    break

                raw_entry = ob["ob_high"] if entry_fill == "level" else (
                    opens[fill_idx] if entry_fill == "next_open" else highs[fill_idx]
                )
                entry_price = _slipped(raw_entry, slip["entry"], "buy")
                stop_price = ob["ob_low"]
                if entry_price <= stop_price:
                    # No positive risk-per-share to size against, so the
                    # trade is dropped -- but the retest still happened, so
                    # the block is consumed rather than left pending to
                    # fire again a bar later at some unrelated price.
                    #
                    # Under "level" this is a flat OB candle and could
                    # never have traded anyway (the comparison is on fixed
                    # values, so it fails identically on every later
                    # touch). Under the reachable specs it means the fill
                    # bar opened at or below the stop, which live is a fill
                    # followed instantly by a stop-out, not a skipped
                    # trade -- the one place this model flatters itself.
                    # It shows up in smc_entry_spec's `signals` column as
                    # part of the gap between the level and next_* rows.
                    pending_bull_obs.remove(ob)
                    continue
                tp1_price = None
                resistances_above = [b["ob_low"] for b in bearish_obs if b["ob_low"] > entry_price]
                if resistances_above:
                    tp1_price = min(resistances_above)

                open_trade = {
                    "entry_idx": fill_idx, "entry_date": dates[fill_idx], "entry_price": entry_price,
                    "signal_idx": i, "signal_price": ob["ob_high"],
                    "stop_price": stop_price, "initial_stop_price": stop_price,
                    "ob_idx": ob["ob_idx"], "tp1_price": tp1_price,
                    "tp1_done": False, "remaining_fraction": 1.0, "fills": [],
                }
                pending_bull_obs.remove(ob)  # mitigated -- first retest consumed it
                break

    # End-of-data safety net: close any still-open trade at the last bar.
    if open_trade is not None:
        open_trade["fills"].append({
            "idx": n - 1, "date": dates[n - 1],
            "price": _slipped(closes[n - 1], slip["end_of_data"], "sell"),
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
    require_ob_reclaim: bool = False,
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

    Equally deliberately runs with entry_fill="level", the unreachable
    spec, and that is not a contradiction: the reachable specs fill on the
    bar AFTER the signal, which by definition does not exist yet at the
    moment the bot has to act. What the caller needs from this function is
    "is there a signal on the last closed bar", and entry_price is the
    TRIGGER level, not a fill prediction -- smc_cycle sends a market order
    and records what it actually got (see its signal_price /
    entry_slippage_bps logging). Running the fill spec here would return
    nothing at all, since entry_idx could never equal n-1.

    require_ob_reclaim, by contrast, IS a signal-side rule and must match
    the backtest exactly, so it is passed straight through. `bars` must
    therefore end on a CLOSED bar: a forming bar's close is just the last
    trade, and a reclaim that has not held to the bar's end is not the
    same signal the backtest scored.
    """
    n = len(bars["close"])
    if n == 0:
        return None
    for trade in find_smc_long_trades(
        bars, time_window_bars, tp1_fraction, swing_window, require_confirmed_trend,
        force_close_same_day=False, require_ob_reclaim=require_ob_reclaim,
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
