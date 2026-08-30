"""Portfolio-level backtest for the SMC Level 1 strategy (smc_signals.py).

Two phases, same pattern as the gap-breakout engine.py:
  1. Per-symbol signal generation (smc_signals.find_smc_long_trades) --
     each symbol's entries/stops/TP1/exits are fully determined by that
     symbol's own price action alone, independent of every other symbol.
  2. A single chronological merge across all symbols applying GLOBAL
     portfolio constraints (max_concurrent_positions, one position per
     symbol at a time, risk-based position sizing against current
     equity) -- a signal that arrives with no open slot is simply not
     taken (not queued to try again later).
"""

from __future__ import annotations

import heapq
import itertools
from pathlib import Path

import pandas as pd

from trading_bot.backtest import portfolio
from trading_bot.backtest.data import DAILY_DIR, INTRADAY_DIR, compute_daily_context, load_daily, load_intraday
from trading_bot.backtest.smc_signals import (
    BAR_INTERVAL_MINUTES,
    DEFAULT_ENTRY_FILL,
    DEFAULT_EXIT_FILL,
    DEFAULT_POST_TP1_STOP_FRACTION,
    DEFAULT_SWING_WINDOW,
    find_smc_long_trades,
)

ET_TZ = "America/New_York"
DEFAULT_MAX_POSITION_PCT = 10.0
DEFAULT_MAX_CONCURRENT_POSITIONS = 2
DEFAULT_TIME_WINDOW_BARS = 33
DEFAULT_TP1_FRACTION = 0.25
DEFAULT_REACTIVE_DERISK_PF_THRESHOLD = 0.8
DEFAULT_REACTIVE_DERISK_SIZE_MULT = 0.3


def _daily_uptrend_dates(ticker: str, daily_dir: Path) -> set | None:
    """Trading dates where this ticker's prior close was above its SMA200
    (same D2 check the gap-breakout strategy uses) -- a simple multi-
    timeframe "is the broader trend even bullish" quality filter. Returns
    None if no daily data is cached for this ticker (caller should then
    skip the filter for that symbol rather than silently allow everything
    through)."""
    daily_df = load_daily(ticker, daily_dir)
    if daily_df is None or daily_df.empty:
        return None
    ctx = compute_daily_context(daily_df)
    uptrend = ctx[ctx["prior_day_close"] > ctx["sma200"]]
    return set(uptrend["Date"].dt.date)


DEFAULT_SMA_WINDOW = 200
DEFAULT_DOLLAR_VOLUME_WINDOW = 20


def daily_watchlist_by_date(
    tickers: list[str],
    daily_dir: Path = DAILY_DIR,
    min_price: float = 0.0,
    max_size: int | None = None,
    sma_window: int = DEFAULT_SMA_WINDOW,
    dollar_volume_window: int = DEFAULT_DOLLAR_VOLUME_WINDOW,
) -> dict:
    """Rebuild, for every trading date, the watchlist the live bot had.

    cli/smc_prefilter.py runs at 09:40 ET and writes smc_watchlist.txt:
    S&P 500 names whose PRIOR close is above their SMA200 and at or above
    universe.min_price_usd, ranked by 20-day average dollar volume, capped
    at universe.max_watchlist_size. smc_cycle only ever scans that file.

    The backtests scanned all 503 tickers, so they were scoring a bot with
    a ~12x larger universe than the one running -- and not a random 12x:
    the cap keeps the most liquid names, which is where fills are best and
    5-minute bars most trustworthy. daily_trend_filter reproduced only the
    SMA200 half of the screen and nothing about the ranking or the cap.

    Everything here is anchored to prior-day data (compute_daily_context
    shifts by one bar, and the dollar-volume mean is shifted to match), so
    a date's watchlist is fully determined before that date opens -- the
    same property that lets the live screen run once each morning.

    Returns {trading date: set of tickers}. Dates with no qualifying
    ticker are present with an empty set, so a caller can tell "the
    universe was empty that day" from "that day is outside the data".
    """
    frames = []
    for ticker in tickers:
        daily_df = load_daily(ticker, daily_dir)
        if daily_df is None or daily_df.empty:
            continue
        ctx = compute_daily_context(daily_df)
        dollar_volume = (daily_df["Close"] * daily_df["Volume"]).rolling(dollar_volume_window).mean()
        frames.append(pd.DataFrame({
            "date": ctx["Date"].dt.date,
            "ticker": ticker,
            "prior_close": ctx["prior_day_close"],
            "sma200": ctx["Close"].rolling(sma_window).mean().shift(1),
            "dollar_volume": dollar_volume.shift(1),
        }))
    if not frames:
        return {}

    table = pd.concat(frames, ignore_index=True).dropna(
        subset=["prior_close", "sma200", "dollar_volume"]
    )
    eligible = table[(table.prior_close > table.sma200) & (table.prior_close >= min_price)]

    watchlist = {d: set() for d in table["date"].unique()}
    for date, group in eligible.groupby("date", sort=False):
        ranked = group.sort_values("dollar_volume", ascending=False)
        if max_size is not None:
            ranked = ranked.head(max_size)
        watchlist[date] = set(ranked["ticker"])
    return watchlist


def watchlist_from_rules(tickers: list[str], daily_dir: Path, rules: dict) -> dict:
    """daily_watchlist_by_date driven by smc_rules.json's universe block --
    the same keys cli/smc_prefilter.py reads, so the two cannot drift."""
    universe = rules.get("universe") or {}
    return daily_watchlist_by_date(
        tickers,
        daily_dir=daily_dir,
        min_price=universe.get("min_price_usd", 0.0),
        max_size=universe.get("max_watchlist_size"),
    )


# How long after a signal bar OPENS the live bot is in a position to act
# on it: 5 minutes for the bar to close, since smc_cycle evaluates closed
# bars only, plus the 2-minute stagger HT_SMC_Cycle runs on to keep its
# data burst off HT_Cycle's (see cli/setup_schedule.py).
CYCLE_STAGGER_MINUTES = 2
DEFAULT_ACTION_DELAY_MINUTES = BAR_INTERVAL_MINUTES + CYCLE_STAGGER_MINUTES


def entry_window_mask(
    dates,
    earliest_et: str | None,
    latest_et: str | None,
    action_delay_minutes: int = DEFAULT_ACTION_DELAY_MINUTES,
) -> list[bool]:
    """Per-bar mask for smc_signals' entry_allowed, from ET wall-clock
    bounds (smc_rules.json time_filter). Bounds are inclusive, matching
    smc_live.get_market_status' reading of the same two keys.

    The bounds are tested against when the bot could ACT on a bar, not
    when that bar opened, and the difference is a bar at each end. A
    signal on the bar opening at 10:00 is invisible until it closes at
    10:05 and is acted on by the 10:07 cycle -- so with earliest 10:05
    that bar qualifies and one opening at 09:55 does not. At the other
    end, a bar opening at 15:25 would be acted on at 15:32, past a 15:30
    cutoff, even though the bar itself opened inside the window. Gating on
    the open would take entries the live bot refuses and skip one it
    takes, which for a window this consequential is not a rounding error.

    `dates` is the intraday frame's tz-aware date column; the comparison
    is vectorized here because doing it per bar inside the signal loop is
    a tz conversion per bar per ticker.
    """
    acted = dates.dt.tz_convert(ET_TZ) + pd.Timedelta(minutes=action_delay_minutes)
    times = acted.dt.strftime("%H:%M")
    mask = pd.Series(True, index=times.index)
    if earliest_et:
        mask &= times >= earliest_et
    if latest_et:
        mask &= times <= latest_et
    return mask.tolist()


def _trade_row(timestamp, symbol: str, side: str, qty: float, price: float, order_id: int, reason: str) -> dict:
    # qty stays int for whole-share sizing (the common case) and float
    # only when fractional sizing is in play -- avoids turning every
    # ordinary backtest's "size": 333 into "size": 333.0 in the output.
    size = qty if isinstance(qty, float) and not qty.is_integer() else int(qty)
    return {
        "timestamp_iso": timestamp.tz_convert("UTC").isoformat(),
        "symbol": symbol,
        "side": side,
        "size": size,
        "fill_price": round(float(price), 4),
        "order_id": order_id,
        "status": "Filled",
        "reason": reason,
    }


def build_smc_candidates(
    tickers: list[str],
    intraday_dir: Path = INTRADAY_DIR,
    time_window_bars: int = DEFAULT_TIME_WINDOW_BARS,
    tp1_fraction: float = DEFAULT_TP1_FRACTION,
    swing_window: int = DEFAULT_SWING_WINDOW,
    require_confirmed_trend: bool = False,
    daily_trend_filter: bool = False,
    daily_dir: Path = DAILY_DIR,
    force_close_same_day: bool = False,
    slippage_bps: dict | float | None = None,
    post_tp1_stop_fraction: float = DEFAULT_POST_TP1_STOP_FRACTION,
    exit_fully_at_tp1: bool = False,
    entry_fill: str = DEFAULT_ENTRY_FILL,
    require_ob_reclaim: bool = False,
    exit_fill: str = DEFAULT_EXIT_FILL,
    tp1_resting_limit: bool = False,
    daily_watchlist: dict | None = None,
    entry_window_et: tuple | None = None,
) -> list[tuple]:
    """Per-symbol signal generation only (see module docstring, phase 1) --
    independent of everything portfolio-level (equity, sizing, concurrency,
    reactive_derisk_*), so a caller sweeping those can build this ONCE per
    (tickers, intraday_dir, ...) combination and reuse it across many
    simulate_smc_portfolio() calls instead of re-scanning every ticker's
    full history per combo. Returns a list of (entry_date, symbol, trade)
    tuples sorted the way simulate_smc_portfolio expects.

    daily_watchlist: {trading date: set of tickers} from
        daily_watchlist_by_date -- the universe the live bot's morning
        prefilter would have handed it that day. A candidate survives only
        if its symbol was on that date's list. Supersedes
        daily_trend_filter, which reproduced the SMA200 screen but not the
        dollar-volume ranking or the 40-name cap, and so still scanned
        roughly twelve times the live universe.

        Filtering here rather than inside find_smc_long_trades is exact
        rather than convenient, and only because entries are gated to one
        day: with force_close_same_day on, no position spans a date
        boundary, so dropping a day's trades cannot change what any other
        day would have done. The gate is on the ENTRY date either way,
        which is also what the live bot does -- it manages an open
        position whether or not the symbol is still on today's list.

    entry_window_et: (earliest, latest) ET wall-clock bounds for opening a
        new position, e.g. ("10:05", "15:30") from smc_rules.json's
        time_filter. Without it the backtest opened positions at 09:35 and
        15:45, times the live cycle does not even scan at.
    """
    candidates = []  # (entry_date, symbol, trade_dict)
    if daily_watchlist is not None:
        # A ticker that never appears on any date's list cannot produce a
        # surviving candidate, so scanning it is pure waste -- and with a
        # 40-name cap over a year that is most of the index (105 of 503 on
        # the current cache). Exact, not an approximation: the filter below
        # would drop every one of their trades anyway.
        ever_listed = set().union(*daily_watchlist.values()) if daily_watchlist else set()
        tickers = [t for t in tickers if t in ever_listed]
    for ticker in tickers:
        df = load_intraday(ticker, intraday_dir)
        if df is None or df.empty:
            continue

        uptrend_dates = None
        if daily_trend_filter:
            uptrend_dates = _daily_uptrend_dates(ticker, daily_dir)
            if uptrend_dates is None:
                continue

        bars = {
            "open": df["open"].tolist(), "high": df["high"].tolist(),
            "low": df["low"].tolist(), "close": df["close"].tolist(),
            "date": df["date"].tolist(),
        }
        entry_allowed = None
        if entry_window_et is not None:
            entry_allowed = entry_window_mask(df["date"], *entry_window_et)
        for trade in find_smc_long_trades(
            bars, time_window_bars, tp1_fraction, swing_window, require_confirmed_trend,
            force_close_same_day, slippage_bps, post_tp1_stop_fraction, exit_fully_at_tp1,
            entry_fill, require_ob_reclaim, exit_fill, tp1_resting_limit, entry_allowed,
        ):
            entry_day = trade["entry_date"].date()
            if uptrend_dates is not None and entry_day not in uptrend_dates:
                continue
            if daily_watchlist is not None and ticker not in daily_watchlist.get(entry_day, ()):
                continue
            candidates.append((trade["entry_date"], ticker, trade))

    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates


def simulate_smc_portfolio(
    candidates: list[tuple],
    initial_capital: float,
    risk_pct: float = 1.0,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    max_concurrent_positions: int | None = DEFAULT_MAX_CONCURRENT_POSITIONS,
    reactive_derisk_window: int | None = None,
    reactive_derisk_pf_threshold: float = DEFAULT_REACTIVE_DERISK_PF_THRESHOLD,
    reactive_derisk_size_mult: float = DEFAULT_REACTIVE_DERISK_SIZE_MULT,
    commission_per_share: float | None = None,
    commission_min: float = portfolio.DEFAULT_COMMISSION_MIN,
    allow_fractional_shares: bool = False,
    fractional_commission_pct: float = portfolio.DEFAULT_FRACTIONAL_COMMISSION_PCT,
    fractional_commission_min: float = portfolio.DEFAULT_FRACTIONAL_COMMISSION_MIN,
) -> dict:
    """Phase 2 (see module docstring): the portfolio-level chronological
    merge over signals already produced by build_smc_candidates(). Returns
    {"trades": [...], "equity_curve": [...]}; `trades` uses the same
    schema as trade.CSV_HEADER plus an extra `reason` field (entry/tp1/
    stop/new_high_exit/end_of_data/same_day_force_close), same convention
    as backtest/engine.py's trade rows.

    max_concurrent_positions: each open position sizes itself against
        CURRENT equity independently, with no check on capital already
        committed to other positions open at the same time -- so this
        cap controls more than "how many symbols at once," it bounds
        worst-case aggregate exposure. At max_position_pct=50%, going
        from 2 to 5 concurrent positions was measured to raise worst-case
        simultaneous notional exposure from ~110-136% of capital to
        ~213-336% (real effective leverage), for a return that wasn't
        consistently better. 2 was the smallest value that still let
        returns compound at a healthy pace across all three backtested
        years while keeping that worst case close to 1x.

    reactive_derisk_window: NOT part of the literal Level 1 spec -- a
        reactive (not predictive) risk control. When set, every new
        entry's position size is scaled by reactive_derisk_size_mult
        whenever the trailing profit factor over the last
        reactive_derisk_window CLOSED round-trip trades (using only
        trades already closed as of that entry's own date -- no
        lookahead) is below reactive_derisk_pf_threshold. Left None (the
        default), sizing is unaffected. Motivated by year-over-year edge
        decay that broad market-breadth/momentum measures failed to
        explain in advance -- this instead responds to the strategy's
        own realized results rather than betting on an external regime
        signal. A grid sweep across three backtested years (smoothed
        against neighboring parameter values to avoid picking an
        isolated lucky combo) found window=200 alongside the pf_threshold/
        size_mult defaults above to be the strongest, most consistent
        setting -- but with only one weak year in that sweep, treat this
        as a reasonable starting point, not a proven-optimal one.
    commission_per_share: $/share commission (see portfolio.commission),
        charged once per FILL (the entry BUY and every SELL fill, so a
        tp1 + final exit pays it twice on the way out). Left None (the
        default), no commission is modeled at all -- every result in this
        module's earlier design/tuning work was on a zero-cost basis.
    allow_fractional_shares: see portfolio.position_size. Defaults to
        False (whole shares -- the live bot's actual current capability).
        Small accounts often can't afford even 1 whole share of many
        S&P 500 names under the risk/position-size caps, which both
        shrinks the effective tradeable universe AND makes the handful
        of positions that DO execute tiny enough that a flat per-order
        commission minimum dominates their economics. When True, fills
        are costed via fractional_commission_pct/_min (IBKR's ACTUAL
        published fractional-share schedule -- 1% of trade value, $0.01
        minimum -- confirmed against IBKR's own commissions pages,
        2026-04) instead of commission_per_share/commission_min, since
        real fractional fills are billed completely differently, not
        just "the per-share rate applied to a non-integer qty". Needs
        real broker support to ever go live: IBKR does offer fractional
        shares, but this codebase's own order-placement path
        (broker/ibkr_client.py) is whole-shares-only today -- this flag
        only affects the backtest.
    """
    equity = float(initial_capital)
    open_symbols: set[str] = set()
    remaining_fills: dict[str, int] = {}
    fill_heap: list[tuple] = []  # (fill_date, seq, symbol, qty, entry_price, R, fill)
    counter = itertools.count()
    trades_out: list[dict] = []
    equity_curve: list[dict] = []
    order_id = 0

    # Per-round-trip realized pnl, one entry per fully-closed position, in
    # the order each one closes -- only used when reactive_derisk_window
    # is set (see docstring).
    pending_trade_pnl: dict[str, float] = {}
    closed_trade_pnls: list[float] = []

    def _drain_fills_before(cutoff_date):
        nonlocal equity, order_id
        while fill_heap and fill_heap[0][0] <= cutoff_date:
            fill_date, _, symbol, qty, entry_price, fill = heapq.heappop(fill_heap)
            fill_pnl = (fill["price"] - entry_price) * qty
            equity += fill_pnl
            if commission_per_share is not None:
                if allow_fractional_shares:
                    equity -= portfolio.fractional_commission(
                        qty, fill["price"], fractional_commission_pct, fractional_commission_min,
                    )
                else:
                    equity -= portfolio.commission(qty, commission_per_share, commission_min)
            order_id += 1
            trades_out.append(_trade_row(fill_date, symbol, "SELL", qty, fill["price"], order_id, fill["reason"]))
            remaining_fills[symbol] -= 1
            if reactive_derisk_window is not None:
                pending_trade_pnl[symbol] = pending_trade_pnl.get(symbol, 0.0) + fill_pnl
            if remaining_fills[symbol] <= 0:
                open_symbols.discard(symbol)
                del remaining_fills[symbol]
                if reactive_derisk_window is not None:
                    closed_trade_pnls.append(pending_trade_pnl.pop(symbol))
            equity_curve.append({"timestamp": fill_date.isoformat(), "equity": equity})

    def _size_multiplier() -> float:
        if reactive_derisk_window is None or len(closed_trade_pnls) < reactive_derisk_window:
            return 1.0
        window = closed_trade_pnls[-reactive_derisk_window:]
        gains = sum(p for p in window if p > 0)
        losses = -sum(p for p in window if p < 0)
        profit_factor = float("inf") if losses == 0 else gains / losses
        return reactive_derisk_size_mult if profit_factor < reactive_derisk_pf_threshold else 1.0

    for entry_date, symbol, trade in candidates:
        _drain_fills_before(entry_date)

        if symbol in open_symbols:
            continue
        if max_concurrent_positions is not None and len(open_symbols) >= max_concurrent_positions:
            continue

        entry_price = trade["entry_price"]
        # initial_stop_price, NOT stop_price: find_smc_long_trades moves
        # stop_price to breakeven in place once TP1 fills, so by the time
        # this pre-built candidate reaches the portfolio sim, stop_price
        # no longer reflects the entry-time risk distance for any trade
        # that went on to hit TP1 (see smc_signals.find_smc_long_trades).
        stop_price = trade["initial_stop_price"]
        size = portfolio.position_size(
            equity, risk_pct, entry_price, stop_price, max_position_pct, allow_fractional_shares,
        )
        size = size * _size_multiplier()
        size = round(size, 6) if allow_fractional_shares else int(size)
        min_size = 1e-6 if allow_fractional_shares else 1
        if size < min_size:
            continue

        open_symbols.add(symbol)
        order_id += 1
        if commission_per_share is not None:
            if allow_fractional_shares:
                equity -= portfolio.fractional_commission(
                    size, entry_price, fractional_commission_pct, fractional_commission_min,
                )
            else:
                equity -= portfolio.commission(size, commission_per_share, commission_min)
        trades_out.append(_trade_row(entry_date, symbol, "BUY", size, entry_price, order_id, "entry"))
        equity_curve.append({"timestamp": entry_date.isoformat(), "equity": equity})

        fills = trade["fills"]
        remaining_fills[symbol] = len(fills)
        remaining_qty = size
        for i, fill in enumerate(fills):
            if i == len(fills) - 1:
                fill_qty = remaining_qty
            elif allow_fractional_shares:
                fill_qty = round(size * fill["qty_fraction"], 6)
            else:
                fill_qty = round(size * fill["qty_fraction"])
            fill_qty = min(fill_qty, remaining_qty)
            remaining_qty -= fill_qty
            if fill_qty < min_size:
                remaining_fills[symbol] -= 1
                continue
            heapq.heappush(fill_heap, (fill["date"], next(counter), symbol, fill_qty, entry_price, fill))

        if remaining_fills[symbol] <= 0:
            open_symbols.discard(symbol)
            del remaining_fills[symbol]

    # Drain everything left (no more candidate entries to interleave with).
    while fill_heap:
        _drain_fills_before(fill_heap[0][0])

    return {"trades": trades_out, "equity_curve": equity_curve}


def run_smc_backtest(
    tickers: list[str],
    initial_capital: float,
    risk_pct: float = 1.0,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    max_concurrent_positions: int | None = DEFAULT_MAX_CONCURRENT_POSITIONS,
    intraday_dir: Path = INTRADAY_DIR,
    time_window_bars: int = DEFAULT_TIME_WINDOW_BARS,
    tp1_fraction: float = DEFAULT_TP1_FRACTION,
    swing_window: int = DEFAULT_SWING_WINDOW,
    require_confirmed_trend: bool = False,
    daily_trend_filter: bool = False,
    daily_dir: Path = DAILY_DIR,
    force_close_same_day: bool = False,
    slippage_bps: dict | float | None = None,
    post_tp1_stop_fraction: float = DEFAULT_POST_TP1_STOP_FRACTION,
    exit_fully_at_tp1: bool = False,
    entry_fill: str = DEFAULT_ENTRY_FILL,
    require_ob_reclaim: bool = False,
    exit_fill: str = DEFAULT_EXIT_FILL,
    tp1_resting_limit: bool = False,
    reactive_derisk_window: int | None = None,
    reactive_derisk_pf_threshold: float = DEFAULT_REACTIVE_DERISK_PF_THRESHOLD,
    reactive_derisk_size_mult: float = DEFAULT_REACTIVE_DERISK_SIZE_MULT,
    commission_per_share: float | None = None,
    commission_min: float = portfolio.DEFAULT_COMMISSION_MIN,
    allow_fractional_shares: bool = False,
    fractional_commission_pct: float = portfolio.DEFAULT_FRACTIONAL_COMMISSION_PCT,
    fractional_commission_min: float = portfolio.DEFAULT_FRACTIONAL_COMMISSION_MIN,
) -> dict:
    """Convenience one-shot wrapper: build_smc_candidates() +
    simulate_smc_portfolio() in a single call, for callers that don't need
    to reuse the (expensive) candidate-generation step across multiple
    portfolio-level parameter variations. See those two functions for the
    full parameter docs.

    The execution-spec arguments (entry_fill, require_ob_reclaim,
    exit_fill, tp1_resting_limit) were not forwarded until 2026-08-30, so
    this wrapper silently produced whatever the signal layer defaulted to
    and gave callers no way to choose. That is the same defect as the
    default itself, one layer up."""
    candidates = build_smc_candidates(
        tickers, intraday_dir, time_window_bars, tp1_fraction, swing_window, require_confirmed_trend,
        daily_trend_filter, daily_dir, force_close_same_day, slippage_bps,
        post_tp1_stop_fraction, exit_fully_at_tp1,
        entry_fill, require_ob_reclaim, exit_fill, tp1_resting_limit,
    )
    return simulate_smc_portfolio(
        candidates, initial_capital, risk_pct, max_position_pct, max_concurrent_positions,
        reactive_derisk_window, reactive_derisk_pf_threshold, reactive_derisk_size_mult,
        commission_per_share, commission_min, allow_fractional_shares,
        fractional_commission_pct, fractional_commission_min,
    )
