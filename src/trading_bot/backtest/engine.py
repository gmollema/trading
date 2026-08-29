"""Event-driven simulation engine for the Trend Join Long backtest.

Two phases:
  1. Vectorized per-symbol precompute (via backtest.data/backtest.filters)
     of daily/intraday context and a per-bar entry_signal, done
     independently per symbol.
  2. A single chronological event loop merging all symbols' bars in time
     order -- required because max_concurrent_positions, daily trade count,
     and portfolio equity are GLOBAL state shared across symbols. At each
     timestamp: manage any open positions with a bar now, then (inside the
     entry window, under the position/day caps) open new positions for
     symbols with a passing entry signal, and force-close everything
     remaining once the force-close time is reached.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from trading_bot.backtest import filters, portfolio
from trading_bot.backtest.data import DAILY_DIR, INTRADAY_DIR, build_symbol_frame
from trading_bot.live import get_market_status

DEFAULT_MAX_POSITION_PCT = 10.0


def _entry_signal(row: dict, rules: dict) -> bool:
    daily_ctx = {
        "prior_day_high": row["prior_day_high"],
        "prior_day_close": row["prior_day_close"],
        "sma200": row["sma200"],
    }
    intraday_ctx = {"today_hod": row["running_hod"], "rvol": row["rvol"]}
    passed, _reasons = filters.evaluate_entry(daily_ctx, intraday_ctx, row["close"], rules)
    return passed


def load_raw_universe(
    tickers: list[str],
    rvol_lookback_days: int,
    daily_dir: Path = DAILY_DIR,
    intraday_dir: Path = INTRADAY_DIR,
) -> pd.DataFrame:
    """Load + precompute daily/intraday context for every ticker, WITHOUT
    `entry_signal` (which depends on filter thresholds, not just
    rvol_lookback_days). This is the expensive step (reading every
    symbol's CSVs, rolling/pivot computations) -- cache the result and
    reuse it across many rules variations that share the same
    rvol_lookback_days via add_entry_signals(), instead of rebuilding it
    per variation (e.g. a parameter sweep)."""
    frames = []
    for ticker in tickers:
        frame = build_symbol_frame(ticker, rvol_lookback_days, daily_dir, intraday_dir)
        if frame is not None and not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    universe = pd.concat(frames, ignore_index=True)
    return universe.sort_values(["date", "symbol"]).reset_index(drop=True)


def add_entry_signals(raw_universe: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Add `entry_signal` (bool) and `gap_pct` columns to a universe from
    load_raw_universe(), for a specific rules config. `gap_pct` is used
    only to break ties when more than one entry candidate appears on the
    same tick, largest gap first -- mirrors morning_prefilter.py's own
    ranking. Cheap relative to load_raw_universe(), so re-run this per
    rules variation rather than reloading from disk each time."""
    if raw_universe.empty:
        return raw_universe.copy()

    universe = raw_universe.copy()

    # Plain dict records (not DataFrame.apply) so filters.evaluate_entry --
    # the single source of truth for D1-D3/I2-I3 -- stays a pure per-row
    # function without needing a duplicated vectorized reimplementation.
    records = universe.to_dict("records")
    universe["entry_signal"] = [_entry_signal(r, rules) for r in records]
    universe["gap_pct"] = (universe["close"] - universe["prior_day_close"]) / universe["prior_day_close"] * 100
    return universe


def _build_universe(
    tickers: list[str], rules: dict, daily_dir: Path = DAILY_DIR, intraday_dir: Path = INTRADAY_DIR
) -> pd.DataFrame:
    rvol_lookback_days = rules["intraday_filters"]["I3_rvol_lookback_days"]
    raw_universe = load_raw_universe(tickers, rvol_lookback_days, daily_dir, intraday_dir)
    return add_entry_signals(raw_universe, rules)


def build_day_frames(universe: pd.DataFrame) -> dict[tuple[str, object], pd.DataFrame]:
    """Per-(symbol, day) frames, timestamp-indexed, for O(1) bar lookups and
    cheaply slicing "today's lows so far" during the event loop.

    Depends only on the raw bar data (symbol/trading_date/date/OHLCV), NOT
    on entry_signal/gap_pct -- so it's identical across every rules
    variation sharing the same underlying universe. Cache and reuse it
    (e.g. via run_backtest's `day_frames` param) across a parameter sweep
    instead of rebuilding it per combo."""
    return {
        key: group.set_index("date").sort_index()
        for key, group in universe.groupby(["symbol", "trading_date"], sort=False)
    }


SLIPPAGE_REASONS = ("entry", "stop", "partial_profit", "force_close",
                    "max_hold_reached", "end_of_data")


def _normalize_slippage(slippage_bps: dict | float | None) -> dict:
    """Accept None, a scalar for every leg, or a partial dict keyed by
    fill reason. Unknown keys are rejected rather than ignored -- a typo'd
    reason would otherwise read as "no slippage on that leg"."""
    if slippage_bps is None:
        return {r: 0.0 for r in SLIPPAGE_REASONS}
    if isinstance(slippage_bps, (int, float)):
        return {r: float(slippage_bps) for r in SLIPPAGE_REASONS}
    unknown = set(slippage_bps) - set(SLIPPAGE_REASONS)
    if unknown:
        raise ValueError(f"unknown slippage_bps keys: {sorted(unknown)}; expected {list(SLIPPAGE_REASONS)}")
    return {r: float(slippage_bps.get(r, 0.0)) for r in SLIPPAGE_REASONS}


def _trade_row(
    timestamp, symbol: str, side: str, qty: float, price: float, order_id: int, reason=None, r_multiple=None
) -> dict:
    """Build one trade row. The base fields match trade.CSV_HEADER exactly
    (so compute_perf.py can read the persisted CSV directly); `reason` and
    `r_multiple` are extra, analysis-only fields carried on the in-memory
    dict -- callers that write this to a CSV using the fixed CSV_HEADER
    fieldnames must pass extrasaction="ignore" (see cli/backtest.py)."""
    return {
        "timestamp_iso": timestamp.tz_convert("UTC").isoformat(),
        "symbol": symbol,
        "side": side,
        "size": int(qty),
        "fill_price": round(float(price), 4),
        "order_id": order_id,
        "status": "Filled",
        "reason": reason,
        "r_multiple": round(r_multiple, 3) if r_multiple is not None else None,
    }


def run_backtest(
    tickers: list[str],
    rules: dict,
    initial_capital: float,
    start_date=None,
    end_date=None,
    daily_dir: Path = DAILY_DIR,
    intraday_dir: Path = INTRADAY_DIR,
    force_close_daily: bool = True,
    max_hold_days: int | None = None,
    overnight_size_reduction_pct: float | None = None,
    universe: pd.DataFrame | None = None,
    day_frames: dict[tuple[str, object], pd.DataFrame] | None = None,
    fill_spec: str = portfolio.DEFAULT_FILL_SPEC,
    slippage_bps: dict | float | None = None,
    commission_per_share: float | None = None,
    commission_min: float = portfolio.DEFAULT_COMMISSION_MIN,
) -> dict:
    """Simulate rules.json's strategy across `tickers`' cached history.

    Returns {"trades": [...], "equity_curve": [...], "overnight_stop_snapshots":
    [...]}. `trades` rows use the same schema as trade.CSV_HEADER
    (timestamp_iso, symbol, side, size, fill_price, order_id, status) so
    existing tooling (compute_perf.py) can read them directly.
    `overnight_stop_snapshots` records, once per (symbol, day) a position
    was already open going into that day, {"date", "symbol", "stop_price",
    "entry_price", "qty"} as they stood before that day's first bar -- for
    post-hoc gap-risk analysis against real premarket data; it has no
    effect on the simulation and is empty unless a position is ever held
    overnight (i.e. always empty when force_close_daily=True).

    Args:
        fill_spec: where each leg fills -- see portfolio.FILL_SPECS.
            "level" (the default) is what every gap-and-go figure in this
            repo was produced on: the entry books the signal bar's own
            close, and the market-order exits book theirs. Nothing
            executes either. "next_open" prices each leg at what its own
            order type gets, the same correction backtest/smc_signals.py
            went through on 2026-08-29.
        slippage_bps: adverse slippage per leg -- None or 0 for the
            frictionless fills this engine has always assumed, a scalar
            for one rate everywhere, or a dict keyed by fill reason
            ("entry", "stop", "partial_profit", "force_close",
            "max_hold_reached", "end_of_data"). Levels still TRIGGER
            unchanged; only the recorded fill price moves, adversely.
        commission_per_share / commission_min: IBKR's whole-share
            schedule, charged once per fill. Left None (the default), no
            commission is modelled at all -- which is what every figure
            this engine has produced so far assumed.
        force_close_daily: when True (the live bot's actual behavior),
            every open position is flattened once the force-close time
            (rules.json's time_filter.force_close_et) is reached each day.
            Set False to let positions ride past that cutoff and continue
            being managed on subsequent days -- an experiment, not the
            live bot's behavior; positions still never open new during the
            force-close window itself, only the flatten-everything action
            is skipped.
        max_hold_days: when set, force-close a position (reason
            "max_hold_reached") once it has been open this many calendar
            days or more, regardless of force_close_daily -- an experiment
            for capping overnight-gap exposure while still giving the
            partial/trailing exit logic more than one session to work.
        overnight_size_reduction_pct: when set (e.g. 0.5), sell this
            fraction of a position's remaining qty at the prior day's last
            close, right before it would be held overnight -- an
            experiment for shrinking the KNOWN overnight gap-risk exposure
            (measured separately against real premarket data) without
            capping how long a position can run. Applied once per (symbol,
            day) transition, using floor() same as other size math; a
            no-op once qty would round to 0.
        universe: a pre-built DataFrame from load_raw_universe() +
            add_entry_signals(rules), to skip rebuilding it from `tickers`
            (e.g. across many rules variations in a parameter sweep that
            share the same cached raw universe). When given, `tickers` is
            ignored for loading purposes (still required as a parameter).
        day_frames: a pre-built dict from build_day_frames(universe), to
            skip rebuilding it -- safe to reuse across every rules
            variation sharing the same `universe` (see build_day_frames'
            docstring). Only valid together with `universe`; ignored if
            `universe` is None. NOTE: must be rebuilt if start_date/
            end_date narrow the universe, since a pre-built day_frames
            would still include out-of-range days.
    """
    if universe is None:
        universe = _build_universe(tickers, rules, daily_dir, intraday_dir)
        day_frames = None  # a passed-in day_frames without a matching universe would be stale
    if universe.empty:
        return {"trades": [], "equity_curve": [], "overnight_stop_snapshots": []}

    date_filtered = start_date is not None or end_date is not None
    if start_date is not None:
        universe = universe[universe["trading_date"] >= start_date]
    if end_date is not None:
        universe = universe[universe["trading_date"] <= end_date]
    universe = universe.reset_index(drop=True)
    if universe.empty:
        return {"trades": [], "equity_curve": [], "overnight_stop_snapshots": []}

    risk_pct = rules["risk"]["max_risk_per_trade_pct"]
    max_position_pct = rules["risk"].get("max_position_size_pct_of_portfolio", DEFAULT_MAX_POSITION_PCT)
    max_concurrent = rules["risk"].get("max_concurrent_positions")
    max_trades_per_day = rules["risk"].get("max_trades_per_day")

    if day_frames is None or date_filtered:
        day_frames = build_day_frames(universe)

    equity = float(initial_capital)
    slip = _normalize_slippage(slippage_bps)

    def _charge(qty: float) -> float:
        if commission_per_share is None:
            return 0.0
        return portfolio.commission(qty, commission_per_share, commission_min)

    open_positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[dict] = []
    overnight_stop_snapshots: list[dict] = []
    order_id = 0
    trades_today = 0
    current_day = None

    for timestamp, tick_rows in universe.groupby("date", sort=True):
        # Same gate the live bot uses, now reading the same rules file --
        # the entry window was hardcoded in it until 2026-08-29, so this
        # backtest silently ignored rules.json's time_filter too.
        status = get_market_status(timestamp, rules)
        trading_date = timestamp.date()

        if trading_date != current_day:
            previous_day = current_day
            current_day = trading_date
            trades_today = 0

            if overnight_size_reduction_pct and previous_day is not None:
                for symbol in list(open_positions.keys()):
                    pos = open_positions[symbol]
                    reduce_qty = math.floor(pos["qty"] * overnight_size_reduction_pct)
                    prev_day_frame = day_frames.get((symbol, previous_day))
                    if reduce_qty < 1 or prev_day_frame is None or prev_day_frame.empty:
                        continue

                    prev_close_ts = prev_day_frame.index[-1]
                    price = float(prev_day_frame.iloc[-1]["close"])
                    order_id += 1
                    equity += (price - pos["entry_price"]) * reduce_qty
                    r_multiple = (price - pos["entry_price"]) / pos["R"] if pos["R"] else None
                    trades.append(
                        _trade_row(
                            prev_close_ts, symbol, "SELL", reduce_qty, price, order_id,
                            reason="overnight_size_reduction", r_multiple=r_multiple,
                        )
                    )
                    open_positions[symbol] = {**pos, "qty": pos["qty"] - reduce_qty}

            # Snapshot each still-open position's stop as it stood BEFORE
            # today's first bar -- i.e. what was resting overnight (after
            # any overnight size reduction above). Used only for post-hoc
            # gap-risk analysis (comparing against real premarket bars);
            # has no effect on the simulation itself.
            for symbol, pos in open_positions.items():
                overnight_stop_snapshots.append(
                    {
                        "date": trading_date.isoformat(),
                        "symbol": symbol,
                        "stop_price": pos["current_stop_price"],
                        "entry_price": pos["entry_price"],
                        "qty": pos["qty"],
                    }
                )

        # --- manage open positions with a bar at this tick ---
        for symbol in list(open_positions.keys()):
            day_frame = day_frames.get((symbol, trading_date))
            if day_frame is None or timestamp not in day_frame.index:
                continue

            bar = day_frame.loc[timestamp]
            pos = open_positions[symbol]

            if max_hold_days is not None and (trading_date - pos["entry_date"]).days >= max_hold_days:
                price = portfolio.slipped(
                    portfolio.market_fill_price(bar, fill_spec),
                    slip.get("max_hold_reached", 0.0), "sell",
                )
                fill = portfolio.close_position(pos, price, reason="max_hold_reached")
                order_id += 1
                equity += (fill["price"] - pos["entry_price"]) * fill["qty"]
                r_multiple = (fill["price"] - pos["entry_price"]) / pos["R"] if pos["R"] else None
                trades.append(
                    _trade_row(
                        timestamp, symbol, "SELL", fill["qty"], fill["price"], order_id,
                        reason=fill["reason"], r_multiple=r_multiple,
                    )
                )
                del open_positions[symbol]
                continue

            recent_lows = day_frame.loc[:timestamp, "low"].tolist()

            updated_pos, fills = portfolio.manage_position(
                pos,
                {
                    "low": float(bar["low"]), "close": float(bar["close"]),
                    "open": float(bar["open"]), "next_open": bar["next_open"],
                },
                recent_lows,
                rules["exit"],
                fill_spec,
            )
            for fill in fills:
                order_id += 1
                fill["price"] = portfolio.slipped(fill["price"], slip.get(fill["reason"], 0.0), "sell")
                equity += (fill["price"] - pos["entry_price"]) * fill["qty"] - _charge(fill["qty"])
                r_multiple = (fill["price"] - pos["entry_price"]) / pos["R"] if pos["R"] else None
                trades.append(
                    _trade_row(
                        timestamp, symbol, "SELL", fill["qty"], fill["price"], order_id,
                        reason=fill["reason"], r_multiple=r_multiple,
                    )
                )

            if updated_pos is None:
                del open_positions[symbol]
            else:
                open_positions[symbol] = updated_pos

        # --- force-close everything remaining once the cutoff is reached ---
        if status == "force_close" and force_close_daily:
            for symbol in list(open_positions.keys()):
                day_frame = day_frames.get((symbol, trading_date))
                if day_frame is None or timestamp not in day_frame.index:
                    continue
                pos = open_positions.pop(symbol)
                price = portfolio.slipped(
                    portfolio.market_fill_price(day_frame.loc[timestamp], fill_spec),
                    slip.get("force_close", 0.0), "sell",
                )
                fill = portfolio.close_position(pos, price)
                order_id += 1
                equity += (fill["price"] - pos["entry_price"]) * fill["qty"]
                r_multiple = (fill["price"] - pos["entry_price"]) / pos["R"] if pos["R"] else None
                trades.append(
                    _trade_row(
                        timestamp, symbol, "SELL", fill["qty"], fill["price"], order_id,
                        reason=fill["reason"], r_multiple=r_multiple,
                    )
                )

        # --- open new positions ---
        elif status == "ok" and not (max_trades_per_day is not None and trades_today >= max_trades_per_day):
            candidates = tick_rows[tick_rows["entry_signal"] & ~tick_rows["symbol"].isin(open_positions.keys())]
            candidates = candidates.sort_values("gap_pct", ascending=False)

            for _, row in candidates.iterrows():
                if max_concurrent is not None and len(open_positions) >= max_concurrent:
                    break
                if max_trades_per_day is not None and trades_today >= max_trades_per_day:
                    break

                symbol = row["symbol"]
                today_lod = float(row["running_lod"])
                initial_stop = portfolio.initial_stop_from_lod(today_lod)

                # The signal is computed from this bar; the order goes in
                # once it closes and fills on the next one. Under "level"
                # that distinction is erased and the entry books the very
                # close it was decided on.
                raw_entry = portfolio.market_fill_price(row, fill_spec)
                if raw_entry != raw_entry:  # no same-session bar to fill on
                    continue
                price = portfolio.slipped(raw_entry, slip.get("entry", 0.0), "buy")
                if price <= initial_stop:
                    # No positive risk-per-share to size against. Live this
                    # is a fill followed at once by a stop-out rather than a
                    # skipped trade -- the one place this model flatters
                    # itself, and only reachable once the fill is priced off
                    # a later bar than the signal.
                    continue

                size = portfolio.position_size(equity, risk_pct, price, initial_stop, max_position_pct)
                if size < 1:
                    continue

                equity -= _charge(size)
                new_pos = portfolio.open_position(symbol, price, today_lod, size)
                new_pos["entry_date"] = trading_date
                open_positions[symbol] = new_pos
                order_id += 1
                trades_today += 1
                trades.append(_trade_row(timestamp, symbol, "BUY", size, price, order_id, reason="entry"))

        equity_curve.append({"timestamp": timestamp.isoformat(), "equity": equity})

    # Safety net: force-close anything still open at the end of the
    # available data. The real strategy always force-closes by 15:51 ET
    # daily, so this should only ever bite at the very tail of history --
    # but with force_close_daily=False, many positions can still be open
    # here, so equity_curve must reflect their P&L too, not just the main
    # loop's last tick.
    for symbol, pos in list(open_positions.items()):
        symbol_rows = universe[universe["symbol"] == symbol]
        if symbol_rows.empty:
            continue
        last_row = symbol_rows.iloc[-1]
        fill = portfolio.close_position(
            pos, float(last_row["close"]), reason="end_of_data",
        )
        order_id += 1
        equity += (fill["price"] - pos["entry_price"]) * fill["qty"]
        r_multiple = (fill["price"] - pos["entry_price"]) / pos["R"] if pos["R"] else None
        trades.append(
            _trade_row(
                last_row["date"], symbol, "SELL", fill["qty"], fill["price"], order_id,
                reason=fill["reason"], r_multiple=r_multiple,
            )
        )
        equity_curve.append({"timestamp": last_row["date"].isoformat(), "equity": equity})

    return {"trades": trades, "equity_curve": equity_curve, "overnight_stop_snapshots": overnight_stop_snapshots}
