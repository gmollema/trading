"""Autonomous trading cycle for the IBKR paper-trading bot.

Intended to run every 5 minutes via Windows Task Scheduler, all day.

Design note: the market-status gate below runs BEFORE any heavy imports
(yfinance, ib_async, pandas, numpy, dotenv) so that weekend / too_early /
closed cycles exit in well under 1 second - this is what makes "every 5
minutes, all day, every day" cheap to run unattended.

IMPORTANT CAVEATS (read before relying on this for anything but paper
trading dry-runs):
  - The D1-D3 / I1-I3 filter implementations below are a reasonable but
    opinionated interpretation of rules.json - rules.json specifies WHAT
    to check, not exactly HOW to compute premarket high, RVOL, or SMA200
    from yfinance data. Validate these against known-good days before
    trusting them.
  - yfinance premarket/RVOL coverage varies by ticker and can be sparse;
    tickers with insufficient data are skipped (fail-closed), not
    force-passed.
  - ib_async's exact Fill/Order field names can vary slightly by version;
    the stop-out matching and order-cancellation logic below was written
    against the documented API shape but has not been run against a live
    session. Watch logs/cycle_errors.log and safety-check-log.json on the
    first several real runs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

WATCHLIST_PATH = Path("watchlist.txt")
POSITIONS_PATH = Path("open_positions.json")
RULES_PATH = Path("rules.json")
TRADES_CSV_PATH = Path("trades.csv")
SAFETY_LOG_PATH = Path("safety-check-log.json")
HEARTBEAT_PATH = Path("heartbeat.json")
LOGS_DIR = Path("logs")
CYCLE_ERRORS_LOG = LOGS_DIR / "cycle_errors.log"

ET = ZoneInfo("America/New_York")

SETTLED_ORDER_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}
FAILED_SELL_STATUSES = {"Cancelled", "ApiCancelled", "Inactive", "Rejected"}
SUBPROCESS_TIMEOUT_SECS = 30
CANCEL_CONFIRM_TIMEOUT_SECS = 5


# --------------------------------------------------------------------------
# STEP 1: TIME GATE (pure stdlib, cheap - must stay fast)
# --------------------------------------------------------------------------

def get_market_status(now_et: datetime) -> str:
    """Returns one of: weekend, too_early, closed, manage_only, force_close, ok."""
    if now_et.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return "weekend"

    hm = now_et.strftime("%H:%M")

    if hm < "10:00":
        return "too_early"
    if hm > "16:00":
        return "closed"
    if "10:00" <= hm < "10:05":
        return "manage_only"
    if "10:05" <= hm < "15:30":
        return "ok"
    if "15:30" <= hm < "15:51":
        return "manage_only"
    if "15:51" <= hm <= "16:00":
        return "force_close"
    return "closed"  # unreachable, defensive fallback


def _fast_exit_check() -> str | None:
    """Cheap gate check performed before any heavy imports."""
    now_et = datetime.now(ET)
    status = get_market_status(now_et)
    if status in ("weekend", "too_early", "closed"):
        return status
    return None


# --- Fast-path gate: this MUST run before heavy imports below. ---
_EARLY_EXIT_STATUS = _fast_exit_check()
if _EARLY_EXIT_STATUS is not None:
    print(f"Market status: {_EARLY_EXIT_STATUS}. Exiting.")
    sys.exit(0)

# --------------------------------------------------------------------------
# Heavy imports - only reached once we know the market is plausibly open.
# --------------------------------------------------------------------------
import numpy as np  # noqa: E402
import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from ib_async import MarketOrder, Stock, StopOrder  # noqa: E402

from trading_bot.broker.ibkr_client import IBKRClient  # noqa: E402
from trading_bot.util.heartbeat import write_heartbeat  # noqa: E402
from trading_bot.util.notifier import notify  # noqa: E402


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def _cast_bools(obj):
    """Recursively cast bool/numpy bool_ to plain Python bool for json.dumps."""
    if isinstance(obj, dict):
        return {k: _cast_bools(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_bools(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def log_event(event: dict) -> None:
    event = dict(event)
    event.setdefault("timestamp_iso", datetime.now(timezone.utc).isoformat())
    safe_event = _cast_bools(event)
    with SAFETY_LOG_PATH.open("a") as f:
        f.write(json.dumps(safe_event) + "\n")


# --------------------------------------------------------------------------
# STEP 2 / 5: STATE LOAD / SAVE
# --------------------------------------------------------------------------

def load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    try:
        with POSITIONS_PATH.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_positions(positions: list[dict]) -> None:
    tmp_path = POSITIONS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(_cast_bools(positions), f, indent=2)
    os.replace(tmp_path, POSITIONS_PATH)


# --------------------------------------------------------------------------
# Ticker format helpers
# --------------------------------------------------------------------------

def ibkr_to_yahoo(ticker: str) -> str:
    return ticker.replace(" ", "-")


# --------------------------------------------------------------------------
# STEP 3: CHECK STOP-OUTS
# --------------------------------------------------------------------------

def check_stop_outs(ib, positions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Match fills to positions by order_id (NOT quantity - quantity matching
    causes false stop-outs after partials). Returns (remaining, events).

    order_id alone is not sufficient: IBKR order IDs are tracked locally per
    client connection, and two independent connections (e.g. this process's
    own connection placing the stop, vs. trade.py's subprocess placing the
    entry BUY) can end up assigning the SAME order ID to two DIFFERENT
    orders (observed in practice - a fresh client's first order can collide
    with another fresh client's first order). Matching order_id alone would
    then misattribute the entry's own BUY fill as a stop-out. A stop-loss
    exit is always a SELL for this long-only strategy, so requiring
    execution.side == "SLD" rules that out regardless of ID collisions.
    """
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

    try:
        fills = ib.fills()
    except Exception as e:
        log_event({"event": "fills_query_failed", "error": str(e)})
        fills = []

    recent_fills = []
    for f in fills:
        fill_time = getattr(f, "time", None)
        if fill_time is None:
            recent_fills.append(f)
            continue
        # Normalize to aware UTC if needed for comparison.
        if fill_time.tzinfo is None:
            fill_time = fill_time.replace(tzinfo=timezone.utc)
        if fill_time >= one_hour_ago:
            recent_fills.append(f)

    remaining = []
    events = []
    for pos in positions:
        stop_order_id = pos.get("stop_order_id")
        matched = None
        for f in recent_fills:
            order_id = getattr(f.execution, "orderId", None)
            side = getattr(f.execution, "side", None)
            if stop_order_id is not None and order_id == stop_order_id and side == "SLD":
                matched = f
                break
        if matched is not None:
            fill_price = float(getattr(matched.execution, "price", 0.0))
            pnl = (fill_price - pos.get("entry_price", fill_price)) * pos.get("qty", 0)
            events.append(
                {
                    "event": "stopped_out",
                    "symbol": pos["symbol"],
                    "stop_order_id": stop_order_id,
                    "fill_price": fill_price,
                    "pnl": pnl,
                }
            )
            notify(f"STOP {pos['symbol']}", f"exit ${fill_price:.2f}, P&L ${pnl:+.2f}", "default")
        else:
            remaining.append(pos)

    return remaining, events


# --------------------------------------------------------------------------
# Order helpers
# --------------------------------------------------------------------------

def _qualify(ib, symbol: str):
    stock = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(stock)
    return qualified[0] if qualified else stock


def _cancel_stop(ib, stop_order_id) -> bool:
    """Cancel the resting stop order and wait for it to leave the open-orders
    book. Returns True if there was nothing to cancel or the cancellation is
    confirmed; False if it could not be confirmed within
    CANCEL_CONFIRM_TIMEOUT_SECS. Callers must treat False as "the old stop
    may still be live" and avoid placing an overlapping order for the same
    shares (that would risk selling more shares than are held if both fire)."""
    if not stop_order_id:
        return True
    try:
        ib.reqOpenOrders()
        ib.sleep(0.5)
    except Exception:
        pass

    order_to_cancel = None
    for order in list(ib.openOrders()):
        if getattr(order, "orderId", None) == stop_order_id:
            order_to_cancel = order
            break

    if order_to_cancel is None:
        # Already gone (filled or previously cancelled) -- nothing to do.
        return True

    try:
        ib.cancelOrder(order_to_cancel)
    except Exception as e:
        log_event({"event": "cancel_stop_failed", "stop_order_id": stop_order_id, "error": str(e)})
        return False

    deadline = time.time() + CANCEL_CONFIRM_TIMEOUT_SECS
    while time.time() < deadline:
        ib.sleep(0.25)
        still_open = any(getattr(o, "orderId", None) == stop_order_id for o in ib.openOrders())
        if not still_open:
            return True

    log_event({"event": "cancel_stop_unconfirmed", "stop_order_id": stop_order_id})
    return False


def _place_stop(ib, symbol: str, qty: int, stop_price: float) -> int | None:
    contract = _qualify(ib, symbol)
    order = StopOrder("SELL", qty, round(float(stop_price), 2))
    order.outsideRth = True
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    return trade.order.orderId


def _sell_market(ib, symbol: str, qty: int):
    contract = _qualify(ib, symbol)
    order = MarketOrder("SELL", qty)
    order.outsideRth = True
    trade = ib.placeOrder(contract, order)
    deadline = time.time() + 10
    while time.time() < deadline:
        ib.sleep(0.25)
        if trade.orderStatus.status in SETTLED_ORDER_STATUSES or trade.orderStatus.status == "Submitted":
            break
    return trade


def get_current_price(ib, symbol: str) -> float | None:
    contract = _qualify(ib, symbol)
    ticker = ib.reqMktData(contract, "", False, False)
    ib.sleep(2)
    price = ticker.marketPrice()
    if price is None or price != price or price <= 0:  # NaN / missing
        last = ticker.last
        price = last if last and last == last and last > 0 else None
    ib.cancelMktData(contract)
    return float(price) if price else None


# --------------------------------------------------------------------------
# 5-minute bar helpers (yfinance)
# --------------------------------------------------------------------------

def get_5m_bars_today(symbol_ibkr: str):
    symbol_yahoo = ibkr_to_yahoo(symbol_ibkr)
    try:
        bars = yf.Ticker(symbol_yahoo).history(period="2d", interval="5m", auto_adjust=True)
    except Exception:
        return None
    if bars is None or bars.empty:
        return None
    bars.index = bars.index.tz_convert(ET)
    today = datetime.now(ET).date()
    today_bars = bars[bars.index.date == today]
    return today_bars if not today_bars.empty else None


def compute_swing_lows(bars_5m) -> list[float]:
    """A bar's low is a swing low if it's lower than the 2 bars before AND
    the 2 bars after it. Returns swing lows in chronological order."""
    lows = bars_5m["Low"].tolist()
    swing_lows = []
    for i in range(2, len(lows) - 2):
        before = lows[i - 2:i]
        after = lows[i + 1:i + 3]
        if lows[i] < min(before) and lows[i] < min(after):
            swing_lows.append(float(lows[i]))
    return swing_lows


# --------------------------------------------------------------------------
# STEP 4: MANAGE EACH POSITION
# --------------------------------------------------------------------------

def manage_position(ib, pos: dict, rules: dict) -> dict:
    symbol = pos["symbol"]
    entry_price = pos["entry_price"]
    R = pos["R"]
    state = pos["state"]

    price = get_current_price(ib, symbol)
    if price is None:
        log_event({"event": "manage_skip_no_price", "symbol": symbol})
        return pos

    exit_cfg = rules["exit"]

    if state == "pre_breakeven":
        breakeven_trigger_R = exit_cfg["breakeven_trigger_R"]
        partial_trigger_R = exit_cfg["partial_profit_trigger_R"]

        reached_partial = price >= entry_price + partial_trigger_R * R
        reached_breakeven = price >= entry_price + breakeven_trigger_R * R

        if reached_partial:
            # Cancel the OLD full-quantity stop and confirm it's gone BEFORE
            # selling the partial quantity. Selling first would leave the old
            # stop (sized for the full qty) resting alongside the market
            # sell; if it triggered in that window you'd sell more shares
            # than you hold.
            cancel_confirmed = _cancel_stop(ib, pos.get("stop_order_id"))
            if not cancel_confirmed:
                log_event(
                    {
                        "event": "partial_profit_skipped",
                        "symbol": symbol,
                        "reason": "could not confirm old stop cancelled",
                    }
                )
            else:
                partial_qty = math.ceil(pos["qty"] / 3)
                partial_qty = min(partial_qty, pos["qty"])
                trade = _sell_market(ib, symbol, partial_qty)
                sell_status = trade.orderStatus.status

                if sell_status in FAILED_SELL_STATUSES:
                    # The old stop is already gone and the partial sell also
                    # failed -- the position is currently unprotected.
                    # Re-place the stop for the full quantity rather than
                    # leave it naked until the next cycle.
                    log_event(
                        {
                            "event": "partial_sell_failed",
                            "symbol": symbol,
                            "status": sell_status,
                        }
                    )
                    restore_stop_price = pos.get("current_stop_price", pos.get("initial_stop", entry_price))
                    pos["stop_order_id"] = _place_stop(ib, symbol, pos["qty"], restore_stop_price)
                    notify(
                        f"PARTIAL FAILED {symbol}",
                        f"sell rejected (status={sell_status}); stop re-placed @ ${restore_stop_price:.2f}",
                        "high",
                    )
                else:
                    remaining_qty = pos["qty"] - partial_qty
                    # A fast move that jumps straight past both the partial
                    # and breakeven triggers within one poll still gets the
                    # better (breakeven) stop, instead of silently only
                    # applying the partial-level discount stop.
                    new_stop_price = entry_price if reached_breakeven else entry_price * 0.99

                    new_stop_id = None
                    if remaining_qty > 0:
                        new_stop_id = _place_stop(ib, symbol, remaining_qty, new_stop_price)

                    pos["qty"] = remaining_qty
                    pos["stop_order_id"] = new_stop_id
                    pos["current_stop_price"] = new_stop_price
                    pos["state"] = "post_breakeven_partial_done"
                    log_event(
                        {
                            "event": "partial_profit_taken",
                            "symbol": symbol,
                            "sold_qty": partial_qty,
                            "remaining_qty": remaining_qty,
                            "new_stop": new_stop_price,
                        }
                    )
                    notify(
                        f"PARTIAL {symbol}",
                        f"sold {partial_qty}/{partial_qty + remaining_qty} @ ${price:.2f}",
                        "default",
                    )

        elif reached_breakeven:
            # Only reachable if rules.json ever configures
            # breakeven_trigger_R < partial_profit_trigger_R -- kept as a
            # defensive fallback for that ordering.
            cancel_confirmed = _cancel_stop(ib, pos.get("stop_order_id"))
            if not cancel_confirmed:
                log_event(
                    {
                        "event": "breakeven_move_skipped",
                        "symbol": symbol,
                        "reason": "could not confirm old stop cancelled",
                    }
                )
            else:
                new_stop_id = _place_stop(ib, symbol, pos["qty"], entry_price)
                pos["stop_order_id"] = new_stop_id
                pos["current_stop_price"] = entry_price
                pos["state"] = "post_breakeven_no_partial"
                log_event({"event": "moved_to_breakeven", "symbol": symbol, "new_stop": entry_price})
                notify(f"BE {symbol}", f"stop -> ${entry_price:.2f}", "default")

    elif state.startswith("post_breakeven"):
        bars_5m = get_5m_bars_today(symbol)
        if bars_5m is not None:
            swing_lows = compute_swing_lows(bars_5m)
            if swing_lows:
                newest_swing_low = swing_lows[-1]
                current_stop = pos.get("current_stop_price", pos.get("initial_stop", entry_price))
                candidate_stop = newest_swing_low - 0.01
                # Stops only ratchet up, never down.
                if candidate_stop > current_stop and pos.get("qty", 0) > 0:
                    cancel_confirmed = _cancel_stop(ib, pos.get("stop_order_id"))
                    if not cancel_confirmed:
                        log_event(
                            {
                                "event": "stop_ratchet_skipped",
                                "symbol": symbol,
                                "reason": "could not confirm old stop cancelled",
                            }
                        )
                    else:
                        new_stop_id = _place_stop(ib, symbol, pos["qty"], candidate_stop)
                        pos["stop_order_id"] = new_stop_id
                        pos["current_stop_price"] = candidate_stop
                        log_event(
                            {
                                "event": "stop_ratcheted",
                                "symbol": symbol,
                                "old_stop": current_stop,
                                "new_stop": candidate_stop,
                            }
                        )
                        notify(f"TRAIL {symbol}", f"stop ${current_stop:.2f} -> ${candidate_stop:.2f}", "default")

    return pos


# --------------------------------------------------------------------------
# STEP 6: FORCE CLOSE
# --------------------------------------------------------------------------

def force_close_all(ib, positions: list[dict]) -> list[dict]:
    """Flatten every position. Returns the positions that could NOT be
    confirmed closed (empty list if everything sold), so a rejected/failed
    EOD sell doesn't get silently forgotten -- the caller keeps managing it
    on future cycles instead of losing track of real broker-side exposure."""
    still_open = []
    for pos in positions:
        cancel_confirmed = _cancel_stop(ib, pos.get("stop_order_id"))
        if not cancel_confirmed:
            log_event({"event": "force_close_cancel_unconfirmed", "symbol": pos["symbol"]})

        trade = _sell_market(ib, pos["symbol"], pos["qty"])
        status = trade.orderStatus.status

        if status in FAILED_SELL_STATUSES:
            log_event(
                {
                    "event": "force_close_failed",
                    "symbol": pos["symbol"],
                    "qty": pos["qty"],
                    "status": status,
                }
            )
            notify(
                f"FORCE CLOSE FAILED {pos['symbol']}",
                f"status={status}; qty {pos['qty']} still open, needs manual attention",
                "high",
            )
            still_open.append(pos)
        else:
            log_event(
                {
                    "event": "force_closed",
                    "symbol": pos["symbol"],
                    "qty": pos["qty"],
                    "status": status,
                }
            )

    return still_open


# --------------------------------------------------------------------------
# STEP 8: ENTRY SCAN - daily/intraday filters (D1-D3 / I1-I3)
# --------------------------------------------------------------------------

def get_daily_context(symbol_yahoo: str) -> dict | None:
    try:
        hist = yf.Ticker(symbol_yahoo).history(period="300d", interval="1d", auto_adjust=True)
    except Exception:
        return None
    if hist is None or len(hist) < 201:
        return None
    prior_day = hist.iloc[-2]
    sma200 = float(hist["Close"].iloc[-201:-1].mean())
    return {
        "prior_day_high": float(prior_day["High"]),
        "prior_day_close": float(prior_day["Close"]),
        "sma200": sma200,
    }


def get_intraday_context(symbol_yahoo: str, rvol_lookback_days: int) -> dict | None:
    try:
        bars = yf.Ticker(symbol_yahoo).history(
            period=f"{rvol_lookback_days + 5}d", interval="5m", prepost=True, auto_adjust=True
        )
    except Exception:
        return None
    if bars is None or bars.empty:
        return None

    bars.index = bars.index.tz_convert(ET)
    today = datetime.now(ET).date()
    today_bars = bars[bars.index.date == today]
    if today_bars.empty:
        return None

    premarket_bars = today_bars[today_bars.index.map(lambda ts: (ts.hour, ts.minute) < (9, 30))]
    premarket_high = float(premarket_bars["High"].max()) if not premarket_bars.empty else None

    regular_bars = today_bars[
        today_bars.index.map(lambda ts: (9, 30) <= (ts.hour, ts.minute) <= (16, 0))
    ]
    if regular_bars.empty:
        return None

    today_hod = float(regular_bars["High"].max())
    today_lod = float(regular_bars["Low"].min())
    latest_price = float(regular_bars["Close"].iloc[-1])
    today_cum_volume = float(regular_bars["Volume"].sum())

    now_hm = (regular_bars.index[-1].hour, regular_bars.index[-1].minute)
    all_dates = sorted(set(bars.index.date) - {today})
    past_days = all_dates[-rvol_lookback_days:] if all_dates else []

    past_cum_volumes = []
    for d in past_days:
        day_bars = bars[bars.index.date == d]
        day_regular = day_bars[
            day_bars.index.map(lambda ts: (9, 30) <= (ts.hour, ts.minute) <= now_hm)
        ]
        if not day_regular.empty:
            past_cum_volumes.append(float(day_regular["Volume"].sum()))

    avg_past_cum_volume = (sum(past_cum_volumes) / len(past_cum_volumes)) if past_cum_volumes else None
    rvol = (
        today_cum_volume / avg_past_cum_volume
        if avg_past_cum_volume and avg_past_cum_volume > 0
        else None
    )

    return {
        "premarket_high": premarket_high,
        "today_hod": today_hod,
        "today_lod": today_lod,
        "latest_price": latest_price,
        "rvol": rvol,
    }


def evaluate_entry_filters(symbol_ibkr: str, rules: dict) -> tuple[bool, list[str], dict]:
    symbol_yahoo = ibkr_to_yahoo(symbol_ibkr)
    reasons: list[str] = []

    daily_ctx = get_daily_context(symbol_yahoo)
    if daily_ctx is None:
        return False, ["insufficient daily data"], {}

    daily_filters = rules["daily_filters"]
    intraday_filters = rules["intraday_filters"]

    intraday_ctx = get_intraday_context(symbol_yahoo, intraday_filters["I3_rvol_lookback_days"])
    if intraday_ctx is None:
        return False, ["insufficient intraday data"], {}

    price = intraday_ctx["latest_price"]

    d1_pass = price > daily_ctx["prior_day_high"]
    if daily_filters.get("D1_above_prior_day_high") and not d1_pass:
        reasons.append("D1 fail: price not above prior day high")

    d2_pass = daily_ctx["prior_day_close"] > daily_ctx["sma200"]
    if daily_filters.get("D2_prior_close_above_sma200") and not d2_pass:
        reasons.append("D2 fail: prior close not above SMA200")

    gap_pct = (price - daily_ctx["prior_day_close"]) / daily_ctx["prior_day_close"] * 100
    d3_threshold = daily_filters.get("D3_min_gap_pct_from_prior_close", 0)
    d3_pass = gap_pct >= d3_threshold
    if not d3_pass:
        reasons.append(f"D3 fail: gap {gap_pct:.2f}% < {d3_threshold}%")

    i1_pass = intraday_ctx["premarket_high"] is not None and price > intraday_ctx["premarket_high"]
    if intraday_filters.get("I1_above_premarket_high") and not i1_pass:
        reasons.append("I1 fail: price not above premarket high")

    i2_pass = price >= intraday_ctx["today_hod"]
    if intraday_filters.get("I2_above_today_hod") and not i2_pass:
        reasons.append("I2 fail: price not at/above today HOD")

    rvol_min = intraday_filters.get("I3_rvol_min", 0)
    i3_pass = intraday_ctx["rvol"] is not None and intraday_ctx["rvol"] >= rvol_min
    if not i3_pass:
        rvol_str = f"{intraday_ctx['rvol']:.2f}" if intraday_ctx["rvol"] is not None else "N/A"
        reasons.append(f"I3 fail: rvol {rvol_str} < {rvol_min}")

    all_pass = bool(d1_pass and d2_pass and d3_pass and i1_pass and i2_pass and i3_pass)

    details = {
        "price": price,
        "today_lod": intraday_ctx["today_lod"],
        "today_hod": intraday_ctx["today_hod"],
        "gap_pct": gap_pct,
    }
    return all_pass, reasons, details


def read_watchlist() -> list[str]:
    if not WATCHLIST_PATH.exists():
        return []
    tickers = []
    for line in WATCHLIST_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ticker = line.split("#")[0].strip()
        if ticker:
            tickers.append(ticker)
    return tickers


def count_today_buys() -> int:
    if not TRADES_CSV_PATH.exists():
        return 0
    today_et = datetime.now(ET).strftime("%Y-%m-%d")
    count = 0
    with TRADES_CSV_PATH.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("side") == "BUY" and row.get("timestamp_iso", "").startswith(today_et):
                count += 1
    return count


def entry_scan(ib, rules: dict, env: dict) -> list[dict]:
    new_positions: list[dict] = []

    today_buy_count = count_today_buys()
    if today_buy_count >= env["max_trades_per_day"]:
        log_event({"event": "entry_scan_skipped", "reason": "max_trades_per_day reached", "count": today_buy_count})
        return new_positions

    watchlist_tickers = read_watchlist()

    # Query current IBKR positions FIRST to prevent silent double-entry
    # (e.g. after a subprocess crash post-fill leading to re-attempt).
    held_symbols = {p.contract.symbol for p in ib.positions()}

    max_concurrent_positions = rules.get("risk", {}).get("max_concurrent_positions")

    if max_concurrent_positions is not None and len(held_symbols) >= max_concurrent_positions:
        log_event(
            {
                "event": "entry_scan_skipped",
                "reason": "max_concurrent_positions reached",
                "count": len(held_symbols),
            }
        )
        return new_positions

    for ticker in watchlist_tickers:
        if today_buy_count >= env["max_trades_per_day"]:
            break

        if max_concurrent_positions is not None and len(held_symbols) >= max_concurrent_positions:
            log_event(
                {
                    "event": "entry_scan_stopped",
                    "reason": "max_concurrent_positions reached",
                    "count": len(held_symbols),
                }
            )
            break

        if ticker in held_symbols:
            log_event({"event": "entry_skipped", "symbol": ticker, "reason": "already held"})
            continue

        passed, reasons, details = evaluate_entry_filters(ticker, rules)
        if not passed:
            log_event({"event": "entry_skipped", "symbol": ticker, "reasons": reasons})
            continue

        price = details["price"]
        today_lod = details["today_lod"]
        initial_stop = today_lod * 0.99
        R = price - initial_stop
        if R <= 0:
            log_event({"event": "entry_skipped", "symbol": ticker, "reason": "non-positive R"})
            continue

        risk_dollars = env["portfolio_value_usd"] * (env["max_risk_per_trade_pct"] / 100)
        size_by_risk = math.floor(risk_dollars / R)
        size_by_cap = math.floor(env["portfolio_value_usd"] * 0.10 / price)
        size = min(size_by_risk, size_by_cap)

        if size < 1:
            log_event({"event": "entry_skipped", "symbol": ticker, "reason": "size < 1"})
            continue

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "trading_bot.cli.trade", "--symbol", ticker, "--side", "BUY", "--size", str(size)],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECS,
            )
        except subprocess.TimeoutExpired:
            log_event({"event": "entry_failed", "symbol": ticker, "reason": "trade.py timeout"})
            continue

        if proc.returncode != 0:
            log_event(
                {
                    "event": "entry_failed",
                    "symbol": ticker,
                    "reason": "trade.py nonzero exit",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )
            continue

        stop_order_id = _place_stop(ib, ticker, size, initial_stop)

        new_pos = {
            "symbol": ticker,
            "entry_price": price,
            "entry_time_iso": datetime.now(timezone.utc).isoformat(),
            "qty": size,
            "initial_stop": initial_stop,
            "current_stop_price": initial_stop,
            "stop_order_id": stop_order_id,
            "state": "pre_breakeven",
            "R": R,
        }
        new_positions.append(new_pos)
        held_symbols.add(ticker)
        today_buy_count += 1
        log_event(
            {
                "event": "entry_opened",
                "symbol": ticker,
                "qty": size,
                "entry_price": price,
                "initial_stop": initial_stop,
            }
        )
        notify(f"BUY {ticker}", f"@ ${price:.2f}, stop ${initial_stop:.2f}, qty {size}", "default")

    return new_positions


# --------------------------------------------------------------------------
# IBKR connect (with one retry)
# --------------------------------------------------------------------------

def connect_ibkr(host: str, port: int, client_id: int) -> IBKRClient:
    try:
        return IBKRClient(host, port, client_id)
    except Exception as e:
        log_event({"event": "connect_failed_retrying", "error": str(e)})
        time.sleep(5)
        return IBKRClient(host, port, client_id)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    now_et = datetime.now(ET)
    status = get_market_status(now_et)

    # Weekend/too_early/closed already handled by the fast-path gate above
    # this point; status here is one of manage_only / force_close / ok.

    load_dotenv()
    env = {
        "host": os.environ.get("IBKR_HOST", "127.0.0.1"),
        "port": int(os.environ.get("IBKR_PORT", "7497")),
        "client_id": int(os.environ.get("IBKR_CLIENT_ID", "2")),
        "portfolio_value_usd": float(os.environ.get("PORTFOLIO_VALUE_USD", "0")),
        "max_trade_size_usd": float(os.environ.get("MAX_TRADE_SIZE_USD", "0")),
        "max_trades_per_day": int(os.environ.get("MAX_TRADES_PER_DAY", "0")),
        "max_risk_per_trade_pct": float(os.environ.get("MAX_RISK_PER_TRADE_PCT", "1.0")),
    }

    rules = json.loads(RULES_PATH.read_text())

    try:
        ibkr = connect_ibkr(env["host"], env["port"], env["client_id"])
    except Exception as e:
        log_event({"event": "connect_failed_final", "error": str(e)})
        return 1

    try:
        positions = load_positions()

        positions, stopped_out_events = check_stop_outs(ibkr.ib, positions)
        for event in stopped_out_events:
            log_event(event)

        # STEP 4: manage existing positions (applies to manage_only,
        # force_close, and ok - we manage before deciding to force-close).
        # Saved incrementally, and isolated per-symbol, so a crash or
        # exception managing one position doesn't lose already-completed
        # broker-side actions for the others, or block force-close/entry
        # scan from running this cycle.
        managed_positions = []
        for pos in positions:
            try:
                managed_positions.append(manage_position(ibkr.ib, pos, rules))
            except Exception as e:
                log_event(
                    {
                        "event": "manage_position_failed",
                        "symbol": pos.get("symbol"),
                        "error": str(e),
                    }
                )
                managed_positions.append(pos)  # keep prior state; retry next cycle
            save_positions(managed_positions + positions[len(managed_positions):])
        positions = managed_positions

        if status == "force_close":
            notify("EOD Force Close", f"flattening {len(positions)} positions", "high")
            positions = force_close_all(ibkr.ib, positions)
            save_positions(positions)
            log_event({"event": "force_close_complete", "positions_remaining": len(positions)})
            write_heartbeat(HEARTBEAT_PATH, status)
            return 0

        if status == "manage_only":
            write_heartbeat(HEARTBEAT_PATH, status)
            return 0

        if status == "ok":
            new_positions = entry_scan(ibkr.ib, rules, env)
            positions.extend(new_positions)
            save_positions(positions)
            write_heartbeat(HEARTBEAT_PATH, status)
            return 0

        return 0

    finally:
        try:
            ibkr.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except SystemExit:
        raise
    except Exception as exc:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with CYCLE_ERRORS_LOG.open("a") as f:
            f.write(f"--- {datetime.now(timezone.utc).isoformat()} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        notify("Cycle CRASHED", str(exc)[:500], "high")
        sys.exit(1)
