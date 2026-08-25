"""Autonomous 5-minute trading cycle for the SMC (order-block) strategy,
paper trading. Runs ALONGSIDE the gap-and-go bot (cycle.py) with fully
separate state (smc_* files), its own IBKR client id, and its own
scheduled task -- either bot can be disabled without touching the other.

Intended to run every 5 minutes via Windows Task Scheduler (see
setup_schedule.py's HT_SMC_Cycle).

How it maps onto the backtest-validated configuration
(daily_trend_filter + force_close_same_day + reactive_derisk):
  - daily_trend_filter: baked into smc_watchlist.txt by smc_prefilter.py
    each morning (prior close > SMA200 uses only prior-day data, so it
    cannot change intraday -- checking it once at 09:40 ET is exactly
    equivalent to the backtest's per-entry-day check).
  - force_close_same_day: the 15:51 ET force-close below flattens every
    position daily, so nothing is ever held overnight.
  - reactive_derisk: smc_live.entry_size scales new entries by trailing
    realized profit factor from smc_trades.csv.
  - Signal logic: each cycle re-runs the SAME find_smc_long_trades pass
    the backtest used, over ~7 days of 5-min bars, and acts only when the
    entry triggers on the latest bar (smc_signals.latest_entry_signal) --
    one source of truth, no live-vs-backtest reimplementation drift.

Known live-vs-backtest deviations (accepted for paper validation):
  - Entries fill at market after the OB retest prints, not at the exact
    OB-high limit the backtest assumes -- live fills measure real
    slippage, which the backtest never modeled.
  - The entry window is 10:05-15:30 ET (matching the existing bot's
    cadence); the backtest entered from the open.
  - yfinance 5-min data can lag or drop bars; symbols with stale data
    are skipped fail-closed that cycle.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

LOGS_DIR = Path("logs")
SMC_CYCLE_ERRORS_LOG = LOGS_DIR / "smc_cycle_errors.log"
SMC_HEARTBEAT_PATH = Path("smc_heartbeat.json")

SETTLED_ORDER_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}
FAILED_SELL_STATUSES = {"Cancelled", "ApiCancelled", "Inactive", "Rejected"}
CANCEL_CONFIRM_TIMEOUT_SECS = 5
BAR_STALENESS_LIMIT_MINUTES = 20


# --------------------------------------------------------------------------
# Fast time gate (pure stdlib) -- same pattern as cycle.py: exit in well
# under a second outside market hours, before any heavy imports.
# --------------------------------------------------------------------------

def _fast_exit_check() -> str | None:
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return "weekend"
    hm = now_et.strftime("%H:%M")
    if hm < "10:00":
        return "too_early"
    if hm > "16:00":
        return "closed"
    return None


if __name__ == "__main__":
    _EARLY_EXIT_STATUS = _fast_exit_check()
    if _EARLY_EXIT_STATUS is not None:
        print(f"Market status: {_EARLY_EXIT_STATUS}. Exiting.")
        sys.exit(0)

# --------------------------------------------------------------------------
# Heavy imports -- only reached once the market is plausibly open (or when
# imported by tests, which is why the gate above is __main__-guarded,
# unlike cycle.py's import-time gate).
# --------------------------------------------------------------------------
import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from ib_async import MarketOrder, Stock, StopOrder  # noqa: E402

from trading_bot import smc_live  # noqa: E402
from trading_bot.backtest.smc_signals import confirmed_new_high_exit, latest_entry_signal  # noqa: E402
from trading_bot.broker.ibkr_client import IBKRClient  # noqa: E402
from trading_bot.util.heartbeat import write_heartbeat  # noqa: E402
from trading_bot.util.notifier import notify  # noqa: E402


# --------------------------------------------------------------------------
# Logging / state
# --------------------------------------------------------------------------

def log_event(event: dict) -> None:
    event = dict(event)
    event.setdefault("timestamp_iso", datetime.now(timezone.utc).isoformat())
    with smc_live.SMC_SAFETY_LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")


def load_positions() -> list[dict]:
    if not smc_live.SMC_POSITIONS_PATH.exists():
        return []
    try:
        with smc_live.SMC_POSITIONS_PATH.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_positions(positions: list[dict]) -> None:
    tmp_path = smc_live.SMC_POSITIONS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(positions, f, indent=2)
    os.replace(tmp_path, smc_live.SMC_POSITIONS_PATH)


def append_trade_row(symbol: str, side: str, size: int, fill_price: float, order_id, status: str, reason: str) -> None:
    """All SMC fills (entries AND every exit) land in smc_trades.csv --
    unlike the gap-and-go bot's trades.csv (BUYs only), because the
    reactive-derisk sizing needs complete round trips to compute a
    trailing profit factor from."""
    exists = smc_live.SMC_TRADES_CSV_PATH.exists()
    with smc_live.SMC_TRADES_CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=smc_live.TRADES_CSV_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": side,
                "size": int(size),
                "fill_price": round(float(fill_price), 4),
                "order_id": order_id,
                "status": status,
                "reason": reason,
            }
        )


def count_today_buys() -> int:
    if not smc_live.SMC_TRADES_CSV_PATH.exists():
        return 0
    today_et = datetime.now(ET).strftime("%Y-%m-%d")
    count = 0
    with smc_live.SMC_TRADES_CSV_PATH.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("side") == "BUY" and row.get("timestamp_iso", "").startswith(today_et):
                count += 1
    return count


# --------------------------------------------------------------------------
# Order helpers -- adapted from cycle.py (not imported: cycle.py exits at
# import time outside market hours).
# --------------------------------------------------------------------------

def _qualify(ib, symbol: str):
    stock = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(stock)
    return qualified[0] if qualified else stock


def _cancel_stop(ib, stop_order_id) -> bool:
    """Cancel the resting stop and confirm it left the open-orders book.
    False means "the old stop may still be live" -- callers must not place
    an overlapping order for the same shares."""
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
        return True

    try:
        ib.cancelOrder(order_to_cancel)
    except Exception as e:
        log_event({"event": "cancel_stop_failed", "stop_order_id": stop_order_id, "error": str(e)})
        return False

    deadline = time.time() + CANCEL_CONFIRM_TIMEOUT_SECS
    while time.time() < deadline:
        ib.sleep(0.25)
        if not any(getattr(o, "orderId", None) == stop_order_id for o in ib.openOrders()):
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


def _market_order(ib, symbol: str, side: str, qty: int):
    contract = _qualify(ib, symbol)
    order = MarketOrder(side, qty)
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
# Bars
# --------------------------------------------------------------------------

def get_5m_bars(symbol_ibkr: str, context: str = "unknown"):
    """~7 days of RTH-only 5-min bars (matches the backtest's RTH-only
    cached data), tz-converted to ET. Returns None on any data problem,
    logging which one -- callers only ever see None, so the three causes
    below used to be indistinguishable after the fact, and the entry
    scan reported all of them as "no fresh 5m bars", which pointed at a
    staleness check that was almost certainly never involved:
    BAR_STALENESS_LIMIT_MINUTES is 20 while the cycle runs every 5, so
    intraday _bars_are_fresh has ~13 minutes of slack and every logged
    occurrence was in fact one of these fetch failures. `context` names
    the call site, since both the entry scan and the new-high exit check
    fetch bars and a bare symbol wouldn't say which one went blind."""
    symbol_yahoo = symbol_ibkr.replace(" ", "-")
    try:
        bars = yf.Ticker(symbol_yahoo).history(period="7d", interval="5m", prepost=False, auto_adjust=True)
    except Exception as e:
        log_event({
            "event": "bars_unavailable",
            "symbol": symbol_ibkr,
            "context": context,
            "cause": "yfinance_error",
            "error": f"{type(e).__name__}: {e}",
        })
        return None
    if bars is None or bars.empty:
        log_event({"event": "bars_unavailable", "symbol": symbol_ibkr, "context": context, "cause": "empty_response"})
        return None
    bars.index = bars.index.tz_convert(ET)
    rth = bars[bars.index.map(lambda ts: (9, 30) <= (ts.hour, ts.minute) < (16, 0))]
    if rth.empty:
        log_event({"event": "bars_unavailable", "symbol": symbol_ibkr, "context": context, "cause": "no_rth_rows"})
        return None
    return rth


def _bars_are_fresh(bars) -> bool:
    last_ts = bars.index[-1]
    now_et = datetime.now(ET)
    return last_ts.date() == now_et.date() and (now_et - last_ts) <= timedelta(minutes=BAR_STALENESS_LIMIT_MINUTES)


# --------------------------------------------------------------------------
# Stop-out detection (fills matching, same approach as cycle.py)
# --------------------------------------------------------------------------

def check_stop_outs(ib, positions: list[dict]) -> list[dict]:
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    try:
        fills = ib.fills()
    except Exception as e:
        log_event({"event": "fills_query_failed", "error": str(e)})
        fills = []

    recent_fills = []
    for f in fills:
        fill_time = getattr(f, "time", None)
        if fill_time is not None and fill_time.tzinfo is None:
            fill_time = fill_time.replace(tzinfo=timezone.utc)
        if fill_time is None or fill_time >= one_hour_ago:
            recent_fills.append(f)

    remaining = []
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
            append_trade_row(pos["symbol"], "SELL", pos.get("qty", 0), fill_price, stop_order_id, "Filled", "stop")
            log_event({"event": "stopped_out", "symbol": pos["symbol"], "fill_price": fill_price, "pnl": pnl})
            notify(f"SMC STOP {pos['symbol']}", f"exit ${fill_price:.2f}, P&L ${pnl:+.2f}", "default")
        else:
            remaining.append(pos)
    return remaining


# --------------------------------------------------------------------------
# Position management: TP1 partial + breakeven move, then confirmed
# new-swing-high full exit -- the same ladder find_smc_long_trades
# simulates, driven by live prices/bars.
# --------------------------------------------------------------------------

def _entry_bar_index(today_bars, entry_bar_iso: str) -> int | None:
    """Index of the position's entry bar within today's bars: exact
    timestamp match preferred, else the first bar at/after it."""
    try:
        entry_ts = datetime.fromisoformat(entry_bar_iso)
    except (TypeError, ValueError):
        return None
    for i, ts in enumerate(today_bars.index):
        if ts >= entry_ts:
            return i
    return None


def manage_position(ib, pos: dict, rules: dict) -> dict | None:
    """Returns the (possibly updated) position, or None once fully closed."""
    symbol = pos["symbol"]

    # --- TP1: sell a fraction, move the stop to breakeven ---
    if not pos.get("tp1_done") and pos.get("tp1_price") is not None:
        price = get_current_price(ib, symbol)
        if price is not None and price >= pos["tp1_price"]:
            if not _cancel_stop(ib, pos.get("stop_order_id")):
                log_event({"event": "tp1_skipped", "symbol": symbol, "reason": "old stop cancel unconfirmed"})
                return pos

            partial_qty = round(pos.get("original_qty", pos["qty"]) * rules["tp1_fraction"])
            partial_qty = min(partial_qty, pos["qty"] - 1)  # always keep a remainder for the runner

            if partial_qty >= 1:
                trade = _market_order(ib, symbol, "SELL", partial_qty)
                status = trade.orderStatus.status
                if status in FAILED_SELL_STATUSES:
                    log_event({"event": "tp1_sell_failed", "symbol": symbol, "status": status})
                    pos["stop_order_id"] = _place_stop(ib, symbol, pos["qty"], pos["current_stop_price"])
                    return pos
                fill_price = trade.orderStatus.avgFillPrice or price
                append_trade_row(symbol, "SELL", partial_qty, fill_price, trade.order.orderId, status, "tp1")
                pos["qty"] -= partial_qty
                notify(f"SMC TP1 {symbol}", f"sold {partial_qty} @ ${fill_price:.2f}, stop -> BE", "default")

            # Signal-level behavior: the stop moves to breakeven on the TP1
            # touch even when the partial itself rounds away to 0 shares.
            new_stop = pos["entry_price"]
            pos["stop_order_id"] = _place_stop(ib, symbol, pos["qty"], new_stop)
            pos["current_stop_price"] = new_stop
            pos["tp1_done"] = True
            log_event({"event": "tp1_done", "symbol": symbol, "sold": max(partial_qty, 0), "new_stop": new_stop})
            return pos

    # --- Full exit on the first confirmed post-entry swing high ---
    # Mirrors the backtest gate: only armed once TP1 is done or was never
    # available (no resistance OB above at entry).
    if pos.get("tp1_done") or pos.get("tp1_price") is None:
        bars = get_5m_bars(symbol, context="new_high_check")
        if bars is None:
            return pos
        today = datetime.now(ET).date()
        today_bars = bars[[ts.date() == today for ts in bars.index]]
        if today_bars.empty:
            return pos
        entry_idx = _entry_bar_index(today_bars, pos.get("entry_bar_iso", ""))
        if entry_idx is None:
            log_event({"event": "new_high_check_skipped", "symbol": symbol, "reason": "entry bar not found"})
            return pos
        highs = today_bars["High"].astype(float).tolist()
        if confirmed_new_high_exit(highs, entry_idx, rules["swing_window"]):
            if not _cancel_stop(ib, pos.get("stop_order_id")):
                log_event({"event": "new_high_exit_skipped", "symbol": symbol, "reason": "old stop cancel unconfirmed"})
                return pos
            trade = _market_order(ib, symbol, "SELL", pos["qty"])
            status = trade.orderStatus.status
            if status in FAILED_SELL_STATUSES:
                log_event({"event": "new_high_sell_failed", "symbol": symbol, "status": status})
                pos["stop_order_id"] = _place_stop(ib, symbol, pos["qty"], pos["current_stop_price"])
                return pos
            # avgFillPrice is only populated once the fill actually
            # settles -- _market_order's wait loop can return as soon as
            # the order merely reaches "Submitted", before that happens.
            # Falling back to 0.0 (as this used to) silently corrupts
            # every downstream P&L calc from smc_trades.csv; the last
            # bar's close is a much closer stand-in for what a market
            # order actually filled near.
            fill_price = trade.orderStatus.avgFillPrice or float(today_bars["Close"].iloc[-1])
            append_trade_row(symbol, "SELL", pos["qty"], fill_price, trade.order.orderId, status, "new_high_exit")
            pnl = (fill_price - pos["entry_price"]) * pos["qty"] if fill_price else None
            log_event({"event": "new_high_exit", "symbol": symbol, "fill_price": fill_price, "pnl": pnl})
            notify(f"SMC EXIT {symbol}", f"new-high exit @ ${fill_price:.2f}", "default")
            return None

    return pos


# --------------------------------------------------------------------------
# EOD force close (this IS the validated force_close_same_day behavior)
# --------------------------------------------------------------------------

def force_close_all(ib, positions: list[dict]) -> list[dict]:
    still_open = []
    for pos in positions:
        if not _cancel_stop(ib, pos.get("stop_order_id")):
            log_event({"event": "force_close_cancel_unconfirmed", "symbol": pos["symbol"]})

        trade = _market_order(ib, pos["symbol"], "SELL", pos["qty"])
        status = trade.orderStatus.status
        if status in FAILED_SELL_STATUSES:
            log_event({"event": "force_close_failed", "symbol": pos["symbol"], "qty": pos["qty"], "status": status})
            notify(
                f"SMC FORCE CLOSE FAILED {pos['symbol']}",
                f"status={status}; qty {pos['qty']} still open, needs manual attention",
                "high",
            )
            still_open.append(pos)
        else:
            # See manage_position's new_high_exit branch: avgFillPrice can
            # still be unset at this point, and falling back to 0.0 (as
            # this used to) silently corrupts every downstream P&L calc
            # from smc_trades.csv. No bars are already in hand here, so
            # fetch the current price as the closest stand-in instead.
            fill_price = trade.orderStatus.avgFillPrice or get_current_price(ib, pos["symbol"]) or 0.0
            append_trade_row(
                pos["symbol"], "SELL", pos["qty"], fill_price, trade.order.orderId, status, "same_day_force_close"
            )
            log_event({"event": "force_closed", "symbol": pos["symbol"], "qty": pos["qty"], "status": status})
    return still_open


# --------------------------------------------------------------------------
# Entry scan
# --------------------------------------------------------------------------

def entry_scan(ib, rules: dict, env: dict, open_positions: list[dict]) -> list[dict]:
    new_positions: list[dict] = []

    max_trades_per_day = rules["risk"].get("max_trades_per_day")
    today_buy_count = count_today_buys()
    if max_trades_per_day is not None and today_buy_count >= max_trades_per_day:
        log_event({"event": "entry_scan_skipped", "reason": "max_trades_per_day reached", "count": today_buy_count})
        return new_positions

    watchlist = smc_live.read_watchlist()
    if not watchlist:
        log_event({"event": "entry_scan_skipped", "reason": "empty smc watchlist"})
        return new_positions

    # Skip anything held ANYWHERE in the account (including the gap-and-go
    # bot's positions) -- prevents the two bots from stacking exposure on
    # the same symbol.
    held_symbols = {p.contract.symbol for p in ib.positions()}
    smc_open_count = len(open_positions)
    max_concurrent = rules["risk"].get("max_concurrent_positions")

    for ticker in watchlist:
        if max_trades_per_day is not None and today_buy_count >= max_trades_per_day:
            break
        if max_concurrent is not None and smc_open_count >= max_concurrent:
            log_event({"event": "entry_scan_stopped", "reason": "max_concurrent_positions reached"})
            break
        if ticker in held_symbols:
            continue

        bars = get_5m_bars(ticker, context="entry_scan")
        if bars is None:
            continue
        if not _bars_are_fresh(bars):
            last_ts = bars.index[-1]
            log_event({
                "event": "entry_skipped",
                "symbol": ticker,
                "reason": "stale 5m bars",
                "last_bar_et": last_ts.isoformat(),
                "age_minutes": round((datetime.now(ET) - last_ts).total_seconds() / 60, 1),
            })
            continue

        signal = latest_entry_signal(
            smc_live.bars_frame_to_dict(bars),
            time_window_bars=rules["time_window_bars"],
            tp1_fraction=rules["tp1_fraction"],
            swing_window=rules["swing_window"],
        )
        if signal is None:
            continue

        size = smc_live.entry_size(
            env["portfolio_value_usd"], signal["entry_price"], signal["stop_price"], rules
        )
        if size < 1:
            log_event({"event": "entry_skipped", "symbol": ticker, "reason": "size < 1"})
            continue

        trade = _market_order(ib, ticker, "BUY", size)
        status = trade.orderStatus.status
        if status in FAILED_SELL_STATUSES:
            log_event({"event": "entry_failed", "symbol": ticker, "status": status})
            continue

        fill_price = trade.orderStatus.avgFillPrice or signal["entry_price"]
        append_trade_row(ticker, "BUY", size, fill_price, trade.order.orderId, status, "entry")
        stop_order_id = _place_stop(ib, ticker, size, signal["stop_price"])

        new_pos = {
            "symbol": ticker,
            "entry_price": float(fill_price),
            "entry_time_iso": datetime.now(timezone.utc).isoformat(),
            "entry_bar_iso": bars.index[-1].isoformat(),
            "qty": size,
            "original_qty": size,
            "stop_price": signal["stop_price"],
            "current_stop_price": signal["stop_price"],
            "stop_order_id": stop_order_id,
            "tp1_price": signal["tp1_price"],
            "tp1_done": False,
        }
        new_positions.append(new_pos)
        held_symbols.add(ticker)
        smc_open_count += 1
        today_buy_count += 1
        log_event(
            {
                "event": "entry_opened",
                "symbol": ticker,
                "qty": size,
                "entry_price": float(fill_price),
                "stop": signal["stop_price"],
                "tp1": signal["tp1_price"],
            }
        )
        notify(
            f"SMC BUY {ticker}",
            f"@ ${float(fill_price):.2f}, stop ${signal['stop_price']:.2f}, qty {size}",
            "default",
        )

    return new_positions


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def connect_ibkr(host: str, port: int, client_id: int) -> IBKRClient:
    try:
        return IBKRClient(host, port, client_id)
    except Exception as e:
        log_event({"event": "connect_failed_retrying", "error": str(e)})
        time.sleep(5)
        return IBKRClient(host, port, client_id)


def main() -> int:
    rules = smc_live.load_smc_rules()
    status = smc_live.get_market_status(datetime.now(ET), rules)
    if status in ("weekend", "too_early", "closed"):
        print(f"Market status: {status}. Exiting.")
        return 0

    load_dotenv()
    env = {
        "host": os.environ.get("IBKR_HOST", "127.0.0.1"),
        "port": int(os.environ.get("IBKR_PORT", "7497")),
        "client_id": int(os.environ.get("IBKR_SMC_CLIENT_ID", "4")),
        "portfolio_value_usd": float(os.environ.get("PORTFOLIO_VALUE_USD", "0")),
    }

    try:
        ibkr = connect_ibkr(env["host"], env["port"], env["client_id"])
    except Exception as e:
        log_event({"event": "connect_failed_final", "error": str(e)})
        return 1

    try:
        positions = load_positions()
        positions = check_stop_outs(ibkr.ib, positions)
        save_positions(positions)

        # Saved incrementally so a crash mid-management doesn't lose
        # broker-side actions already completed for earlier positions.
        # Sliced by loop index (not len(managed)): a position that fully
        # closed contributes nothing to `managed`, which would otherwise
        # skew the not-yet-processed remainder.
        managed = []
        for i, pos in enumerate(positions):
            try:
                updated = manage_position(ibkr.ib, pos, rules)
            except Exception as e:
                log_event({"event": "manage_position_failed", "symbol": pos.get("symbol"), "error": str(e)})
                updated = pos  # keep prior state; retry next cycle
            if updated is not None:
                managed.append(updated)
            save_positions(managed + positions[i + 1:])
        positions = managed
        save_positions(positions)

        if status == "force_close":
            if positions:
                notify("SMC EOD Force Close", f"flattening {len(positions)} positions", "high")
            positions = force_close_all(ibkr.ib, positions)
            save_positions(positions)
            log_event({"event": "force_close_complete", "positions_remaining": len(positions)})
            write_heartbeat(SMC_HEARTBEAT_PATH, status)
            return 0

        if status == "ok":
            new_positions = entry_scan(ibkr.ib, rules, env, positions)
            positions.extend(new_positions)
            save_positions(positions)

        write_heartbeat(SMC_HEARTBEAT_PATH, status)
        return 0
    finally:
        try:
            ibkr.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with SMC_CYCLE_ERRORS_LOG.open("a") as f:
            f.write(f"--- {datetime.now(timezone.utc).isoformat()} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        notify("SMC Cycle CRASHED", str(exc)[:500], "high")
        sys.exit(1)
