"""Paper-trading cycle for the UT Bot ATR trailing-stop FX strategy. Runs
ALONGSIDE the gap-and-go (cycle.py) and SMC (smc_cycle.py) bots with fully
separate state (ut_bot_* files), its own IBKR client id, and (once
scheduled -- not wired into setup_schedule.py yet, see below) its own
task, so any of the three can be disabled without touching the others.

Intended to run once per hour, a couple minutes after each hour's bar
closes (e.g. HH:02), NOT every 5 minutes like the equity bots -- UT Bot's
signal is computed from CLOSED hourly bars, so checking more often than
that can't produce a new signal, only burn yfinance/IBKR calls. Not yet
added to setup_schedule.py: this should be run manually against a paper
account first to confirm it behaves correctly before it's put on a timer.

IMPORTANT -- read ut_bot_live.py's module docstring before enabling
anything beyond USDJPY: only USDJPY's configured parameters have cleared
a walk-forward validation. EURUSD/USDCAD are wired up and tradeable, but
their edge only showed up in a narrow parameter corner during backtesting
-- exactly the pattern that turned out to be curve-fit noise for every
OTHER pair tested besides USDJPY. `ut_bot_rules.json`'s "validated" flag
per pair records this; this module does not read that flag to change
behavior (it doesn't gate trading on it) -- it's there to inform you, not
to enforce a decision only you should make.

How it maps onto the backtest-validated configuration:
  - Signal logic: each cycle re-runs the SAME find_ut_bot_long_trades pass
    the backtest used (via latest_long_entry_signal/latest_sell_signal),
    over a fresh 1h-bar fetch per pair -- one source of truth, no live-vs-
    backtest reimplementation drift.
  - Volatility regime filter: applied per pair exactly as configured in
    ut_bot_rules.json (only USDJPY has it enabled, matching what was
    actually validated).
  - Partial profit-taking / breakeven: NOT wired up here at all, even
    though ut_bot_engine.py supports it -- it was found to hurt USDJPY's
    backtested returns (see that module's docstring), so the live bot
    always runs a position through to its plain crossunder exit.
  - The resting stop order at the broker tracks the ATR trailing-stop
    LINE, not a simple ratcheting price the way cycle.py's does -- it is
    cancelled and replaced every cycle a position stays open, since the
    line can move by design (see manage_position).

Known live-vs-backtest deviations (accepted for paper validation):
  - Entries/exits fill at market on the bar AFTER the signal is detected
    (this cycle runs some minutes after the hour closes), not exactly at
    that bar's own close the way the backtest assumes -- live fills
    measure real slippage the backtest never modeled.
  - FX commission/spread cost, while modeled in the backtest
    (ut_bot_engine.py's fx_commission_bps/spread_pips), is not separately
    re-modeled here -- IBKR bills/fills for real, so there's nothing to
    simulate live.
  - yfinance FX bars can lag or gap; a pair with a stale last bar is
    skipped fail-closed that cycle (see _bars_are_fresh).

VERIFIED against a real paper-account order (2026-07-31): IDEALPRO's
minimum order size is USD 25,000 notional -- IBKR's own rejection
message for a smaller test order was "Warning 399: Your order size is
below the USD 25000 IdealPro minimum and will be routed as an odd lot
order." portfolio.position_size's raw sized quantity is NOT checked
against this anywhere in this module -- a signal that sizes below 25,000
units of the base currency will still attempt an order, which IBKR will
either odd-lot-route or reject depending on account settings (see the
next paragraph). Worth adding an explicit floor/skip before this runs
unattended.

BLOCKING, discovered by the same test: this account currently rejects
ALL forex orders outright with "Error 201: FX trade would expose account
to currency leverage" -- an IBKR ACCOUNT-LEVEL permission/setting, not a
bug in this module's order-placement code (confirmed with both a market
and a limit order, both rejected identically). This needs to be resolved
in IBKR's own account configuration (enabling whatever margin/FX trading
permission this paper account is currently missing) before this bot can
place a single real order, paper or otherwise -- everything else in this
module (connection, data fetch, signal detection, contract qualification)
has been confirmed working end-to-end; order placement is the one piece
still blocked, and it's blocked outside this codebase entirely.
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
UT_BOT_CYCLE_ERRORS_LOG = LOGS_DIR / "ut_bot_cycle_errors.log"

SETTLED_ORDER_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"}
FAILED_SELL_STATUSES = {"Cancelled", "ApiCancelled", "Inactive", "Rejected"}
CANCEL_CONFIRM_TIMEOUT_SECS = 5
BAR_STALENESS_LIMIT_MINUTES = 90  # hourly cadence + slack for yfinance lag
STOP_OUT_LOOKBACK_HOURS = 2  # wider than the equity bots' 1h: hourly (not 5m) cadence


# --------------------------------------------------------------------------
# Fast time gate (pure stdlib) -- same pattern as cycle.py/smc_cycle.py:
# exit in well under a second when the FX market's closed, before any
# heavy imports. Duplicates ut_bot_live.get_market_status's weekday logic
# rather than importing it, same reason smc_cycle.py duplicates
# smc_live.get_market_status here: keeping this module safely importable
# by tests without a side-effecting exit.
# --------------------------------------------------------------------------

def _fast_exit_check() -> str | None:
    now_et = datetime.now(ET)
    weekday = now_et.weekday()  # Monday=0 .. Sunday=6
    hm = now_et.strftime("%H:%M")
    if weekday == 5:
        return "closed"
    if weekday == 6 and hm < "17:00":
        return "closed"
    if weekday == 4 and hm >= "17:00":
        return "closed"
    return None


if __name__ == "__main__":
    _EARLY_EXIT_STATUS = _fast_exit_check()
    if _EARLY_EXIT_STATUS is not None:
        print(f"Market status: {_EARLY_EXIT_STATUS}. Exiting.")
        sys.exit(0)

# --------------------------------------------------------------------------
# Heavy imports -- only reached once the FX market is plausibly open (or
# when imported by tests, which is why the gate above is __main__-guarded).
# --------------------------------------------------------------------------
import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from ib_async import Forex, MarketOrder, StopOrder  # noqa: E402

from trading_bot import ut_bot_live  # noqa: E402
from trading_bot.backtest import portfolio  # noqa: E402
from trading_bot.backtest.ut_bot_signals import atr_trailing_stop, latest_long_entry_signal, latest_sell_signal  # noqa: E402
from trading_bot.broker.ibkr_client import IBKRClient  # noqa: E402
from trading_bot.util.notifier import notify  # noqa: E402


# --------------------------------------------------------------------------
# Logging / state
# --------------------------------------------------------------------------

def log_event(event: dict) -> None:
    event = dict(event)
    event.setdefault("timestamp_iso", datetime.now(timezone.utc).isoformat())
    with ut_bot_live.UT_BOT_SAFETY_LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")


def load_positions() -> list[dict]:
    if not ut_bot_live.UT_BOT_POSITIONS_PATH.exists():
        return []
    try:
        with ut_bot_live.UT_BOT_POSITIONS_PATH.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_positions(positions: list[dict]) -> None:
    tmp_path = ut_bot_live.UT_BOT_POSITIONS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(positions, f, indent=2)
    os.replace(tmp_path, ut_bot_live.UT_BOT_POSITIONS_PATH)


def append_trade_row(symbol: str, side: str, size: float, fill_price: float, order_id, status: str, reason: str) -> None:
    exists = ut_bot_live.UT_BOT_TRADES_CSV_PATH.exists()
    with ut_bot_live.UT_BOT_TRADES_CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ut_bot_live.TRADES_CSV_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": side,
                "size": size,
                "fill_price": round(float(fill_price), 6),  # FX rates need more precision than 2dp
                "order_id": order_id,
                "status": status,
                "reason": reason,
            }
        )


# --------------------------------------------------------------------------
# Order helpers -- adapted from cycle.py/smc_cycle.py, swapping the
# hardcoded Stock(symbol, "SMART", "USD") contract for an FX one. Neither
# outsideRth (an RTH-session concept) nor a max-trades-per-day cap apply
# to a ~24/5 FX market, so both are simply absent here.
# --------------------------------------------------------------------------

def _qualify(ib, symbol: str):
    fx = Forex(symbol)
    qualified = ib.qualifyContracts(fx)
    return qualified[0] if qualified else fx


def _cancel_stop(ib, stop_order_id) -> bool:
    """Cancel the resting stop and confirm it left the open-orders book.
    False means "the old stop may still be live" -- callers must not place
    an overlapping order for the same position."""
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


def _place_stop(ib, symbol: str, qty: float, stop_price: float):
    contract = _qualify(ib, symbol)
    order = StopOrder("SELL", qty, round(float(stop_price), 6))
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    return trade.order.orderId


def _market_order(ib, symbol: str, side: str, qty: float):
    contract = _qualify(ib, symbol)
    order = MarketOrder(side, qty)
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

def get_1h_bars(symbol: str):
    """Up to ~2 years of hourly bars (yfinance's practical cap for
    intraday intervals) for `symbol`'s FX ticker, tz-converted to UTC --
    matches how the backtest's own cached fx_1h/*.csv data was loaded, so
    live signal computation sees the same shape of data the backtest
    validated against. Returns None on any data problem. Needs this much
    history because vol_filter_lookback (USDJPY: 500 bars) has no valid
    trailing average until that many bars exist."""
    yahoo_symbol = f"{symbol}=X"
    try:
        bars = yf.Ticker(yahoo_symbol).history(period="730d", interval="1h", auto_adjust=True)
    except Exception:
        return None
    if bars is None or bars.empty:
        return None
    bars.index = bars.index.tz_convert("UTC")
    return bars


def _bars_are_fresh(bars) -> bool:
    last_ts = bars.index[-1]
    now_utc = datetime.now(timezone.utc)
    return (now_utc - last_ts) <= timedelta(minutes=BAR_STALENESS_LIMIT_MINUTES)


# --------------------------------------------------------------------------
# Stop-out detection (fills matching, same approach as cycle.py/smc_cycle.py)
# --------------------------------------------------------------------------

def check_stop_outs(ib, positions: list[dict]) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=STOP_OUT_LOOKBACK_HOURS)
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
        if fill_time is None or fill_time >= cutoff:
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
            qty = pos.get("qty", 0)
            pnl = (fill_price - pos.get("entry_price", fill_price)) * qty
            append_trade_row(pos["symbol"], "SELL", qty, fill_price, stop_order_id, "Filled", "stop")
            log_event({"event": "stopped_out", "symbol": pos["symbol"], "fill_price": fill_price, "pnl": pnl})
            notify(f"UT Bot STOP {pos['symbol']}", f"exit {fill_price:.5f}, P&L {pnl:+.2f}", "default")
        else:
            remaining.append(pos)
    return remaining


# --------------------------------------------------------------------------
# Position management: exit on the plain crossunder, else refresh the
# resting stop to track the ATR trailing-stop LINE (which moves every
# bar by design -- unlike cycle.py's stop, this isn't just a one-way
# ratchet, so it's unconditionally recomputed and replaced each cycle).
# --------------------------------------------------------------------------

def manage_position(ib, pos: dict) -> dict | None:
    symbol = pos["symbol"]
    key_value = pos["key_value"]
    atr_period = pos["atr_period"]

    bars = get_1h_bars(symbol)
    if bars is None or not _bars_are_fresh(bars):
        log_event({"event": "manage_skipped", "symbol": symbol, "reason": "no fresh 1h bars"})
        return pos

    bars_dict = ut_bot_live.bars_frame_to_dict(bars)

    if latest_sell_signal(bars_dict, key_value, atr_period):
        if not _cancel_stop(ib, pos.get("stop_order_id")):
            log_event({"event": "exit_skipped", "symbol": symbol, "reason": "old stop cancel unconfirmed"})
            return pos
        trade = _market_order(ib, symbol, "SELL", pos["qty"])
        status = trade.orderStatus.status
        if status in FAILED_SELL_STATUSES:
            log_event({"event": "exit_sell_failed", "symbol": symbol, "status": status})
            pos["stop_order_id"] = _place_stop(ib, symbol, pos["qty"], pos["current_stop_price"])
            return pos
        fill_price = trade.orderStatus.avgFillPrice or bars_dict["close"][-1]
        append_trade_row(symbol, "SELL", pos["qty"], fill_price, trade.order.orderId, status, "sell_signal")
        pnl = (fill_price - pos["entry_price"]) * pos["qty"]
        log_event({"event": "exited", "symbol": symbol, "fill_price": fill_price, "pnl": pnl})
        notify(f"UT Bot EXIT {symbol}", f"sell_signal @ {fill_price:.5f}, P&L {pnl:+.2f}", "default")
        return None

    new_stop_price = atr_trailing_stop(bars_dict["high"], bars_dict["low"], bars_dict["close"], key_value, atr_period)[-1]
    if new_stop_price != pos.get("current_stop_price"):
        if _cancel_stop(ib, pos.get("stop_order_id")):
            pos["stop_order_id"] = _place_stop(ib, symbol, pos["qty"], new_stop_price)
            pos["current_stop_price"] = new_stop_price
        else:
            log_event({"event": "stop_refresh_skipped", "symbol": symbol, "reason": "old stop cancel unconfirmed"})
    return pos


# --------------------------------------------------------------------------
# Entry scan
# --------------------------------------------------------------------------

def entry_scan(ib, rules: dict, env: dict, open_positions: list[dict]) -> list[dict]:
    new_positions: list[dict] = []

    max_concurrent = rules["risk"].get("max_concurrent_positions")
    open_count = len(open_positions)

    # Skip anything held ANYWHERE in the account (including the other two
    # bots' positions) -- prevents stacking exposure on the same symbol.
    held_symbols = {p.contract.symbol for p in ib.positions()}

    for symbol, pair_cfg in rules["pairs"].items():
        if max_concurrent is not None and open_count >= max_concurrent:
            log_event({"event": "entry_scan_stopped", "reason": "max_concurrent_positions reached"})
            break
        if symbol in held_symbols:
            continue

        bars = get_1h_bars(symbol)
        if bars is None or not _bars_are_fresh(bars):
            log_event({"event": "entry_skipped", "symbol": symbol, "reason": "no fresh 1h bars"})
            continue

        signal = latest_long_entry_signal(
            ut_bot_live.bars_frame_to_dict(bars),
            key_value=pair_cfg["key_value"],
            atr_period=pair_cfg["atr_period"],
            vol_filter_lookback=pair_cfg.get("vol_filter_lookback"),
            vol_filter_max_ratio=pair_cfg.get("vol_filter_max_ratio", 1.5),
            vol_filter_atr_period=pair_cfg.get("vol_filter_atr_period", 14),
        )
        if signal is None:
            continue

        size = portfolio.position_size(
            env["portfolio_value_usd"],
            rules["risk"]["max_risk_per_trade_pct"],
            signal["entry_price"],
            signal["stop_at_entry"],
            rules["risk"]["max_position_size_pct_of_portfolio"],
        )
        if size < 1:
            log_event({"event": "entry_skipped", "symbol": symbol, "reason": "size < 1"})
            continue

        trade = _market_order(ib, symbol, "BUY", size)
        status = trade.orderStatus.status
        if status in FAILED_SELL_STATUSES:
            log_event({"event": "entry_failed", "symbol": symbol, "status": status})
            continue

        fill_price = trade.orderStatus.avgFillPrice or signal["entry_price"]
        append_trade_row(symbol, "BUY", size, fill_price, trade.order.orderId, status, "entry")
        stop_order_id = _place_stop(ib, symbol, size, signal["stop_at_entry"])

        new_pos = {
            "symbol": symbol,
            "entry_price": float(fill_price),
            "entry_time_iso": datetime.now(timezone.utc).isoformat(),
            "qty": size,
            "initial_stop": signal["stop_at_entry"],
            "current_stop_price": signal["stop_at_entry"],
            "stop_order_id": stop_order_id,
            # Persisted per-position (not re-looked-up from rules each
            # cycle) so a later rules.json edit can't silently change how
            # an ALREADY-OPEN position is managed.
            "key_value": pair_cfg["key_value"],
            "atr_period": pair_cfg["atr_period"],
        }
        new_positions.append(new_pos)
        held_symbols.add(symbol)
        open_count += 1
        log_event(
            {
                "event": "entry_opened",
                "symbol": symbol,
                "qty": size,
                "entry_price": float(fill_price),
                "stop": signal["stop_at_entry"],
            }
        )
        notify(f"UT Bot BUY {symbol}", f"@ {float(fill_price):.5f}, stop {signal['stop_at_entry']:.5f}, qty {size}", "default")

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
    rules = ut_bot_live.load_ut_bot_rules()
    status = ut_bot_live.get_market_status(datetime.now(ET))
    if status == "closed":
        print(f"Market status: {status}. Exiting.")
        return 0

    load_dotenv()
    env = {
        "host": os.environ.get("IBKR_HOST", "127.0.0.1"),
        "port": int(os.environ.get("IBKR_PORT", "7497")),
        "client_id": int(os.environ.get("IBKR_UTBOT_CLIENT_ID", "5")),
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
        managed = []
        for i, pos in enumerate(positions):
            try:
                updated = manage_position(ibkr.ib, pos)
            except Exception as e:
                log_event({"event": "manage_position_failed", "symbol": pos.get("symbol"), "error": str(e)})
                updated = pos  # keep prior state; retry next cycle
            if updated is not None:
                managed.append(updated)
            save_positions(managed + positions[i + 1:])
        positions = managed
        save_positions(positions)

        new_positions = entry_scan(ibkr.ib, rules, env, positions)
        positions.extend(new_positions)
        save_positions(positions)

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
        with UT_BOT_CYCLE_ERRORS_LOG.open("a") as f:
            f.write(f"--- {datetime.now(timezone.utc).isoformat()} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        notify("UT Bot Cycle CRASHED", str(exc)[:500], "high")
        sys.exit(1)
