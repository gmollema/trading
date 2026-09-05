"""Scheduled daily entrypoint for the rsi2 strategy (see rsi2_live.py).

Runs ONCE per trading day, a few minutes before the 09:30 ET open, and
does nothing at any other time. It decides from the last COMPLETED daily
bar and fills with a market order at the open -- the reachable fill spec,
measured as equivalent to the backtested close-to-close one (6.43% CAGR
vs 6.36% at $75k). See rsi2_live's module docstring for why the
backtested fill itself is not reachable.

The regime filter is ON in rsi2_rules.json (see rsi2_live's docstring):
a decision of "regime_veto" in the log means a real dip signal existed
and was turned down, which is different from "no_signal" and worth
telling apart when reading a dry run.

DRY RUN IS THE DEFAULT. This code places orders on a $50-a-point
contract and has never been run. It logs exactly what it would do and
touches nothing until invoked with --arm. That is a deliberate departure
from cycle.py and smc_cycle.py, which have both been through paper
validation already.

Paper vs live is the PORT, as elsewhere in this repo: IBKR_PORT=7497 is
paper TWS (the default), 7496 is live. Nothing here special-cases one or
the other, so pointing it at a live account is a one-variable change and
should be treated as such.

Run it manually first:
    python -m trading_bot.cli.rsi2_cycle                 # dry run, prints the decision
    python -m trading_bot.cli.rsi2_cycle --ignore-window  # dry run outside 09:25-09:45 ET
    python -m trading_bot.cli.rsi2_cycle --arm            # actually sends orders

To schedule it once you trust it (NOT run here, and deliberately not
added to setup_schedule.py, which would revive the disabled gap-and-go
tasks as a side effect):
    schtasks /create /tn "HT_RSI2_Cycle" /tr "<python> -m trading_bot.cli.rsi2_cycle --arm" ^
             /sc weekly /d MON,TUE,WED,THU,FRI /st 09:26 /f
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from trading_bot import rsi2_live
from trading_bot.util.heartbeat import write_heartbeat

ET = ZoneInfo("America/New_York")
# Written by log_event as well as stdout. The scheduled task runs under
# pythonw with no console, so a shell redirect was the obvious way to keep
# the decisions -- but Task Scheduler mangles the "&" needed to create the
# directory first, and tests/integration/util/test_logger_it.py deletes
# logs/ outright, so a redirect into a missing directory silently discards
# every decision. Owning the file here means the cycle recreates the
# directory itself and the task line stays as plain as smc_cycle's.
CYCLE_LOG_PATH = Path("logs/rsi2_cycle.log")
# Its own client id: sharing one with cycle.py (default), smc_cycle.py (4)
# or the data fetcher (95) would have TWS drop one of the connections.
DEFAULT_CLIENT_ID = 6
DAILY_BAR_LOOKBACK = "2 Y"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true",
                        help="actually place orders; without this nothing is sent")
    parser.add_argument("--ignore-window", action="store_true",
                        help="skip the 09:25-09:45 ET gate (for manual dry runs only)")
    return parser.parse_args(argv)


def log_event(payload: dict, path: Path | None = None) -> None:
    """One JSON line to stdout and to the cycle log.

    A logging failure must never take the cycle down with it: the point of
    the run is the decision, and losing the record of it is strictly less
    bad than not making it.
    """
    line = json.dumps({"ts": datetime.now(ET).isoformat(), **payload})
    target = CYCLE_LOG_PATH if path is None else path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    # The file comes FIRST and the console second. Under pythonw -- which is
    # what the scheduled task runs -- there is no console, sys.stdout is None
    # and print() raises; doing it the other way round threw before anything
    # was ever written, and the task failed silently with an empty log.
    if sys.stdout is not None:
        print(line)


def fetch_daily_bars(ib, contract) -> dict:
    """Daily bars for `contract` as the dict shape rsi2_signals expects.

    useRTH=True so the daily bar matches the regular-session bar the
    backtest was built on. ES trades nearly 24 hours; an overnight-inclusive
    daily bar has a different close and would compute a different RSI.
    """
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr=DAILY_BAR_LOOKBACK,
        barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1,
    )
    return {
        "date": [b.date for b in bars],
        "open": [float(b.open) for b in bars],
        "high": [float(b.high) for b in bars],
        "low": [float(b.low) for b in bars],
        "close": [float(b.close) for b in bars],
    }


def record_fill(trade, contract, side: str, contracts: int, reason: str,
                signal_bar_date: str | None) -> dict:
    """Trade-log row from a settled order. avgFillPrice is 0.0 on an
    accepted-but-unfilled order, which is recorded as-is rather than
    guessed at -- a zero fill price in the log means "check the broker"."""
    status = trade.orderStatus
    return {
        "timestamp_iso": datetime.now(ET).isoformat(),
        "symbol": contract.symbol,
        "local_symbol": getattr(contract, "localSymbol", "") or "",
        "side": side,
        "size": contracts,
        "fill_price": float(status.avgFillPrice or 0.0),
        "order_id": trade.order.orderId,
        "status": status.status,
        "reason": reason,
        "signal_bar_date": signal_bar_date or "",
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    now_et = datetime.now(ET)

    if not args.ignore_window and not rsi2_live.in_decision_window(now_et):
        log_event({"event": "outside_decision_window", "now_et": now_et.strftime("%a %H:%M"),
                   "window": f"{rsi2_live.DECISION_WINDOW_START_ET}-{rsi2_live.DECISION_WINDOW_END_ET} ET"})
        return 0

    try:
        rules = rsi2_live.load_rules()
    except (ValueError, json.JSONDecodeError) as e:
        log_event({"event": "bad_rules", "error": str(e)})
        return 1

    load_dotenv()
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PORT", "7497"))
    client_id = int(os.environ.get("IBKR_RSI2_CLIENT_ID", str(DEFAULT_CLIENT_ID)))

    from trading_bot.broker.ibkr_client import IBKRClient  # noqa: PLC0415 - keep the gate cheap

    try:
        ibkr = IBKRClient(host, port, client_id)
    except Exception as e:  # noqa: BLE001
        log_event({"event": "connect_failed", "error": str(e), "port": port})
        return 1

    try:
        positions = rsi2_live.load_positions()
        position = positions[0] if positions else None
        if len(positions) > 1:
            log_event({"event": "unexpected_multiple_positions", "count": len(positions)})
            return 1

        contract = ibkr.front_future(rules["symbol"], rules["exchange"], rules["min_days_to_expiry"])
        bars_all = fetch_daily_bars(ibkr.ib, contract)
        bars = rsi2_live.completed_bars(bars_all, now_et)
        decision = rsi2_live.decide(bars, position, rules)

        # Expiry overrides the signal: a held contract inside its expiry
        # window is closed regardless of what RSI says, because the
        # alternative is the broker liquidating it at a price of its choosing.
        forced = rsi2_live.expiry_action(position, date.today(), rules["min_days_to_expiry"]) if position else ""
        if forced:
            decision = {**decision, "action": "sell", "reason": forced,
                        "contracts": position.get("contracts", rules["contracts"])}

        log_event({
            "event": "decision", "armed": args.arm, "action": decision["action"],
            "reason": decision["reason"], "rsi": decision["rsi"], "dip": decision["dip"],
            "regime_ok": decision["regime_ok"],
            "signal_bar": decision["signal_bar_date"], "contracts": decision["contracts"],
            "bars_completed": len(bars["close"]),
            "held": None if position is None else position.get("local_symbol"),
            "contract": getattr(contract, "localSymbol", "") or contract.symbol,
        })

        if decision["action"] == "hold":
            write_heartbeat(rsi2_live.RSI2_HEARTBEAT_PATH, "hold")
            return 0

        if not args.arm:
            log_event({"event": "dry_run_no_order", "would": decision["action"]})
            write_heartbeat(rsi2_live.RSI2_HEARTBEAT_PATH, "dry_run")
            return 0

        side = "BUY" if decision["action"] == "buy" else "SELL"
        trade = ibkr.place_futures_order(contract, side, decision["contracts"])
        row = record_fill(trade, contract, side, decision["contracts"],
                          decision["reason"], decision["signal_bar_date"])
        rsi2_live.append_trade(row)
        log_event({"event": "order_placed", **row})

        if decision["action"] == "buy":
            rsi2_live.save_positions([{
                "symbol": contract.symbol,
                "local_symbol": row["local_symbol"],
                "expiry": contract.lastTradeDateOrContractMonth,
                "contracts": decision["contracts"],
                "entry_price": row["fill_price"],
                "entry_date": row["timestamp_iso"],
                "entry_reason": decision["reason"],
                "signal_bar_date": decision["signal_bar_date"],
            }])
        else:
            rsi2_live.save_positions([])

        write_heartbeat(rsi2_live.RSI2_HEARTBEAT_PATH, decision["action"])
        return 0
    except Exception as e:  # noqa: BLE001
        log_event({"event": "cycle_failed", "error": f"{type(e).__name__}: {e}"})
        return 1
    finally:
        try:
            ibkr.disconnect()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
