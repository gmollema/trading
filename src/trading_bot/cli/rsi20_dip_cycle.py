"""Scheduled daily entrypoint for the RSI(20) dip strategy
(see rsi20_dip_live.py).

Runs ONCE per trading day a few minutes before the 09:30 ET open and does
nothing at any other time. Decides from the last COMPLETED daily bar and
fills with a market order at the open -- the same reachable fill spec
rsi2 uses, and the one find_rsi20_dip_trades models.

DRY RUN IS THE DEFAULT. This places orders on a real futures contract and
has never been run. It logs exactly what it would do and touches nothing
until invoked with --arm.

READ THIS BEFORE ARMING IT. The strategy loses to buy-and-hold on all
eight instruments tested, over windows up to 57 years. Its only edge is
return per unit of drawdown, which needs leverage to become money. And
`entry_level` is fitted to US large-cap: the 55-60 ridge does not exist
on the DAX, FTSE, Nikkei or Euro Stoxx. See
backtest/rsi20_dip_signals.py.

DO NOT ARM THIS WHILE rsi2 IS ARMED ON THE SAME CONTRACT. Both default to
MES and IBKR nets per contract, so the real exposure would be two
contracts while each bot's state file claims one. rsi2's task is disabled
as of 2026-09-07, so there is no conflict today.

Paper vs live is the PORT, as elsewhere in this repo: IBKR_PORT=7497 is
paper TWS (the default), 7496 is live. Nothing here special-cases either,
so pointing it at a live account is a one-variable change.

Run it manually first:
    python -m trading_bot.cli.rsi20_dip_cycle                 # dry run
    python -m trading_bot.cli.rsi20_dip_cycle --ignore-window  # dry run, any time
    python -m trading_bot.cli.rsi20_dip_cycle --arm            # actually sends orders

To schedule it once you trust it (deliberately NOT added to
setup_schedule.py, which would revive the disabled gap-and-go tasks as a
side effect). The window is wide because Task Scheduler works in local
time while 09:25-09:45 ET is 15:26 local most of the year and 14:26 in
the weeks US and EU clocks disagree:

    powershell -Command "$a=New-ScheduledTaskAction -Execute
      '<repo>\\.venv\\Scripts\\pythonw.exe' -Argument
      '-m trading_bot.cli.rsi20_dip_cycle' -WorkingDirectory '<repo>' ..."
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

from trading_bot import rsi20_dip_live
from trading_bot.util.heartbeat import write_heartbeat

ET = ZoneInfo("America/New_York")
# Its own client id: cycle.py uses the default, smc_cycle.py 4, rsi2 6,
# the data fetcher 95. Sharing one would have TWS drop a connection.
DEFAULT_CLIENT_ID = 7
DAILY_BAR_LOOKBACK = "2 Y"
# Written by log_event as well as stdout. The file comes FIRST and the
# console second: the scheduled task runs pythonw, where there is no
# console, sys.stdout is None and print() raises -- doing it the other way
# round loses the decision. Same fix as rsi2_cycle.
CYCLE_LOG_PATH = Path("logs/rsi20_dip_cycle.log")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true",
                        help="actually place orders; without this nothing is sent")
    parser.add_argument("--ignore-window", action="store_true",
                        help="skip the 09:25-09:45 ET gate (for manual dry runs only)")
    return parser.parse_args(argv)


def log_event(payload: dict, path: Path | None = None) -> None:
    """One JSON line to the cycle log and to stdout.

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
    if sys.stdout is not None:
        print(line)


def fetch_daily_bars(ib, contract) -> dict:
    """Daily bars for `contract` as the dict shape the signals expect.

    useRTH=True so the daily bar matches the regular-session bar the
    backtest was built on. MES trades nearly 24 hours; an
    overnight-inclusive daily bar has a different close and would compute
    a different RSI.
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
    accepted-but-unfilled order and is recorded as-is rather than guessed
    at -- a zero fill price in the log means "check the broker"."""
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

    if not args.ignore_window and not rsi20_dip_live.in_decision_window(now_et):
        log_event({"event": "outside_decision_window", "now_et": now_et.strftime("%a %H:%M"),
                   "window": f"{rsi20_dip_live.DECISION_WINDOW_START_ET}-"
                             f"{rsi20_dip_live.DECISION_WINDOW_END_ET} ET"})
        return 0

    try:
        rules = rsi20_dip_live.load_rules()
    except (ValueError, json.JSONDecodeError) as e:
        log_event({"event": "bad_rules", "error": str(e)})
        return 1

    load_dotenv()
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PORT", "7497"))
    client_id = int(os.environ.get("IBKR_RSI20_DIP_CLIENT_ID", str(DEFAULT_CLIENT_ID)))

    from trading_bot.broker.ibkr_client import IBKRClient  # noqa: PLC0415 - keep the gate cheap

    try:
        ibkr = IBKRClient(host, port, client_id)
    except Exception as e:  # noqa: BLE001
        log_event({"event": "connect_failed", "error": str(e), "port": port})
        return 1

    try:
        positions = rsi20_dip_live.load_positions(rsi20_dip_live.POSITIONS_PATH)
        position = positions[0] if positions else None
        if len(positions) > 1:
            log_event({"event": "unexpected_multiple_positions", "count": len(positions)})
            return 1

        contract = ibkr.front_future(rules["symbol"], rules["exchange"],
                                     rules["min_days_to_expiry"])
        bars_all = fetch_daily_bars(ibkr.ib, contract)
        bars = rsi20_dip_live.completed_bars(bars_all, now_et)
        decision = rsi20_dip_live.decide(bars, position, rules)

        # Expiry overrides the signal: a held contract inside its expiry
        # window is closed regardless of what RSI says, because the
        # alternative is the broker liquidating it at a price of its choosing.
        forced = rsi20_dip_live.expiry_action(
            position, date.today(), rules["min_days_to_expiry"]) if position else ""
        if forced:
            decision = {**decision, "action": "sell", "reason": forced,
                        "contracts": position.get("contracts", rules["contracts"])}

        log_event({
            "event": "decision", "armed": args.arm, "action": decision["action"],
            "reason": decision["reason"], "rsi": decision["rsi"],
            "trend_ok": decision["trend_ok"], "signal_bar": decision["signal_bar_date"],
            "contracts": decision["contracts"], "bars_completed": len(bars["close"]),
            "held": None if position is None else position.get("local_symbol"),
            "contract": getattr(contract, "localSymbol", "") or contract.symbol,
        })

        if decision["action"] == "hold":
            write_heartbeat(rsi20_dip_live.HEARTBEAT_PATH, "hold")
            return 0

        if not args.arm:
            log_event({"event": "dry_run_no_order", "would": decision["action"]})
            write_heartbeat(rsi20_dip_live.HEARTBEAT_PATH, "dry_run")
            return 0

        side = "BUY" if decision["action"] == "buy" else "SELL"
        trade = ibkr.place_futures_order(contract, side, decision["contracts"])
        row = record_fill(trade, contract, side, decision["contracts"],
                          decision["reason"], decision["signal_bar_date"])
        rsi20_dip_live.append_trade(row, rsi20_dip_live.TRADES_CSV_PATH)
        log_event({"event": "order_placed", **row})

        if decision["action"] == "buy":
            rsi20_dip_live.save_positions([{
                "symbol": contract.symbol,
                "local_symbol": row["local_symbol"],
                "expiry": contract.lastTradeDateOrContractMonth,
                "contracts": decision["contracts"],
                "entry_price": row["fill_price"],
                "entry_date": row["timestamp_iso"],
                "entry_reason": decision["reason"],
                "signal_bar_date": decision["signal_bar_date"],
            }], rsi20_dip_live.POSITIONS_PATH)
        else:
            rsi20_dip_live.save_positions([], rsi20_dip_live.POSITIONS_PATH)

        write_heartbeat(rsi20_dip_live.HEARTBEAT_PATH, decision["action"])
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
