"""Scheduled daily entrypoint for the RSI(20) dip strategy traded through
a leveraged ETF (see rsi20_dip_etf_live.py).

Runs ONCE per trading day a few minutes before the 09:30 ET open and does
nothing at any other time. Decides from the last COMPLETED daily bar of
the INDEX PROXY and fills the ETP with a market order at the US open --
the same reachable fill spec the futures variant uses, and the one
find_rsi20_dip_trades models.

WHY AN LSE-LISTED ETP STILL FILLS AT THE US OPEN
------------------------------------------------
09:30 ET is 14:30 London, the middle of the LSE session, so the order
fills while the ETP's market makers are quoting against live S&P
futures. That is the price the backtest assumes. Filling at the LSE
08:00 open instead would be a price set six and a half hours before the
signal's own reference market opened, which is a different strategy.

DRY RUN IS THE DEFAULT. This spends real money and has never been run.
It logs exactly what it would do and touches nothing until --arm.

READ THIS BEFORE ARMING IT. Over 2015-2026 the 3x version returned
12.96% against the index's 12.04%, with a 44% drawdown against its 34%:
about one point of CAGR for ten points of extra drawdown. The
full-sample win leans on being out of the market through 2008-09. See
rsi20_dip_etf_live.py for the measured table.

VERIFY THE INSTRUMENT FIRST:
    python -m trading_bot.cli.rsi20_dip_etf_cycle --check

--check resolves both contracts, reads account equity, prices the ETP off
its last completed daily bar and prints the share count, the idle cash
and the effective leverage -- without evaluating the signal or placing
anything. Run it before anything else: the default trade_symbol is a
UCITS 3x S&P listing that your account may not have permissions for, and
leveraged ETPs generally need a broker appropriateness test.

    python -m trading_bot.cli.rsi20_dip_etf_cycle                  # dry run
    python -m trading_bot.cli.rsi20_dip_etf_cycle --ignore-window  # dry run, any time
    python -m trading_bot.cli.rsi20_dip_etf_cycle --arm            # sends orders

Paper vs live is the PORT, as elsewhere in this repo: IBKR_PORT=7497 is
paper TWS (the default), 7496 is live.

DO NOT RUN THIS AND THE FUTURES VARIANT ON THE SAME EXPOSURE. They are
different instruments so IBKR will not net them, which is worse, not
better: you would hold 3x of the account in an ETP and $38,800 of index
in MES, each state file claiming to be the whole position.

To schedule it once you trust it (deliberately NOT added to
setup_schedule.py, which would revive the disabled gap-and-go tasks as a
side effect). The window is wide because Task Scheduler works in local
time while 09:25-09:45 ET is 15:26 local most of the year and 14:26 in
the weeks US and EU clocks disagree:

    powershell -Command "$a=New-ScheduledTaskAction -Execute
      '<repo>\\.venv\\Scripts\\pythonw.exe' -Argument
      '-m trading_bot.cli.rsi20_dip_etf_cycle' -WorkingDirectory '<repo>' ..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from trading_bot import rsi20_dip_etf_live as etf
from trading_bot.util.heartbeat import write_heartbeat

ET = ZoneInfo("America/New_York")
# Its own client id: cycle.py uses the default, smc_cycle.py 4, rsi2 6,
# the futures rsi20_dip 7, the data fetcher 95. Sharing one would have
# TWS drop a connection.
DEFAULT_CLIENT_ID = 8
DAILY_BAR_LOOKBACK = "2 Y"
# Written by log_event as well as stdout. The file comes FIRST and the
# console second: the scheduled task runs pythonw, where there is no
# console, sys.stdout is None and print() raises -- doing it the other way
# round loses the decision. Same fix as rsi2_cycle.
CYCLE_LOG_PATH = Path("logs/rsi20_dip_etf_cycle.log")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true",
                        help="actually place orders; without this nothing is sent")
    parser.add_argument("--ignore-window", action="store_true",
                        help="skip the 09:25-09:45 ET gate (for manual dry runs only)")
    parser.add_argument("--check", action="store_true",
                        help="resolve contracts and print sizing, then exit; "
                             "evaluates no signal and places no order")
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
    backtest was built on.
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


def last_completed_close(bars: dict, now_et: datetime) -> float:
    """The ETP's most recent completed daily close, used to size the order.

    A bar close, not a live quote: IBKRClient asks for delayed market
    data, and this repo has already been bitten by treating a delayed
    marketPrice() as current (see smc_cycle.get_current_price). A close
    from yesterday is late but factual, and `sizing_buffer` is what
    absorbs the gap between it and the fill.

    Raises:
        RuntimeError: if there is no completed bar to size from.
    """
    completed = etf.completed_bars(bars, now_et)
    if not completed["close"]:
        raise RuntimeError("no completed daily bar for the traded ETP -- cannot size")
    return float(completed["close"][-1])


def read_equity(ibkr) -> tuple[float | None, str]:
    """(equity, currency), or (None, "") if the summary is unavailable.

    Soft-failing because equity is a CAP here, not the sizing input: the
    configured allocation still applies without it (see usable_capital),
    so an account-summary hiccup should not skip a trading day.
    """
    try:
        return ibkr.net_liquidation()
    except Exception as e:  # noqa: BLE001
        log_event({"event": "equity_unavailable", "error": f"{type(e).__name__}: {e}"})
        return None, ""


def resolve_contracts(ibkr, rules: dict):
    """(signal_contract, trade_contract), both qualified."""
    signal = ibkr.qualify_stock(
        rules["signal_symbol"], rules["signal_exchange"], rules["signal_currency"],
        rules["signal_primary_exchange"])
    trade = ibkr.qualify_stock(
        rules["trade_symbol"], rules["trade_exchange"], rules["trade_currency"],
        rules["trade_primary_exchange"])
    return signal, trade


def record_fill(trade, contract, side: str, shares: float, reason: str,
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
        "size": shares,
        "fill_price": float(status.avgFillPrice or 0.0),
        "order_id": trade.order.orderId,
        "status": status.status,
        "reason": reason,
        "signal_bar_date": signal_bar_date or "",
    }


def run_check(ibkr, rules: dict, now_et: datetime) -> int:
    """Resolve, price and size without touching the signal or the market."""
    signal_contract, trade_contract = resolve_contracts(ibkr, rules)
    equity, equity_ccy = read_equity(ibkr)
    trade_bars = fetch_daily_bars(ibkr.ib, trade_contract)
    price = last_completed_close(trade_bars, now_et)
    capital = etf.usable_capital(rules, equity, equity_ccy)
    sized = etf.shares_for(capital, price, rules)

    signal_bars = etf.completed_bars(fetch_daily_bars(ibkr.ib, signal_contract), now_et)
    needed = etf.bars_needed(rules)

    log_event({
        "event": "check",
        "signal": f"{signal_contract.symbol}@{signal_contract.exchange}",
        "signal_bars_completed": len(signal_bars["close"]),
        "signal_bars_needed": needed,
        "signal_ready": len(signal_bars["close"]) >= needed,
        "trade": f"{trade_contract.symbol}@{trade_contract.exchange}",
        "trade_local_symbol": getattr(trade_contract, "localSymbol", "") or "",
        "trade_currency": trade_contract.currency,
        "etp_last_close": round(price, 4),
        "account_equity": equity,
        "account_currency": equity_ccy,
        "capital_used": round(capital, 2),
        "shares": sized["shares"],
        "size_increment": rules["size_increment"],
        "notional": round(sized["notional"], 2),
        "idle_cash": round(sized["idle_cash"], 2),
        "idle_pct": round(sized["idle_pct"], 1),
        "effective_leverage": round(etf.effective_leverage(sized, capital, rules), 2),
        "configured_leverage": rules["trade_leverage"],
    })
    if sized["shares"] <= 0:
        log_event({"event": "check_failed",
                   "error": f"capital {capital:.2f} buys nothing at {price:.2f} with "
                            f"size_increment {rules['size_increment']} -- add funds, or "
                            f"check the increment is really the contract's"})
        return 1
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    now_et = datetime.now(ET)

    if not args.check and not args.ignore_window and not etf.in_decision_window(now_et):
        log_event({"event": "outside_decision_window", "now_et": now_et.strftime("%a %H:%M"),
                   "window": f"{etf.DECISION_WINDOW_START_ET}-"
                             f"{etf.DECISION_WINDOW_END_ET} ET"})
        return 0

    try:
        rules = etf.load_rules(etf.RULES_PATH)
    except (ValueError, json.JSONDecodeError) as e:
        log_event({"event": "bad_rules", "error": str(e)})
        return 1

    load_dotenv()
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PORT", "7497"))
    client_id = int(os.environ.get("IBKR_RSI20_DIP_ETF_CLIENT_ID", str(DEFAULT_CLIENT_ID)))

    from trading_bot.broker.ibkr_client import IBKRClient  # noqa: PLC0415 - keep the gate cheap

    try:
        ibkr = IBKRClient(host, port, client_id)
    except Exception as e:  # noqa: BLE001
        log_event({"event": "connect_failed", "error": str(e), "port": port})
        return 1

    try:
        if args.check:
            return run_check(ibkr, rules, now_et)

        positions = etf.load_positions(etf.POSITIONS_PATH)
        position = positions[0] if positions else None
        if len(positions) > 1:
            log_event({"event": "unexpected_multiple_positions", "count": len(positions)})
            return 1

        signal_contract, trade_contract = resolve_contracts(ibkr, rules)

        # The signal is computed on the INDEX PROXY. Getting this wrong --
        # feeding the ETP's own bars to decide() -- would not error, it
        # would quietly evaluate the fitted 60/65 levels against a series
        # whose daily moves are three times as large.
        signal_bars = etf.completed_bars(
            fetch_daily_bars(ibkr.ib, signal_contract), now_et)

        # A held position carries the share count it was opened with, so
        # an exit sells exactly what is held rather than a freshly sized
        # quantity -- the price has moved since entry, and a re-sized
        # sell would leave a residue or go short.
        decision = etf.decide_action(signal_bars, position, rules)

        capital = None
        sized = None
        if decision["action"] == "buy":
            equity, equity_ccy = read_equity(ibkr)
            capital = etf.usable_capital(rules, equity, equity_ccy)
            trade_bars = fetch_daily_bars(ibkr.ib, trade_contract)
            sized = etf.shares_for(capital, last_completed_close(trade_bars, now_et), rules)
            decision = {**decision, "shares": sized["shares"]}
            if sized["shares"] <= 0:
                log_event({"event": "buy_skipped_unfundable",
                           "capital": round(capital, 2),
                           "size_increment": rules["size_increment"],
                           "reason": "capital buys less than one size increment"})
                write_heartbeat(etf.HEARTBEAT_PATH, "unfundable")
                return 0

        log_event({
            "event": "decision", "armed": args.arm, "action": decision["action"],
            "reason": decision["reason"], "rsi": decision["rsi"],
            "trend_ok": decision["trend_ok"], "signal_bar": decision["signal_bar_date"],
            "shares": decision.get("shares", 0),
            "bars_completed": len(signal_bars["close"]),
            "held": None if position is None else position.get("shares"),
            "signal_symbol": signal_contract.symbol,
            "trade_symbol": trade_contract.symbol,
            "capital": None if capital is None else round(capital, 2),
            "idle_pct": None if sized is None else round(sized["idle_pct"], 1),
            "effective_leverage": (
                None if sized is None
                else round(etf.effective_leverage(sized, capital, rules), 2)),
        })

        if decision["action"] == "hold":
            write_heartbeat(etf.HEARTBEAT_PATH, "hold")
            return 0

        if not args.arm:
            log_event({"event": "dry_run_no_order", "would": decision["action"],
                       "shares": decision.get("shares", 0)})
            write_heartbeat(etf.HEARTBEAT_PATH, "dry_run")
            return 0

        # float, not int: fractional orders are the whole point of the
        # XS2D default, and int() would floor 1.3572 shares to 1.
        shares = float(decision.get("shares", 0.0))
        if shares <= 0:
            log_event({"event": "order_skipped", "reason": "no shares to trade",
                       "action": decision["action"]})
            write_heartbeat(etf.HEARTBEAT_PATH, "no_shares")
            return 1

        side = "BUY" if decision["action"] == "buy" else "SELL"
        trade = ibkr.place_stock_order(trade_contract, side, shares)
        row = record_fill(trade, trade_contract, side, shares,
                          decision["reason"], decision["signal_bar_date"])
        etf.append_trade(row, etf.TRADES_CSV_PATH)
        log_event({"event": "order_placed", **row})

        if decision["action"] == "buy":
            etf.save_positions([{
                "symbol": trade_contract.symbol,
                "local_symbol": row["local_symbol"],
                "exchange": trade_contract.exchange,
                "currency": trade_contract.currency,
                "shares": shares,
                "entry_price": row["fill_price"],
                "entry_date": row["timestamp_iso"],
                "entry_reason": decision["reason"],
                "signal_bar_date": decision["signal_bar_date"],
            }], etf.POSITIONS_PATH)
        else:
            etf.save_positions([], etf.POSITIONS_PATH)

        write_heartbeat(etf.HEARTBEAT_PATH, decision["action"])
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
