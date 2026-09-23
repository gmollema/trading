"""Compact daily view of all 3 strategies (RSI2, MA 30/90, Relative Strength).

Usage:
    python -m trading_bot.cli.signals [capital] [spx_pct] [qqq_pct]
    python -m trading_bot.cli.signals 1000 5 2.5

Downloads S&P 500 and Nasdaq once, prints one table of signals, the open
positions from trades.csv / trades_ma.csv / trades_rs.csv, and a list of
actions with the dollar amount per trade (capital * pct).
"""

from __future__ import annotations

import argparse
import csv
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf

from trading_bot.backtest.rsi2_signals import simple_moving_average
from trading_bot.cli.ma_crossover_signals import check_signal as ma_check
from trading_bot.cli.relative_strength_signals import rs_signal_from_closes
from trading_bot.cli.rsi2_daily_signals import check_signal as rsi2_check
from trading_bot.util.notifier import notify

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

SPX, NDX = "^GSPC", "^IXIC"
NAMES = {SPX: "SPY", NDX: "QQQ"}
STOP_POINTS = {SPX: 225.0, NDX: 125.0}
RSI2_HOLD_DAYS = 12
COLORS = {"BUY": GREEN, "SETUP": YELLOW, "SELL": RED}


def colored(text: str, width: int) -> str:
    """Pad to width, then color by the leading signal word."""
    color = COLORS.get(text.split()[0], "")
    return f"{color}{text:<{width}}{RESET}" if color else f"{text:<{width}}"


def fetch_closes() -> dict[str, list[float]]:
    """~310 aligned daily closes per index (enough for RSI2's SMA(250))."""
    df = yf.download([SPX, NDX], period="450d", progress=False)["Close"].dropna()
    return {sym: [float(x) for x in df[sym]] for sym in (SPX, NDX)}


def ma_state(closes: list[float]) -> str:
    """MA 30/90 state for an open position: SELL on death cross or while 30 < 90."""
    ma30, ma90 = simple_moving_average(closes, 30), simple_moving_average(closes, 90)
    return "HOLD" if ma30[-1] > ma90[-1] else "SELL"


def load_csv(name: str) -> list[dict]:
    path = Path(name)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("entry_price")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capital", nargs="?", type=float, default=500.0, help="account size in $ (default 500)")
    parser.add_argument("spx_pct", nargs="?", type=float, default=5.0, help="%% of capital per S&P 500 trade (default 5)")
    parser.add_argument("qqq_pct", nargs="?", type=float, default=2.5, help="%% of capital per Nasdaq trade (default 2.5)")
    args = parser.parse_args()

    size = {SPX: args.capital * args.spx_pct / 100, NDX: args.capital * args.qqq_pct / 100}

    try:
        closes = fetch_closes()
    except Exception as e:
        print(f"{RED}[ERROR] Could not download prices: {e}{RESET}")
        return
    price = {sym: closes[sym][-1] for sym in closes}

    print(f"{BOLD}SIGNALS {datetime.now():%Y-%m-%d %H:%M}{RESET}  capital ${args.capital:,.0f}"
          f" | SPY {args.spx_pct:g}% = ${size[SPX]:.2f} | QQQ {args.qqq_pct:g}% = ${size[NDX]:.2f}")

    # --- Signals -------------------------------------------------------------
    rsi2 = {sym: rsi2_check(sym, {"close": closes[sym]}) for sym in closes}
    ma = {sym: ma_check(sym, {"close": closes[sym]}) for sym in closes}
    rs = rs_signal_from_closes(closes[SPX], closes[NDX])

    def rsi2_cell(r: dict) -> str:
        trend = "" if r["price_above_sma"] else " <SMA"
        return f"{r['signal']} {r['today_rsi']:.0f}/{r['entry_level']:g}{trend}"

    ma_word = lambda m: {"HOLD": "UPTREND", "WAIT": "DOWNTREND"}.get(m["signal"], m["signal"])
    rs_word = {"HOLD": "UPTREND", "WAIT": "DOWNTREND"}.get(rs["signal"], rs["signal"])
    rs_cell = f"{rs_word} {rs['ratio_distance']:+.1f}%" if "ratio_distance" in rs else rs_word

    print(f"\n{'':<10}{'SPY ' + format(price[SPX], '.2f'):<22}{'QQQ ' + format(price[NDX], '.2f')}")
    print(f"{'RSI(2)':<10}{colored(rsi2_cell(rsi2[SPX]), 22)}{colored(rsi2_cell(rsi2[NDX]), 0)}")
    print(f"{'MA 30/90':<10}{colored(ma_word(ma[SPX]), 22)}{colored(ma_word(ma[NDX]), 0)}")
    print(f"{'RS':<10}{DIM}{'-':<22}{RESET}{colored(rs_cell, 0)}")

    actions: list[str] = []
    for sym in (SPX, NDX):
        stop = price[sym] - STOP_POINTS[sym]
        buy = f"BUY  {NAMES[sym]}  ${size[sym]:.2f}  stop {stop:.2f}"
        if rsi2[sym]["signal"] == "BUY":
            actions.append(f"{buy}  (RSI2 -> trades.csv)")
        if ma[sym]["signal"] == "BUY":
            actions.append(f"{buy}  (MA 30/90 -> trades_ma.csv)")
    if rs["signal"] == "BUY":
        actions.append(f"BUY  QQQ  ${size[NDX]:.2f}  stop {price[NDX] - STOP_POINTS[NDX]:.2f}  (RS -> trades_rs.csv)")

    # --- Open positions --------------------------------------------------------
    positions = (
        [("RSI2", row.get("symbol", SPX), row) for row in load_csv("trades.csv")]
        + [("MA", row.get("symbol", SPX), row) for row in load_csv("trades_ma.csv")]
        + [("RS", NDX, row) for row in load_csv("trades_rs.csv")]
    )
    if positions:
        print(f"\n{BOLD}POSITIONS{RESET}")
    for strat, sym, row in positions:
        entry, amount = float(row["entry_price"]), float(row["entry_amount"])
        days = (datetime.now() - datetime.strptime(row["entry_date"], "%Y-%m-%d")).days
        pct = (price[sym] - entry) / entry * 100
        stop_pts = float(row.get("stop_loss") or STOP_POINTS[sym])

        if price[sym] <= entry - stop_pts:
            verdict = "SELL stop hit"
        elif strat == "RSI2":
            if days >= RSI2_HOLD_DAYS and pct > 0:
                verdict = "SELL 12+ days & profitable"
            else:
                verdict = f"HOLD {max(RSI2_HOLD_DAYS - days, 0)}d to exit window" if days < RSI2_HOLD_DAYS else "HOLD not profitable yet"
        elif strat == "MA":
            verdict = "SELL 30-MA < 90-MA" if ma_state(closes[sym]) == "SELL" else "HOLD"
        else:
            verdict = "SELL ratio < MA" if rs["signal"] in ("SELL", "WAIT") else "HOLD"

        print(f"  {strat:<5}{NAMES[sym]}  {entry:>9.2f} -> {price[sym]:<9.2f} {pct:+5.1f}% ${amount * pct / 100:+7.2f}"
              f"  {days:>3}d  {colored(verdict, 0)}")
        if verdict.startswith("SELL"):
            actions.append(f"SELL {NAMES[sym]}  {strat} position from {row['entry_date']}  ({verdict[5:]})")

    # --- Actions -------------------------------------------------------------
    print(f"\n{BOLD}ACTIONS{RESET}")
    for a in actions:
        print(f"  {colored(a, 0)}")
    if not actions:
        print(f"  {DIM}none{RESET}")
    else:
        notify("Trading signals", "\n".join(actions))


if __name__ == "__main__":
    main()
