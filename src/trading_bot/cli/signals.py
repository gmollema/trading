"""Compact daily view of all 3 strategies (RSI2, MA 30/90, Relative Strength).

Usage:
    python -m trading_bot.cli.signals [capital] [spx_pct] [qqq_pct]
    python -m trading_bot.cli.signals 1000 5 2.5

Downloads S&P 500 and Nasdaq once, prints one table of signals, the open
positions from trades.csv / trades_ma.csv / trades_rs.csv, and a list of
actions with the dollar amount per trade (capital * pct).

Signals are computed on the indexes (S&P 500, Nasdaq); the trades are placed in
Trading 212 on the UCITS ETFs that follow them (SXR8, SXRV). Entry prices in
the trade CSVs are index levels and stop losses are index points, so BUY lines
print the index level to log and the stop loss as a % to set on the ETF.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday,
)

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
NAMES = {SPX: "SXR8", NDX: "SXRV"}  # Trading 212 ETFs: iShares Core S&P 500 / Nasdaq 100 (EUR, Xetra)
INDEX = {SPX: "S&P-500", NDX: "Nasdaq"}
STOP_POINTS = {SPX: 225.0, NDX: 125.0}
RSI2_HOLD_DAYS = 12
RS_MAX_HOLD_DAYS = 60  # test_relative_strength.py exits after 60 trading days regardless of the ratio
COLORS = {"BUY": GREEN, "SETUP": YELLOW, "SELL": RED}


def colored(text: str, width: int) -> str:
    """Pad to width, then color by the leading signal word."""
    color = COLORS.get(text.split()[0], "")
    return f"{color}{text:<{width}}{RESET}" if color else f"{text:<{width}}"


class NYSEHolidays(AbstractHolidayCalendar):
    """Full-day NYSE closures (Yahoo's daily data can also skip days, so it can't be used to count)."""
    rules = [
        Holiday("New Year", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def trading_days_since(entry_date: str) -> int:
    """NYSE trading days after entry_date up to and including today, as the backtest counts min_hold_days."""
    start = np.datetime64(entry_date) + 1
    end = np.datetime64(datetime.now().date()) + 1
    holidays = NYSEHolidays().holidays(start=entry_date, end=str(end)).values.astype("datetime64[D]")
    return int(np.busday_count(start, end, holidays=holidays)) if end > start else 0


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
    parser.add_argument("capital", nargs="?", type=float, default=1000.0, help="account size in $ (default 1000)")
    parser.add_argument("spx_pct", nargs="?", type=float, default=5.0, help="%% of capital per S&P 500 (SXR8) trade (default 5)")
    parser.add_argument("qqq_pct", nargs="?", type=float, default=2.5, help="%% of capital per Nasdaq (SXRV) trade (default 2.5)")
    args = parser.parse_args()

    size = {SPX: args.capital * args.spx_pct / 100, NDX: args.capital * args.qqq_pct / 100}

    # Download before printing anything so legends and signals appear together
    try:
        closes = fetch_closes()
    except Exception as e:
        print(f"{RED}[ERROR] Could not download prices: {e}{RESET}")
        return
    price = {sym: closes[sym][-1] for sym in closes}

    print_legends()

    print(f"\n{BOLD}=== SIGNALS {datetime.now():%Y-%m-%d %H:%M} ==={RESET}  capital ${args.capital:,.0f}"
          f" | {NAMES[SPX]} {args.spx_pct:g}% = ${size[SPX]:.2f} | {NAMES[NDX]} {args.qqq_pct:g}% = ${size[NDX]:.2f}")

    # --- Signals -------------------------------------------------------------
    rsi2 = {sym: rsi2_check(sym, {"close": closes[sym]}) for sym in closes}
    ma = {sym: ma_check(sym, {"close": closes[sym]}) for sym in closes}
    rs = rs_signal_from_closes(closes[SPX], closes[NDX])

    def rsi2_cell(r: dict) -> str:
        trend = "" if r["price_above_sma"] else " <SMA"
        return f"{r['signal']} {r['today_rsi']:.0f}/{r['entry_level']:g}{trend}"

    def ma_cell(m: dict) -> str:
        word = {"HOLD": "UPTREND", "WAIT": "DOWNTREND"}.get(m["signal"], m["signal"])
        return f"{word} {m['today_ma30']:.0f}/{m['today_ma90']:.0f}"

    rs_word = {"HOLD": "UPTREND", "WAIT": "DOWNTREND"}.get(rs["signal"], rs["signal"])
    rs_cell = f"{rs_word} {rs['ratio_distance']:+.1f}%" if "ratio_distance" in rs else rs_word

    # Row label = strategy name + the format of the numbers in its cells
    label = lambda name, fmt: f"{name:<10}{DIM}{fmt:<12}{RESET}"
    head = {sym: f"{NAMES[sym]} {INDEX[sym]} {price[sym]:.2f}" for sym in (SPX, NDX)}
    print(f"\n{'':<22}{head[SPX]:<22}{head[NDX]}")
    print(f"{label('RSI(2)', 'now/entry')}{colored(rsi2_cell(rsi2[SPX]), 22)}{colored(rsi2_cell(rsi2[NDX]), 0)}")
    print(f"{label('MA 30/90', '30d/90d')}{colored(ma_cell(ma[SPX]), 22)}{colored(ma_cell(ma[NDX]), 0)}")
    print(f"{label('RS', 'vs 20d avg')}{DIM}{NAMES[NDX] + ' only':<22}{RESET}{colored(rs_cell, 0)}")

    def buy_line(sym: str, source: str) -> str:
        """Stop loss as % (works on the ETF) plus the index level to log as entry_price."""
        stop_pct = STOP_POINTS[sym] / price[sym] * 100
        return (f"BUY  {NAMES[sym]}  ${size[sym]:.2f}  stop loss -{stop_pct:.1f}%"
                f"  log {INDEX[sym]} {price[sym]:.2f}  ({source})")

    actions: list[str] = []
    for sym in (SPX, NDX):
        if rsi2[sym]["signal"] == "BUY":
            actions.append(buy_line(sym, "RSI2 -> trades.csv"))
        if ma[sym]["signal"] == "BUY":
            actions.append(buy_line(sym, "MA 30/90 -> trades_ma.csv"))
    if rs["signal"] == "BUY":
        actions.append(buy_line(NDX, "RS -> trades_rs.csv"))

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
        days = trading_days_since(row["entry_date"])
        pct = (price[sym] - entry) / entry * 100
        stop_pts = float(row.get("stop_loss") or STOP_POINTS[sym])

        if price[sym] <= entry - stop_pts:
            verdict = "SELL stop loss hit"
        elif strat == "RSI2":
            if days >= RSI2_HOLD_DAYS and pct > 0:
                verdict = "SELL 12+ trading days & profitable"
            else:
                verdict = f"HOLD {RSI2_HOLD_DAYS - days} trading days to exit window" if days < RSI2_HOLD_DAYS else "HOLD not profitable yet"
        elif strat == "MA":
            verdict = "SELL 30-MA < 90-MA" if ma_state(closes[sym]) == "SELL" else "HOLD"
        else:
            if days >= RS_MAX_HOLD_DAYS:
                verdict = f"SELL {RS_MAX_HOLD_DAYS}+ trading days held"
            elif rs["signal"] in ("SELL", "WAIT"):
                verdict = "SELL ratio < MA"
            else:
                verdict = f"HOLD {RS_MAX_HOLD_DAYS - days} trading days to max hold"

        print(f"  {strat:<5}{NAMES[sym]}  {entry:>9.2f} -> {price[sym]:<9.2f} {pct:+5.1f}% ${amount * pct / 100:+7.2f}"
              f"  {days:>3}td  {colored(verdict, 0)}")
        if verdict.startswith("SELL"):
            actions.append(f"SELL {NAMES[sym]}  today {datetime.now():%Y-%m-%d}  ({strat} bought {row['entry_date']}: {verdict[5:]})")

    # --- Actions -------------------------------------------------------------
    print(f"\n{BOLD}ACTIONS{RESET}")
    for a in actions:
        print(f"  {colored(a, 0)}")
    if not actions:
        print(f"  {DIM}none{RESET}")
    else:
        notify("Trading signals", "\n".join(actions))


LEGENDS = (
    (
        "RSI(2)",
        (
            ("WAIT", "RSI(2) is above the entry level", "Stay out"),
            ("SETUP", "RSI(2) is below the entry level, but it crossed on an earlier day", "Watch, don't enter"),
            ("BUY", "RSI(2) crossed below the entry level today and price is above its long-term average", "Enter"),
            ("<SMA", "Price is below its long-term average (175-day S&P, 250-day Nasdaq)",
             "No new buys, even on a dip (not a sell signal)"),
            ("SELL", "Open position held 12+ trading days and in profit, or stop loss hit (see POSITIONS)", "Exit"),
        ),
        (
            "RSI(2) measures the last 2 days' moves on a 0-100 scale; below 10 (S&P) or 7 (Nasdaq)"
            " means a sharp short-term dip.",
            "The strategy buys that dip only in a long-term uptrend, and sells after 12+ trading days once"
            " profitable, or at the stop loss.",
        ),
    ),
    (
        "MA 30/90",
        (
            ("DOWNTREND", "30-day average is below the 90-day", "Stay out"),
            ("BUY", "30-day average crossed above the 90-day today", "Enter"),
            ("UPTREND", "30-day average is still above the 90-day, but the cross was on an earlier day",
             "Hold if you're in, don't enter if you're not"),
            ("SELL", "30-day average crossed below the 90-day today", "Exit"),
        ),
        (
            "Golden cross (BUY): the fast 30-day average moves up past the slow 90-day,"
            " so recent prices have climbed above the longer-term level.",
            "Death cross (SELL): the 30-day average drops below the 90-day,"
            " so recent prices have fallen below the longer-term level.",
        ),
    ),
    (
        "RELATIVE STRENGTH (SXRV only)",
        (
            ("DOWNTREND", "Nasdaq/S&P ratio is below its 20-day average", "Stay out"),
            ("BUY", "Ratio crossed above its 20-day average today", "Enter"),
            ("UPTREND", "Ratio is still above its 20-day average, but the cross was on an earlier day",
             "Hold if you're in, don't enter if you're not"),
            ("SELL", "Ratio crossed below its 20-day average today", "Exit"),
            ("SELL", "Open position held 60+ trading days, whatever the ratio (see POSITIONS)", "Exit"),
        ),
        (
            "Trades SXRV (Nasdaq) only. Ratio = Nasdaq / S&P 500, where the S&P is just the benchmark;"
            " +1.6% means the ratio is 1.6% above its 20-day average.",
            "Golden cross (BUY): Nasdaq starts outperforming. Death cross (SELL): Nasdaq starts underperforming.",
            "Why no S&P trades: buying the S&P when it outperforms Nasdaq was backtested (2021-2026)"
            " and did no better than simply holding the S&P.",
        ),
    ),
)


def print_legends() -> None:
    # One width for all blocks so the "What to do" column lines up
    width = max(len(meaning) for _, rows, _ in LEGENDS for _, meaning, _ in rows) + 3
    for name, rows, explanation in LEGENDS:
        print(f"\n{BOLD}=== LEGEND - {name} ==={RESET}")
        for line in explanation:
            print(f"  {line}")
        print(f"\n  {DIM}{'State':<11}{'Condition':<{width}}What to do{RESET}")
        for state, meaning, todo in rows:
            print(f"  {colored(state, 11)}{meaning:<{width}}{todo}")


if __name__ == "__main__":
    main()
