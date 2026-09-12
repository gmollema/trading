#!/usr/bin/env python3
"""Daily RSI(15) signal checker for VUAA + Nasdaq 100 strategy.

Fetches S&P 500 data, calculates RSI(15), checks for buy/sell signals.
Run once daily (e.g., 4 PM ET after market close).

Usage:
    python -m trading_bot.cli.rsi15_signal_checker
    python -m trading_bot.cli.rsi15_signal_checker --verbose

Output: Logs signals to rsi15_signals.log + console
"""

import sys
from datetime import datetime
import yfinance as yf
import pandas as pd
from pathlib import Path


def wilder_rsi(closes: list[float], period: int) -> list[float | None]:
    """Wilder's RSI calculation (matches TradingView default)."""
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsi = [None] * period
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    seed_up = sum(d for d in deltas[:period] if d > 0) / period
    seed_down = -sum(d for d in deltas[:period] if d < 0) / period

    rs = seed_up / seed_down if seed_down else 0
    rsi.append(100 - (100 / (1 + rs)))

    for i in range(period + 1, len(closes)):
        up = deltas[i-1] if deltas[i-1] > 0 else 0
        down = -deltas[i-1] if deltas[i-1] < 0 else 0

        avg_up = (seed_up * (period - 1) + up) / period
        avg_down = (seed_down * (period - 1) + down) / period

        seed_up, seed_down = avg_up, avg_down
        rs = avg_up / avg_down if avg_down else 0
        rsi.append(100 - (100 / (1 + rs)))

    return rsi


def get_signal(data: pd.DataFrame, verbose: bool = False) -> str:
    """Check for entry (RSI < 60) or exit (RSI > 65) signals."""
    if len(data) < 20:
        return "INSUFFICIENT_DATA"

    closes = data['Close'].tolist()
    rsi_values = wilder_rsi(closes, period=15)

    prev_rsi = rsi_values[-2] if len(rsi_values) > 1 and rsi_values[-2] else None
    curr_rsi = rsi_values[-1] if rsi_values[-1] else None

    if prev_rsi is None or curr_rsi is None:
        return "INSUFFICIENT_RSI_DATA"

    if verbose:
        print(f"  RSI(15): prev={prev_rsi:.2f}, curr={curr_rsi:.2f}")

    # Check for crossings
    if prev_rsi > 60 and curr_rsi <= 60:
        return "BUY"
    elif prev_rsi < 65 and curr_rsi >= 65:
        return "SELL"
    else:
        return "NO_SIGNAL"


def main():
    verbose = "--verbose" in sys.argv
    log_path = Path("rsi15_signals.log")

    try:
        # Fetch S&P 500 data
        print(f"[{datetime.now().isoformat()}] Fetching S&P 500 data...")
        sp500 = yf.download("^GSPC", period="60d", progress=False)

        if sp500.empty or len(sp500) < 20:
            msg = "ERROR: Insufficient data"
            print(msg)
            with open(log_path, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] {msg}\n")
            return

        signal = get_signal(sp500, verbose=verbose)

        # Latest price and RSI
        latest_close = sp500['Close'].iloc[-1]
        print(f"  S&P 500: {latest_close:.2f}")
        print(f"  Signal: {signal}")

        # Log signal
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {signal}\n")

        # Alert on actual signals
        if signal == "BUY":
            print("\n*** BUY SIGNAL ***")
            print("Action: BUY both VUAA and Nasdaq 100 ETF on Trading 212")
            print("Target: 5% of capital per instrument (€25 each if €500 account)")
        elif signal == "SELL":
            print("\n*** SELL SIGNAL ***")
            print("Action: SELL both VUAA and Nasdaq 100 ETF positions on Trading 212")

    except Exception as e:
        msg = f"ERROR: {e}"
        print(msg)
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")


if __name__ == "__main__":
    main()
