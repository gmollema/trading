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
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Install with: pip install yfinance")
    sys.exit(1)

from trading_bot.util.notifier import notify


def wilder_rsi(closes: list[float], period: int) -> list[float]:
    """Wilder's RSI calculation (matches TradingView default)."""
    if len(closes) < period + 1:
        return [0.0] * len(closes)

    rsi = [0.0] * period
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


def get_signal(closes: list[float], verbose: bool = False) -> tuple[str, list[float]]:
    """Check for entry (RSI < 60) or exit (RSI > 65) signals. Returns (signal, rsi_values)."""
    if len(closes) < 20:
        return "INSUFFICIENT_DATA", []

    rsi_values = wilder_rsi(closes, period=15)

    prev_rsi = rsi_values[-2]
    curr_rsi = rsi_values[-1]

    if verbose:
        print(f"  RSI(15): prev={prev_rsi:.2f}, curr={curr_rsi:.2f}")

    # Check for crossings
    if prev_rsi > 60 and curr_rsi <= 60:
        return "BUY", rsi_values
    elif prev_rsi < 65 and curr_rsi >= 65:
        return "SELL", rsi_values
    else:
        return "NO_SIGNAL", rsi_values


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

        # yfinance returns multi-index DataFrame, extract the ^GSPC column
        closes = sp500['Close']['^GSPC'].values.tolist()
        signal, rsi_values = get_signal(closes, verbose=verbose)

        # Latest price
        latest_close = closes[-1]
        print(f"  S&P 500: {latest_close:.2f}")
        print(f"  Signal: {signal}")

        # Log signal
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {signal}\n")

        # Alert on actual signals
        if signal == "BUY":
            print("\n*** BUY SIGNAL ***")
            msg = f"BUY: VUAA + Nasdaq 100 ETF\nS&P 500: {latest_close:.2f}\nRSI(15): {rsi_values[-1]:.2f}"
            print(msg)
            print("Target: €250 each (5% of €500 capital, or adjust to your size)")
            notify("RSI(15) BUY SIGNAL", msg, priority="high")
        elif signal == "SELL":
            print("\n*** SELL SIGNAL ***")
            msg = f"SELL: VUAA + Nasdaq 100 ETF\nS&P 500: {latest_close:.2f}\nRSI(15): {rsi_values[-1]:.2f}"
            print(msg)
            notify("RSI(15) SELL SIGNAL", msg, priority="high")

    except Exception as e:
        msg = f"ERROR: {type(e).__name__}: {e}"
        print(msg)
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")


if __name__ == "__main__":
    main()
