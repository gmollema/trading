#!/usr/bin/env python3
"""Continuous RSI(15) signal bot. Start once, runs forever.

Checks for signals once daily at a set time (default: 22:00 CET).
Stays running in background, sends Telegram alerts on signals.

Usage:
    python -m trading_bot.cli.rsi15_signal_bot
    python -m trading_bot.cli.rsi15_signal_bot --check-time 22:00  # Custom time (HH:MM)

Press Ctrl+C to stop.
"""

import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Install with: pip install yfinance")
    sys.exit(1)

from trading_bot.util.notifier import notify


def wilder_rsi(closes: list[float], period: int) -> list[float]:
    """Wilder's RSI calculation."""
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


def check_signal() -> tuple[str, list[float], float]:
    """Check for signal. Returns (signal, rsi_values, latest_close)."""
    try:
        sp500 = yf.download("^GSPC", period="60d", progress=False)

        if sp500.empty or len(sp500) < 20:
            return "ERROR", [], 0.0

        closes = sp500['Close']['^GSPC'].values.tolist()
        rsi_values = wilder_rsi(closes, period=15)

        prev_rsi = rsi_values[-2]
        curr_rsi = rsi_values[-1]
        latest_close = closes[-1]

        # Check for crossings
        if prev_rsi > 60 and curr_rsi <= 60:
            return "BUY", rsi_values, latest_close
        elif prev_rsi < 65 and curr_rsi >= 65:
            return "SELL", rsi_values, latest_close
        else:
            return "NO_SIGNAL", rsi_values, latest_close

    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Error checking signal: {e}")
        return "ERROR", [], 0.0


def main():
    # Parse check time from args
    check_time_str = "22:00"  # Default: 10 PM CET
    if "--check-time" in sys.argv:
        idx = sys.argv.index("--check-time")
        if idx + 1 < len(sys.argv):
            check_time_str = sys.argv[idx + 1]

    try:
        check_hour, check_minute = map(int, check_time_str.split(":"))
        check_time = dt_time(check_hour, check_minute)
    except ValueError:
        print(f"ERROR: Invalid time format '{check_time_str}'. Use HH:MM (e.g., 22:00)")
        sys.exit(1)

    log_path = Path("rsi15_signals.log")

    print(f"[{datetime.now().isoformat()}] RSI(15) Signal Bot started")
    print(f"[{datetime.now().isoformat()}] Check time: {check_time_str} CET")
    print(f"[{datetime.now().isoformat()}] Instruments: VUAA + Nasdaq 100 ETF")
    print(f"[{datetime.now().isoformat()}] Telegram alerts: ON")
    print(f"[{datetime.now().isoformat()}] Press Ctrl+C to stop\n")

    last_check_date = None

    while True:
        now = datetime.now()
        current_time = now.time()

        # Check if it's time to run (once per day)
        if current_time >= check_time and last_check_date != now.date():
            print(f"[{now.isoformat()}] Checking signal...")

            signal, rsi_values, latest_close = check_signal()

            if signal != "ERROR" and rsi_values:
                print(f"[{now.isoformat()}] S&P 500: {latest_close:.2f}, RSI(15): {rsi_values[-1]:.2f}, Signal: {signal}")

                # Log it
                with open(log_path, "a") as f:
                    f.write(f"[{now.isoformat()}] {signal}\n")

                # Alert on signals
                if signal == "BUY":
                    msg = f"BUY: VUAA + Nasdaq 100 ETF\nS&P 500: {latest_close:.2f}\nRSI(15): {rsi_values[-1]:.2f}"
                    notify("RSI(15) BUY SIGNAL", msg, priority="high")
                    print("*** BUY SIGNAL - Telegram alert sent ***\n")

                elif signal == "SELL":
                    msg = f"SELL: VUAA + Nasdaq 100 ETF\nS&P 500: {latest_close:.2f}\nRSI(15): {rsi_values[-1]:.2f}"
                    notify("RSI(15) SELL SIGNAL", msg, priority="high")
                    print("*** SELL SIGNAL - Telegram alert sent ***\n")

                last_check_date = now.date()

            else:
                print(f"[{now.isoformat()}] Failed to fetch data, retrying tomorrow\n")
                last_check_date = now.date()

        # Sleep 1 minute to avoid busy-waiting
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().isoformat()}] Bot stopped by user")
        sys.exit(0)
