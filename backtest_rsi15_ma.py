#!/usr/bin/env python3
"""Backtest RSI(15) Mean Reversion strategy with and without 200-day MA filter.

Compares:
1. RSI(15) only (current)
2. RSI(15) + 200-day MA filter

Over 2 years of S&P 500 data.
"""

import sys
try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
except ImportError:
    print("ERROR: Required packages not installed. Install with:")
    print("pip install yfinance numpy pandas")
    sys.exit(1)


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


def backtest_strategy(data, use_ma=False):
    """Backtest strategy. Returns list of trades."""
    closes = data['Close'].values
    closes = [float(x) for x in closes]
    rsi_values = wilder_rsi(closes, period=15)

    trades = []
    position = None
    entry_price = 0
    entry_date = None

    # Calculate 200-day MA if needed
    ma200 = None
    if use_ma:
        ma200 = data['Close'].rolling(window=200).mean().values

    for i in range(1, len(closes)):
        prev_rsi = rsi_values[i-1]
        curr_rsi = rsi_values[i]
        curr_price = closes[i]
        curr_date = data.index[i]

        # BUY signal: RSI crosses below 60
        if position is None and prev_rsi > 60 and curr_rsi <= 60:
            # Check MA filter if enabled
            if use_ma and ma200[i] > 0:
                if curr_price < ma200[i]:  # Price below MA
                    continue  # Skip this signal

            position = "long"
            entry_price = curr_price
            entry_date = curr_date

        # SELL signal: RSI crosses above 65
        elif position == "long" and prev_rsi < 65 and curr_rsi >= 65:
            # Check MA filter if enabled
            if use_ma and ma200[i] > 0:
                if curr_price > ma200[i]:  # Price above MA
                    continue  # Skip this signal

            # Close the position
            exit_price = curr_price
            pnl = exit_price - entry_price
            pnl_pct = (pnl / entry_price) * 100
            trades.append({
                'entry_date': entry_date,
                'exit_date': curr_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            position = None

    return trades


def calculate_metrics(trades):
    """Calculate backtest metrics."""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade_pnl_pct': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'profit_factor': 0.0
        }

    pnls = [t['pnl'] for t in trades]
    pnls_pct = [t['pnl_pct'] for t in trades]

    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]

    total_win = sum(winning) if winning else 0
    total_loss = abs(sum(losing)) if losing else 0

    return {
        'total_trades': len(trades),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': (len(winning) / len(trades) * 100) if trades else 0,
        'total_pnl': sum(pnls),
        'total_pnl_pct': sum(pnls_pct),
        'avg_win': np.mean(winning) if winning else 0,
        'avg_loss': np.mean(losing) if losing else 0,
        'avg_trade_pnl_pct': np.mean(pnls_pct),
        'max_win': max(winning) if winning else 0,
        'max_loss': min(losing) if losing else 0,
        'profit_factor': total_win / total_loss if total_loss > 0 else 0
    }


def main():
    print("Downloading 2 years of S&P 500 data...")
    data = yf.download("^GSPC", period="2y", progress=False)

    print(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total candles: {len(data)}\n")

    # Backtest version 1: RSI only
    print("=" * 60)
    print("VERSION 1: RSI(15) ONLY")
    print("=" * 60)
    trades_v1 = backtest_strategy(data, use_ma=False)
    metrics_v1 = calculate_metrics(trades_v1)

    print(f"Total trades: {metrics_v1['total_trades']}")
    print(f"Winning trades: {metrics_v1['winning_trades']}")
    print(f"Losing trades: {metrics_v1['losing_trades']}")
    print(f"Win rate: {metrics_v1['win_rate']:.2f}%")
    print(f"Total P&L: ${metrics_v1['total_pnl']:.2f}")
    print(f"Total P&L %: {metrics_v1['total_pnl_pct']:.2f}%")
    print(f"Avg trade P&L %: {metrics_v1['avg_trade_pnl_pct']:.2f}%")
    print(f"Avg win: ${metrics_v1['avg_win']:.2f}")
    print(f"Avg loss: ${metrics_v1['avg_loss']:.2f}")
    print(f"Max win: ${metrics_v1['max_win']:.2f}")
    print(f"Max loss: ${metrics_v1['max_loss']:.2f}")
    print(f"Profit factor: {metrics_v1['profit_factor']:.2f}\n")

    # Backtest version 2: RSI + 200-day MA
    print("=" * 60)
    print("VERSION 2: RSI(15) + 200-DAY MA FILTER")
    print("=" * 60)
    trades_v2 = backtest_strategy(data, use_ma=True)
    metrics_v2 = calculate_metrics(trades_v2)

    print(f"Total trades: {metrics_v2['total_trades']}")
    print(f"Winning trades: {metrics_v2['winning_trades']}")
    print(f"Losing trades: {metrics_v2['losing_trades']}")
    print(f"Win rate: {metrics_v2['win_rate']:.2f}%")
    print(f"Total P&L: ${metrics_v2['total_pnl']:.2f}")
    print(f"Total P&L %: {metrics_v2['total_pnl_pct']:.2f}%")
    print(f"Avg trade P&L %: {metrics_v2['avg_trade_pnl_pct']:.2f}%")
    print(f"Avg win: ${metrics_v2['avg_win']:.2f}")
    print(f"Avg loss: ${metrics_v2['avg_loss']:.2f}")
    print(f"Max win: ${metrics_v2['max_win']:.2f}")
    print(f"Max loss: ${metrics_v2['max_loss']:.2f}")
    print(f"Profit factor: {metrics_v2['profit_factor']:.2f}\n")

    # Comparison
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Trades reduced: {metrics_v1['total_trades'] - metrics_v2['total_trades']} " +
          f"({((metrics_v1['total_trades'] - metrics_v2['total_trades']) / metrics_v1['total_trades'] * 100) if metrics_v1['total_trades'] > 0 else 0:.1f}%)")
    print(f"Win rate change: {metrics_v2['win_rate'] - metrics_v1['win_rate']:+.2f}%")
    print(f"Total P&L change: ${metrics_v2['total_pnl'] - metrics_v1['total_pnl']:+.2f}")
    print(f"Avg trade P&L % change: {metrics_v2['avg_trade_pnl_pct'] - metrics_v1['avg_trade_pnl_pct']:+.2f}%")
    print(f"Profit factor change: {metrics_v2['profit_factor'] - metrics_v1['profit_factor']:+.2f}\n")

    if metrics_v2['total_pnl'] > metrics_v1['total_pnl']:
        print("✅ VERSION 2 (with MA) is better")
    elif metrics_v2['total_pnl'] < metrics_v1['total_pnl']:
        print("❌ VERSION 1 (RSI only) is better")
    else:
        print("➖ Both versions are equal")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBacktest interrupted")
        sys.exit(0)
