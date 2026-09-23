"""Dollar P&L for the index backtests at the trade sizes actually used.

The live signal view (trading_bot.cli.signals) buys a fixed dollar amount of
the ETF per trade: 5% of capital for the S&P 500 (SXR8) and 2.5% for Nasdaq
(SXRV), on a $1,000 default account. A trade's dollar P&L is that amount
times the index's percentage move.

The old test_*.py scripts used index points times a fixed $/point ($25/225
and $12/125). That is the P&L of a position sized so the stop loses $25,
i.e. roughly $500-$850 on the S&P and $1,500-$2,600 on Nasdaq -- 30-100x
the $25/$12.50 actually bought -- which inflated the estimates to ~$370/yr.
"""

from __future__ import annotations

from datetime import datetime

CAPITAL = 1000.0
POSITION_PCT = {"^GSPC": 5.0, "^IXIC": 2.5}


def position_size(symbol: str, capital: float = CAPITAL) -> float:
    """Dollars bought per trade for this index's ETF."""
    return capital * POSITION_PCT[symbol] / 100


def window_years(start: str, last_date: datetime) -> float:
    """Length of the backtest window from `start` (YYYY-MM-DD) to the last bar."""
    first = datetime.strptime(start, "%Y-%m-%d").date()
    return (last_date.date() - first).days / 365.25


def trade_dollars(trade: dict, symbol: str, capital: float = CAPITAL) -> float:
    """Dollar P&L of one trade: position size x percentage move, no costs."""
    return position_size(symbol, capital) * (trade["exit_price"] / trade["entry_price"] - 1)


def annual_dollars(trades: list[dict], symbol: str, years: float, capital: float = CAPITAL) -> float:
    """Average dollar P&L per year over the window (fixed size per trade, not compounded)."""
    return sum(trade_dollars(t, symbol, capital) for t in trades) / years
