"""Signal generation for the RSI(20) dip strategy, reconstructed from a
closed-source TradingView indicator ("Webinar Strategy") on 2026-09-07.

The reference was shared as a protected script, so only its inputs
(200, 3, 20, 60) and its Strategy Tester output were visible. The rules
below were recovered by comparing trade lists, not by reading code.

Rules, as recovered:

  Trend filter    The signal bar's close must be above SMA(200) as it
      stood `offset_bars` bars ago. The offset is the reference's own
      third input and its purpose is not obvious; it is reproduced
      because the reference has it, not because it helps. Removing it
      changes results only marginally.
  Entry           RSI(20) crosses DOWN through `entry_level` -- a
      pullback inside an uptrend. Fills at the next bar's open.
  Exit            RSI(20) crosses UP through `exit_level`, OR the close
      falls below the UNSHIFTED SMA(200). Fills at the next bar's open.
  Position count  One at a time. In-trade signals are ignored.

WHAT THE RECONSTRUCTION GOT WRONG, AND WHY IT MATTERS
-----------------------------------------------------
The user's own reconstruction had the entry and exit crossings inverted
-- it bought RSI crossing UP through the level and sold it crossing down,
making it a momentum system rather than a mean-reversion one. On the same
data that scored +34% against the reference's +158%. Correcting the
polarity alone took it to +153%. A 35% win rate against the reference's
82% is the tell: opposite payoff profiles mean opposite logic, and no
amount of parameter tuning closes that gap.

THE LEVELS MUST BE ASYMMETRIC
-----------------------------
`entry_level` and `exit_level` must differ. Set equal, the strategy
leaves on the mirror image of the condition it arrived on, so it churns:
on ES 2008-2026, 60/60 takes 186 trades for +178% while 60/65 takes 83
for +208%. The reference exposes only one threshold in its inputs, which
is almost certainly why the reconstruction wired one input to both ends.

Exit level is a broad plateau -- 65/70/75/80 all land within a few
percent -- so the specific value is not load-bearing. Entry level is
NOT: 55-60 is a sharp ridge on US indices and the peak moves to 40, 45
or 65 on the DAX, FTSE, Nikkei, AEX and Euro Stoxx. Treat entry_level as
fitted to US large-cap behaviour over 2008-2026, not as a discovered
constant. See cli/rsi20_dip_backtest.py --grid.

MEASURED, entry 60 / exit 65, 2008-2026, 100% of equity, no costs:

    instrument    CAGR    max DD   trades   win%    buy-and-hold CAGR
    Nasdaq 100   10.10%    24.4%      91    76.9%        15.18%
    S&P 500       6.55%    16.6%      83    74.7%         9.32%
    ES futures    6.21%    15.4%      83    73.5%         9.29%
    DAX           0.19%    34.3%      68    56%           6.49%
    FTSE 100      0.12%    33.8%      70    51%           2.81%
    Euro Stoxx   -0.60%    33.4%      54    48%           2.10%

It loses to buy-and-hold on every instrument tested, and only reduces
drawdown on the US ones. What it does have is a better return per unit
of drawdown (0.39 vs 0.17 on the S&P) and per unit of time invested
(13.1% vs 9.32%, being in the market ~50% of days). That is a
leverage-dependent edge, not a return edge -- see the same finding in
[[project-rsi2-dropped]] for the RSI(2) family.
"""

from __future__ import annotations

from trading_bot.backtest.rsi2_signals import simple_moving_average, wilder_rsi

DEFAULT_RSI_PERIOD = 20
DEFAULT_MA_PERIOD = 200
DEFAULT_OFFSET_BARS = 3
DEFAULT_ENTRY_LEVEL = 60.0
DEFAULT_EXIT_LEVEL = 65.0

REASON_RSI_EXIT = "rsi_exit"
REASON_TREND_BREAK = "trend_break"
REASON_END_OF_DATA = "end_of_data"


def _crossed_down(series: list[float | None], level: float, i: int) -> bool:
    """Pine's ta.crossunder(series, level) at bar i: strictly below now,
    at or above on the previous bar. A run of sub-level bars therefore
    produces ONE signal, on the first of them."""
    return (
        i > 0
        and series[i] is not None
        and series[i - 1] is not None
        and series[i - 1] >= level
        and series[i] < level
    )


def _crossed_up(series: list[float | None], level: float, i: int) -> bool:
    """Pine's ta.crossover(series, level) at bar i."""
    return (
        i > 0
        and series[i] is not None
        and series[i - 1] is not None
        and series[i - 1] <= level
        and series[i] > level
    )


def shifted_trend_ma(closes: list[float], ma_period: int, offset_bars: int) -> list[float | None]:
    """SMA(ma_period) as it stood `offset_bars` bars ago.

    Bars before the offset can see is None rather than falling back to
    the unshifted value: a placeholder there would let the trend filter
    pass on bars where the reference's filter has no opinion.
    """
    sma = simple_moving_average(closes, ma_period)
    return [None if i < offset_bars else sma[i - offset_bars] for i in range(len(closes))]


def find_rsi20_dip_trades(
    bars: dict,
    rsi_period: int = DEFAULT_RSI_PERIOD,
    ma_period: int = DEFAULT_MA_PERIOD,
    offset_bars: int = DEFAULT_OFFSET_BARS,
    entry_level: float = DEFAULT_ENTRY_LEVEL,
    exit_level: float = DEFAULT_EXIT_LEVEL,
) -> list[dict]:
    """Long-only walk over one symbol's daily bars, one position at a time.

    Args:
        bars: {"date","open","high","low","close"} -- equal-length lists.
        entry_level: RSI level the signal must cross DOWN through.
        exit_level: RSI level the signal must cross UP through. Must
            exceed entry_level; equal levels churn (see module docstring).

    Returns:
        One dict per trade with keys {"entry_idx","entry_date",
        "entry_price","exit_idx","exit_date","exit_price","bars_held",
        "points","pct","reason"}. `pct` is the simple return, which is
        what compounds in the equity walk; `points` is kept for parity
        with the rsi2 modules.

    Raises:
        ValueError: if exit_level does not exceed entry_level, or the
            periods are not positive.
    """
    if exit_level <= entry_level:
        raise ValueError(
            f"exit_level ({exit_level}) must exceed entry_level ({entry_level}) -- "
            "equal or inverted levels exit on the mirror of the entry and churn")
    if rsi_period < 1 or ma_period < 1:
        raise ValueError("rsi_period and ma_period must be >= 1")
    if offset_bars < 0:
        raise ValueError("offset_bars must be >= 0")

    dates, opens, closes = bars["date"], bars["open"], bars["close"]
    n = len(closes)
    if n == 0:
        return []

    rsi = wilder_rsi(closes, rsi_period)
    trend = simple_moving_average(closes, ma_period)
    shifted = shifted_trend_ma(closes, ma_period, offset_bars)

    out: list[dict] = []
    pos: dict | None = None
    pending_entry = False
    pending_exit_reason = ""

    def _close(i: int, price: float, reason: str) -> None:
        out.append({
            **pos,
            "exit_idx": i,
            "exit_date": dates[i],
            "exit_price": price,
            "bars_held": i - pos["entry_idx"],
            "points": price - pos["entry_price"],
            "pct": price / pos["entry_price"] - 1.0,
            "reason": reason,
        })

    for i in range(n):
        # 1. Orders signalled on the previous close are market orders and
        # fill at THIS bar's open -- Pine's default, and the only fill a
        # bot reading completed daily bars can actually reach.
        if pos is not None and pending_exit_reason:
            _close(i, opens[i], pending_exit_reason)
            pos, pending_exit_reason = None, ""
        if pending_entry and pos is None:
            pos = {"entry_idx": i, "entry_date": dates[i], "entry_price": opens[i]}
        pending_entry = False

        if rsi[i] is None or trend[i] is None or shifted[i] is None:
            continue

        # 2. Exit before entry, so an exit and a re-entry cannot collide
        # on one bar and so the trade order matches the reference's.
        if pos is not None:
            if _crossed_up(rsi, exit_level, i):
                pending_exit_reason = REASON_RSI_EXIT
            elif closes[i] < trend[i]:
                pending_exit_reason = REASON_TREND_BREAK
            continue

        # 3. A fresh crossing down through entry_level, in an uptrend.
        if closes[i] > shifted[i] and _crossed_down(rsi, entry_level, i):
            pending_entry = True

    if pos is not None:
        _close(n - 1, closes[n - 1], REASON_END_OF_DATA)
    return out


def equity_curve(bars: dict, trades: list[dict], start_idx: int = 0) -> list[dict]:
    """Mark-to-market equity, 100% of equity per trade, starting at 1.0.

    Marked to the close on every bar rather than only on exits: an open
    losing position is real risk, and a closed-trade-only curve reports a
    drawdown the account never experienced. Same reasoning as the
    open-equity basis used across the rsi2 work.
    """
    closes = bars["close"]
    at_entry = {t["entry_idx"]: t for t in trades}
    eq, pos = 1.0, None
    curve = []
    for i in range(start_idx, len(closes)):
        if pos is not None and pos["exit_idx"] == i:
            eq *= 1.0 + pos["pct"]
            pos = None
        if i in at_entry and pos is None:
            pos = at_entry[i]
            if pos["exit_idx"] == i:
                eq *= 1.0 + pos["pct"]
                pos = None
        marked = eq * (closes[i] / pos["entry_price"]) if pos is not None else eq
        curve.append({"date": bars["date"][i], "equity": marked, "in_market": pos is not None})
    return curve


def max_drawdown_pct(curve: list[dict]) -> float:
    peak, worst = 0.0, 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        if peak > 0:
            worst = max(worst, (peak - point["equity"]) / peak * 100.0)
    return worst
