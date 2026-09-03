"""Dollar-level position sizing and cost modeling for the 2-period RSI
strategy on CME equity-index futures (see rsi2_signals.py for the rules).

Deliberately does NOT reuse portfolio.position_size. That function sizes
SHARES: risk per unit is `price - stop` and its second cap is a fraction
of portfolio value divided by share price. Neither holds for a futures
contract, where risk per unit is `(price - stop) * multiplier` and the
binding capital constraint is margin per contract, which has no fixed
relationship to the contract's notional value. Silently feeding futures
into the share-based function would undersize by the multiplier -- 5x for
MES, 50x for ES.

Two consequences of futures sizing that stock backtests never surface,
both reported by cli/rsi2_backtest.py rather than hidden:

  Contracts are integral. There is no fractional MES. A signal whose
  risk-based size floors to zero is not a small position, it is a SKIPPED
  TRADE, and a backtest that quietly sizes it at 1 contract anyway is
  reporting a strategy nobody with that account could have traded. Those
  skips are counted.

  The video's 200-point stop costs $1,000 of risk per MES contract
  (200 * $5). At 1% risk per trade that implies a ~$100K account before a
  single contract is takeable, which is the strategy's real capital floor
  and is nowhere in the video.

Cost and margin defaults are ORDER-OF-MAGNITUDE ASSUMPTIONS, not quoted
figures, and every one is a parameter:

  MES_COMMISSION_PER_SIDE ~ $0.62/contract/side, IBKR's ~$0.25 commission
      plus ~$0.37 of exchange and regulatory pass-throughs.
  MES_MARGIN_PER_CONTRACT ~ $2,400 overnight initial. This one genuinely
      moves: CME raises and cuts it with volatility and with the index
      level, and it has spent time anywhere from roughly $1,200 to
      $2,600. It only binds for small accounts, and where it binds the
      result is sensitive to it -- check it against the current figure
      before drawing a conclusion at low capital.
  DEFAULT_SLIPPAGE_TICKS = 1 tick (0.25 index points, $1.25 on MES) per
      side. Entries are market orders at the open and exits at the close
      or open, so crossing a one-tick spread is the realistic assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TICK_SIZE = 0.25
MES_MULTIPLIER = 5.0
ES_MULTIPLIER = 50.0
MES_COMMISSION_PER_SIDE = 0.62
ES_COMMISSION_PER_SIDE = 2.32
MES_MARGIN_PER_CONTRACT = 2400.0
ES_MARGIN_PER_CONTRACT = 24000.0
DEFAULT_SLIPPAGE_TICKS = 1.0
DEFAULT_RISK_PCT = 1.0
# Margin a single position may consume. Well below 100% because margin is
# a maintenance floor, not a budget: sizing to the margin limit means a
# routine adverse move triggers a liquidation rather than the strategy's
# own stop.
DEFAULT_MAX_MARGIN_PCT = 25.0


@dataclass(frozen=True)
class ContractSpec:
    """One futures contract's economics. `name` is descriptive only."""

    name: str
    multiplier: float
    commission_per_side: float
    margin_per_contract: float
    tick_size: float = TICK_SIZE


MES = ContractSpec("MES", MES_MULTIPLIER, MES_COMMISSION_PER_SIDE, MES_MARGIN_PER_CONTRACT)
ES = ContractSpec("ES", ES_MULTIPLIER, ES_COMMISSION_PER_SIDE, ES_MARGIN_PER_CONTRACT)
CONTRACTS = {"MES": MES, "ES": ES}


def contracts_for_trade(
    equity: float,
    entry_price: float,
    stop_price: float | None,
    spec: ContractSpec,
    risk_pct: float = DEFAULT_RISK_PCT,
    max_margin_pct: float = DEFAULT_MAX_MARGIN_PCT,
) -> int:
    """Whole contracts to take: the lesser of a risk-based and a
    margin-based size, floored.

    Returns 0 when the account cannot fund even one contract at the
    requested risk -- the caller must treat that as a skipped trade, not
    round it up.

    `stop_price` of None (the strategy run without a stop) has no
    definable risk per contract, so risk sizing is impossible and only
    the margin cap applies. That is not a safe way to trade this system
    -- an unstopped first-profitable-close position has unbounded
    adverse excursion (see rsi2_signals on why its win rate is a
    tautology) -- so it is sized at the margin cap and left to the
    caller to interpret.
    """
    by_margin = equity * (max_margin_pct / 100.0) / spec.margin_per_contract
    if stop_price is None:
        return max(0, math.floor(by_margin))
    risk_per_contract = (entry_price - stop_price) * spec.multiplier
    if risk_per_contract <= 0:
        return 0
    by_risk = equity * (risk_pct / 100.0) / risk_per_contract
    return max(0, math.floor(min(by_risk, by_margin)))


def round_costs(points: float, contracts: int, spec: ContractSpec, slippage_ticks: float) -> tuple[float, float]:
    """(commission_dollars, slippage_dollars) for one round trip.

    Both legs pay, hence the doubling: two fills per round trip.
    Slippage is charged in ticks of adverse fill on each leg, which is
    the honest model for a market order into a one-tick spread.
    """
    commission = 2 * contracts * spec.commission_per_side
    slippage = 2 * contracts * slippage_ticks * spec.tick_size * spec.multiplier
    return commission, slippage


def run_rsi2_futures_backtest(
    point_trades: list[dict],
    bars: dict,
    initial_capital: float,
    spec: ContractSpec = MES,
    risk_pct: float = DEFAULT_RISK_PCT,
    max_margin_pct: float = DEFAULT_MAX_MARGIN_PCT,
    slippage_ticks: float = DEFAULT_SLIPPAGE_TICKS,
) -> dict:
    """Walk already-generated point trades, sizing each against the equity
    standing when it opened and charging costs on both legs.

    Sizing compounds: each trade is sized off the equity at ITS entry, so
    a drawdown shrinks subsequent positions and a run-up grows them.
    That coupling is the whole point of asking the dollar question --
    the point-level backtest is invariant to it.

    `bars` is used only to mark the open position to market for the
    equity curve, which is what makes the reported drawdown a real one
    rather than the closed-trade figure a first-profitable-close exit
    flatters (see rsi2_signals).

    Returns {"trades", "equity_curve", "skipped_trades", "final_equity",
    "locked_out_from"}.
    `trades` carries the point trade's fields plus contracts, gross_pnl,
    commission, slippage, net_pnl and equity_after. Skipped trades are
    counted, not returned: the account could not take them.

    `locked_out_from` is the date of the first trade after which sizing
    never recovered to one contract, or None. This is not a rare edge:
    whole-contract risk sizing makes zero contracts an ABSORBING STATE.
    An account sitting near the one-contract boundary takes one loss,
    floors to zero, and can then never place another trade to earn the
    equity back -- so the strategy is dead while the backtest keeps
    politely reporting a small negative return. A run with a
    `locked_out_from` date is not a result about the strategy, it is a
    result about the account being too small for the stop, and the two
    must not be confused.
    """
    closes = bars["close"]
    equity = initial_capital
    out_trades: list[dict] = []
    equity_curve: list[dict] = []
    skipped = 0
    lockout_from = None

    for trade in point_trades:
        contracts = contracts_for_trade(
            equity, trade["entry_price"], trade["stop_price"], spec, risk_pct, max_margin_pct
        )
        if contracts < 1:
            skipped += 1
            if lockout_from is None:
                lockout_from = trade["entry_date"]
            continue
        # Sizing recovered, so the earlier skips were a lull, not a
        # lockout. Only a skip run that reaches the end of the data is one.
        lockout_from = None

        # Mark to market across the holding period, before the realized
        # result lands. The exit bar is not marked: its P&L arrives at the
        # actual fill, below.
        for i in range(trade["entry_idx"], trade["exit_idx"]):
            unrealized = (closes[i] - trade["entry_price"]) * spec.multiplier * contracts
            equity_curve.append({"date": bars["date"][i], "equity": equity + unrealized})

        gross = trade["points"] * spec.multiplier * contracts
        commission, slippage = round_costs(trade["points"], contracts, spec, slippage_ticks)
        net = gross - commission - slippage
        equity += net
        out_trades.append({
            **trade,
            "contracts": contracts,
            "gross_pnl": round(gross, 2),
            "commission": round(commission, 2),
            "slippage": round(slippage, 2),
            "net_pnl": round(net, 2),
            "equity_after": round(equity, 2),
        })
        equity_curve.append({"date": trade["exit_date"], "equity": equity})

    return {
        "trades": out_trades,
        "equity_curve": equity_curve,
        "skipped_trades": skipped,
        "final_equity": equity,
        "locked_out_from": lockout_from,
    }


def max_drawdown_pct(equity_curve: list[dict]) -> float:
    """Peak-to-trough of the mark-to-market equity curve, in percent."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]["equity"]
    worst = 0.0
    for point in equity_curve:
        peak = max(peak, point["equity"])
        if peak > 0:
            worst = max(worst, (peak - point["equity"]) / peak * 100.0)
    return worst


def cagr_pct(initial: float, final: float, years: float) -> float:
    """Compound annual growth rate. Returns 0 for a non-positive final
    equity (a blown account has no meaningful growth rate) or a
    degenerate span."""
    if initial <= 0 or final <= 0 or years <= 0:
        return 0.0
    return ((final / initial) ** (1.0 / years) - 1.0) * 100.0
