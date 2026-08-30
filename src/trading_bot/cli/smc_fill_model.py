"""Estimate what limit-order entries would and would not have filled.

Written to test a limit entry resting at the order block's high, which
smc_cycle briefly used (commit 5ee5816, reverted in a561a2d on the
strength of this result). A limit removes a measured cost -- the
market-order legs gave up 48.5 and 79.4 bps on TP1 against a median stop
distance of 17.3 bps -- but introduces an unmeasured one: it only fills if
price comes back to the level, and it fails precisely on the setups that
turn straight around off it. If the trades limits MISS are systematically
better than the ones they catch, the cost saving is bought back through
adverse selection. They are, by a wide margin.

The cohort split this found is now a strategy rule rather than only a
diagnostic: smc_signals' require_ob_reclaim takes the retests that close
back ABOVE the level (this module's "missed" group) and skips the ones
that close below it. See smc_entry_spec for what that is worth once the
entry is priced at a fill an order can actually get.

find_smc_long_trades fills every entry the moment a bar's low touches
ob_high, which is a touch, not a fill. This asks the counterfactual from
the outside, without changing that shared function (the live bot calls it).

Two models bracket the answer rather than pretending to one number. The
limit was placed after the signal bar closed and cancelled about 7% of a
5-minute bar later, so:

  immediate: the signal bar CLOSED at or below ob_high, so price is
      already at the limit when the order arrives and it fills at once.
      Pessimistic: it ignores any dip back inside the order's short life.

  next_bar: the following bar trades at or below ob_high at some point.
      Optimistic: it grants the order the whole bar, far longer than it
      actually rests, and a bar low says nothing about WHEN in the bar.

Truth sits between them. A limit is not carried across a session boundary
(the strategy force-closes daily and nothing rests overnight), so a signal
on a day's final bar simply misses under next_bar.

Known limitation: this classifies the trades find_smc_long_trades already
produced. That function holds one position per symbol at a time, so a
missed entry would in reality free capacity for a later one it currently
suppresses -- meaning the filled cohort's aggregate return understates
limit-order trading. The cohort COMPARISON, which is the actual question
here, is unaffected: both groups come from the same generated set.

Usage:
    python -m trading_bot.cli.smc_fill_model \
        --intraday-dir backtest_data/intraday_5m_merged_2026-08-27
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_bot.backtest.data import load_intraday
from trading_bot.backtest.smc_signals import find_smc_long_trades
from trading_bot.data.sp500_tickers import SP500_TICKERS

FILL_MODELS = ("immediate", "next_bar")


def would_fill(bars: dict, entry_idx: int, level: float, model: str) -> bool:
    """Would a limit buy resting at `level` have filled after `entry_idx`?

    `bars` is the dict of lists find_smc_long_trades consumes. `entry_idx`
    is the bar whose low touched the level -- the bar the live bot acts on
    once it closes.
    """
    if model not in FILL_MODELS:
        raise ValueError(f"unknown fill model: {model!r}; expected one of {list(FILL_MODELS)}")

    if model == "immediate":
        return bars["close"][entry_idx] <= level

    nxt = entry_idx + 1
    if nxt >= len(bars["low"]):
        return False
    # Nothing rests overnight: the strategy force-closes daily, and a limit
    # placed at 15:55 is not live at the next open.
    if bars["date"][nxt].date() != bars["date"][entry_idx].date():
        return False
    return bars["low"][nxt] <= level


def trade_return_pct(trade: dict) -> float:
    """Net return over the entry notional, summed across the fill ladder."""
    entry = trade["entry_price"]
    if entry <= 0:
        return 0.0
    return sum(f["qty_fraction"] * (f["price"] - entry) / entry for f in trade["fills"]) * 100


def classify_trades(bars: dict, trades: list[dict], model: str) -> list[dict]:
    """Tag each trade with whether a limit at its entry level would fill."""
    out = []
    for t in trades:
        # signal_price, not entry_price: the question is where a limit
        # would have rested, which is the OB high the retest triggered on.
        # They coincide only under the "level" fill spec, and entry_price
        # is whatever the configured spec paid.
        level = t["signal_price"]
        out.append({
            "entry_idx": t["entry_idx"],
            "entry_date": t["entry_date"],
            "filled": would_fill(bars, t["entry_idx"], level, model),
            "ret_pct": trade_return_pct(t),
            "hit_tp1": any(f["reason"] == "tp1" for f in t["fills"]),
            "stopped": bool(t["fills"]) and t["fills"][0]["reason"] == "stop",
        })
    return out


def bars_dict(df: pd.DataFrame) -> dict:
    return {
        "open": df["open"].tolist(), "high": df["high"].tolist(),
        "low": df["low"].tolist(), "close": df["close"].tolist(),
        "date": df["date"].tolist(),
    }


def analyse(tickers, intraday_dir: Path, rules: dict, models=FILL_MODELS) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        df = load_intraday(ticker, intraday_dir)
        if df is None or df.empty:
            continue
        bars = bars_dict(df)
        # Pinned to "level" rather than taking the default: this module's
        # committed cohort figures were produced on it, and the question
        # it asks -- where a resting limit would have filled -- is defined
        # against the trigger price, not against whatever the configured
        # spec now pays.
        trades = find_smc_long_trades(
            bars,
            rules["time_window_bars"],
            rules["tp1_fraction"],
            rules["swing_window"],
            entry_fill="level",
            exit_fill="level",
        )
        if not trades:
            continue
        for model in models:
            for row in classify_trades(bars, trades, model):
                rows.append({"symbol": ticker, "model": model, **row})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Fill rate and cohort outcomes per model -- the adverse-selection read."""
    out = []
    for model, g in df.groupby("model"):
        filled, missed = g[g.filled], g[~g.filled]
        out.append({
            "model": model,
            "signals": len(g),
            "fill_rate_pct": round(len(filled) / len(g) * 100, 2) if len(g) else float("nan"),
            "filled_mean_ret": round(filled.ret_pct.mean(), 4) if len(filled) else float("nan"),
            "missed_mean_ret": round(missed.ret_pct.mean(), 4) if len(missed) else float("nan"),
            "filled_win_pct": round((filled.ret_pct > 0).mean() * 100, 2) if len(filled) else float("nan"),
            "missed_win_pct": round((missed.ret_pct > 0).mean() * 100, 2) if len(missed) else float("nan"),
            "filled_tp1_pct": round(filled.hit_tp1.mean() * 100, 2) if len(filled) else float("nan"),
            "missed_tp1_pct": round(missed.hit_tp1.mean() * 100, 2) if len(missed) else float("nan"),
        })
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intraday-dir", type=Path,
                        default=Path("backtest_data/intraday_5m_merged_2026-08-27"))
    parser.add_argument("--rules", type=Path, default=Path("smc_rules.json"))
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override list")
    parser.add_argument("--out", type=Path, default=Path("smc_fill_model_results.csv"))
    args = parser.parse_args()

    if not args.intraday_dir.is_dir():
        print(f"error: no such intraday dir: {args.intraday_dir}")
        return 2

    rules = json.loads(args.rules.read_text())
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else list(SP500_TICKERS)

    print(f"classifying entries across {len(tickers)} tickers...")
    df = analyse(tickers, args.intraday_dir, rules)
    if df.empty:
        print("no trades found")
        return 1
    df.to_csv(args.out, index=False)

    summary = summarize(df)
    print("\n===== LIMIT-ORDER FILL MODEL =====")
    print(summary.to_string(index=False))
    print("\nadverse selection: missed_mean_ret ABOVE filled_mean_ret means the "
          "trades a limit declines are the better ones.")
    for _, r in summary.iterrows():
        gap = r["missed_mean_ret"] - r["filled_mean_ret"]
        verdict = "ADVERSE (missed trades were better)" if gap > 0 else "favourable (missed trades were worse)"
        print(f"  {r['model']:10s} fill {r['fill_rate_pct']:5.1f}%  gap {gap:+.4f}pp  -> {verdict}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
