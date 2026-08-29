"""What the SMC bot's fills actually cost, against what the backtest assumes.

Every SMC figure in this repo rests on per-leg slippage rates that are
mostly assumptions. smc_full_backtest.leg_slippage charges the entry, TP1
and the new-high exit a residual 2.0 bps under the next-bar fill spec, on
the reasoning that the fill bar already prices the chase and only spread
and impact are left. That number is borrowed from the stop leg -- a market
execution that does not chase -- and has never been measured on an entry.

It decides the answer. At the configured spec the strategy returns +0.93%
out of sample with commission; the same legs at the 48-79 bps the early
TP1 fills gave up would put it clearly negative. So this reads the live
log and reports the real distribution per leg, next to the rate the
backtest charged for it.

Expect it to be empty at first, and that is not a bug. The fields it reads
landed on 2026-08-27/28 (commits eb643ac, 80103eb), after every fill in
the log at that point, so the sample starts from the next trade rather
than from the ten entries already recorded.

Three legs, and they are not equally well founded:

  entry   signal_price (the OB high that triggered) against the fill.
      Pure assumption today -- zero samples.
  tp1     tp1_level against the fill, plus bars_since_touch, which is the
      lag this leg is actually paying for.
  stop    the resting stop level against the fill. Recoverable only by
      tracking the level forward through the events that move it
      (entry_opened -> tp1_done's new_stop -> stop_replaced), because
      stopped_out records the fill and not the level it was resting at.

Usage:
    python -m trading_bot.cli.smc_fill_audit
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from trading_bot import smc_live
from trading_bot.cli.smc_full_backtest import leg_slippage

# Which backtest leg each measurement should be compared against.
LEG_FOR = {"entry": "entry", "tp1": "tp1", "stop": "stop"}


def read_events(path: Path) -> list[dict]:
    """The safety log is JSON Lines, appended one event per cycle action.

    A truncated final line is skipped rather than fatal: the bot may be
    mid-write, and refusing to report on 200 good events because of one
    partial one would be the wrong trade.
    """
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def measure(events: list[dict]) -> dict[str, list[dict]]:
    """Per-leg slippage in bps, adverse-positive, from the live log.

    Adverse-positive throughout: a buy filled above its level and a sell
    filled below it both count as a cost, so the legs are comparable to
    each other and to the backtest's rates without sign juggling.
    """
    out: dict[str, list[dict]] = {"entry": [], "tp1": [], "stop": []}
    resting_stop: dict[str, float] = {}

    for event in events:
        name = event.get("event")
        symbol = event.get("symbol")

        if name == "entry_opened":
            if event.get("stop") is not None:
                resting_stop[symbol] = float(event["stop"])
            level, fill = event.get("signal_price"), event.get("entry_price")
            if level and fill:
                out["entry"].append({
                    "symbol": symbol, "when": event.get("timestamp_iso"),
                    "level": float(level), "fill": float(fill),
                    "bps": (float(fill) - float(level)) / float(level) * 10_000,
                })

        elif name == "tp1_done":
            if event.get("new_stop") is not None:
                resting_stop[symbol] = float(event["new_stop"])
            level, fill = event.get("tp1_level"), event.get("fill_price")
            if level and fill:
                out["tp1"].append({
                    "symbol": symbol, "when": event.get("timestamp_iso"),
                    "level": float(level), "fill": float(fill),
                    "bps": (float(level) - float(fill)) / float(level) * 10_000,
                    "bars_since_touch": event.get("bars_since_touch"),
                })

        elif name == "stop_replaced":
            if event.get("stop_price") is not None:
                resting_stop[symbol] = float(event["stop_price"])

        elif name == "stopped_out":
            level, fill = resting_stop.pop(symbol, None), event.get("fill_price")
            if level and fill:
                out["stop"].append({
                    "symbol": symbol, "when": event.get("timestamp_iso"),
                    "level": level, "fill": float(fill),
                    "bps": (level - float(fill)) / level * 10_000,
                })

    return out


def summarize(samples: list[dict]) -> dict | None:
    if not samples:
        return None
    bps = sorted(s["bps"] for s in samples)
    return {
        "n": len(bps),
        "median": statistics.median(bps),
        "mean": statistics.fmean(bps),
        "min": bps[0],
        "max": bps[-1],
    }


def verdict(stat: dict | None, assumed: float, min_samples: int) -> str:
    """What the measurement says about the rate the backtest charged.

    Deliberately refuses to call it on a handful of fills: the whole point
    of this tool is to replace an assumption with evidence, and three
    samples is not evidence.
    """
    if stat is None:
        return "no samples yet"
    if stat["n"] < min_samples:
        return f"only {stat['n']} fills -- too few to conclude (want {min_samples}+)"
    median = stat["median"]
    if median <= assumed * 1.5:
        return f"median {median:.1f} bps is at or under the assumed {assumed:.1f} -- the basis holds"
    if median <= assumed * 5:
        return f"median {median:.1f} bps runs {median / assumed:.1f}x the assumed {assumed:.1f} -- re-run the backtest at the measured rate"
    return f"median {median:.1f} bps dwarfs the assumed {assumed:.1f} -- the published figures do not survive this"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log", type=Path, default=smc_live.SMC_SAFETY_LOG_PATH)
    parser.add_argument("--rules", type=Path, default=smc_live.SMC_RULES_PATH)
    parser.add_argument("--min-samples", type=int, default=8,
                        help="fills needed before this will call a leg either way")
    parser.add_argument("--show", action="store_true", help="list every measured fill")
    args = parser.parse_args()

    events = read_events(args.log)
    if not events:
        print(f"no events in {args.log}")
        return 1

    rules = smc_live.load_smc_rules(args.rules)
    entry, exit_ = smc_live.entry_rules(rules), smc_live.exit_rules(rules)
    assumed = leg_slippage(entry["fill"], exit_["fill"], exit_["tp1_resting_limit"])
    measured = measure(events)

    print(f"{len(events)} events from {args.log}")
    print(f"backtest basis: entry_fill={entry['fill']} exit_fill={exit_['fill']}\n")
    print(f"{'leg':6s} {'n':>3s} {'median':>8s} {'mean':>8s} {'min':>8s} {'max':>8s} {'assumed':>8s}  verdict")
    for leg, samples in measured.items():
        stat = summarize(samples)
        rate = assumed[LEG_FOR[leg]]
        if stat is None:
            print(f"{leg:6s} {'0':>3s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>8s} {rate:>8.1f}  no samples yet")
            continue
        print(f"{leg:6s} {stat['n']:>3d} {stat['median']:>8.1f} {stat['mean']:>8.1f} "
              f"{stat['min']:>8.1f} {stat['max']:>8.1f} {rate:>8.1f}  "
              f"{verdict(stat, rate, args.min_samples)}")

    if args.show:
        for leg, samples in measured.items():
            for s in samples:
                lag = f"  lag {s['bars_since_touch']} bars" if s.get("bars_since_touch") is not None else ""
                print(f"  {leg:6s} {s['symbol']:6s} {s['when']}  "
                      f"level {s['level']:.2f} -> fill {s['fill']:.2f}  {s['bps']:+.1f} bps{lag}")

    print("\nPositive bps is adverse in every row: a buy filled above its level, a sell below it.")
    print("The entry leg is the one that matters -- its rate was never measured, only borrowed "
          "from the stop leg, and it is what the strategy's viability turns on.")
    print("If a leg comes in high, re-run: python -m trading_bot.cli.smc_full_backtest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
