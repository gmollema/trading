"""Daily performance summary for the IBKR paper-trading bot.

Reads trades.csv, FIFO-pairs BUY/SELL rows by symbol for today's Eastern
Time date, computes per-pair P&L and aggregate stats, prints a JSON
summary to stdout, and pushes a Telegram/ntfy notification via
src.notify.notify().

Does NOT modify trades.csv.

Usage:
    python -m trading_bot.cli.compute_perf   (best run after market close, ~16:05 ET)
"""

from __future__ import annotations

import html
import json
import sys
import traceback
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from trading_bot import smc_live
from trading_bot.util.notifier import notify

# The SMC bot's state, not the decommissioned gap-and-go bot's
# (trades.csv / open_positions.json / safety-check-log.json). Reading
# gap-and-go meant every daily summary said "No closed trades today"
# no matter what SMC did: its trades.csv last gained a row on 2026-07-16
# and holds no closed round trips at all, so pair_trades_fifo always came
# back empty. Same oversight as commit bb47ba0, which stopped the
# heartbeat monitor paging about that bot but left this one reading it.
TRADES_CSV_PATH = smc_live.SMC_TRADES_CSV_PATH
POSITIONS_PATH = smc_live.SMC_POSITIONS_PATH
SAFETY_LOG_PATH = smc_live.SMC_SAFETY_LOG_PATH
DASHBOARD_DIR = Path("dashboard")
DASHBOARD_PATH = DASHBOARD_DIR / "index.html"
LOGS_DIR = Path("logs")
COMPUTE_PERF_ERRORS_LOG = LOGS_DIR / "compute_perf_errors.log"
ET = ZoneInfo("America/New_York")

R_BUCKET_LABELS = [
    "\u2264 -2R",
    "(-2R, -1R]",
    "(-1R, 0R]",
    "(0R, +1R]",
    "(+1R, +2R]",
    "(+2R, +3R]",
    "> +3R",
]


def load_today_trades() -> pd.DataFrame:
    if not TRADES_CSV_PATH.exists():
        return pd.DataFrame(
            columns=["timestamp_iso", "symbol", "side", "size", "fill_price", "order_id", "status"]
        )

    df = pd.read_csv(TRADES_CSV_PATH, dtype=str)
    if df.empty:
        return df

    today_et = datetime.now(ET).strftime("%Y-%m-%d")
    today_df = df[df["timestamp_iso"].astype(str).str.startswith(today_et)].copy()

    # Coerce numeric columns; keep timestamp_iso as string for parsing.
    for col in ("size", "fill_price"):
        if col in today_df.columns:
            today_df[col] = pd.to_numeric(today_df[col], errors="coerce")

    return today_df


def _parse_ts(ts_str: str) -> datetime:
    # trades.csv timestamps are written as UTC ISO 8601 by trade.py.
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def pair_trades_fifo(df: pd.DataFrame) -> list[dict]:
    """FIFO-pair BUY rows with SELL rows by symbol. Returns list of closed
    pair dicts: symbol, pnl_usd, pnl_pct, hold_minutes, buy_price, sell_price.
    """
    if df.empty:
        return []

    df_sorted = df.sort_values("timestamp_iso")

    open_buys: dict[str, deque] = defaultdict(deque)
    closed_pairs: list[dict] = []

    for _, row in df_sorted.iterrows():
        symbol = row.get("symbol")
        side = row.get("side")
        size = row.get("size")
        fill_price = row.get("fill_price")
        ts_str = row.get("timestamp_iso")

        if symbol is None or side is None or pd.isna(size) or pd.isna(fill_price):
            continue

        try:
            ts = _parse_ts(str(ts_str))
        except (ValueError, TypeError):
            continue

        if side == "BUY":
            open_buys[symbol].append({"size": float(size), "price": float(fill_price), "ts": ts})
        elif side == "SELL":
            remaining_to_close = float(size)
            sell_price = float(fill_price)
            sell_ts = ts

            while remaining_to_close > 0 and open_buys[symbol]:
                buy_lot = open_buys[symbol][0]
                matched_size = min(buy_lot["size"], remaining_to_close)

                buy_price = buy_lot["price"]
                pnl_usd = (sell_price - buy_price) * matched_size
                pnl_pct = ((sell_price - buy_price) / buy_price * 100) if buy_price else 0.0
                hold_minutes = (sell_ts - buy_lot["ts"]).total_seconds() / 60.0

                closed_pairs.append(
                    {
                        "symbol": symbol,
                        "qty": matched_size,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "pnl_usd": pnl_usd,
                        "pnl_pct": pnl_pct,
                        "hold_minutes": hold_minutes,
                        "sell_ts": sell_ts.isoformat(),
                    }
                )

                buy_lot["size"] -= matched_size
                remaining_to_close -= matched_size
                if buy_lot["size"] <= 0:
                    open_buys[symbol].popleft()
            # Any leftover SELL size with no matching BUY lot is ignored
            # (e.g. a stray/duplicate row) rather than crashing the summary.

    return closed_pairs


def aggregate(closed_pairs: list[dict]) -> dict:
    total_trades = len(closed_pairs)

    if total_trades == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "gross_pnl_usd": 0.0,
            "largest_winner": None,
            "largest_loser": None,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "profit_factor": "n/a",
        }

    wins = [p for p in closed_pairs if p["pnl_usd"] > 0]
    losses = [p for p in closed_pairs if p["pnl_usd"] < 0]

    gross_pnl_usd = sum(p["pnl_usd"] for p in closed_pairs)
    win_rate_pct = (len(wins) / total_trades * 100) if total_trades else 0.0

    # Drawn from wins/losses specifically, NOT closed_pairs as a whole --
    # otherwise an all-losing day would label its least-bad loss "Best",
    # and an all-winning day would label its smallest win "Worst".
    largest_winner = max(wins, key=lambda p: p["pnl_usd"]) if wins else None
    largest_loser = min(losses, key=lambda p: p["pnl_usd"]) if losses else None

    avg_winner = (sum(p["pnl_usd"] for p in wins) / len(wins)) if wins else 0.0
    avg_loser = (sum(p["pnl_usd"] for p in losses) / len(losses)) if losses else 0.0

    sum_wins = sum(p["pnl_usd"] for p in wins)
    sum_losses = abs(sum(p["pnl_usd"] for p in losses))
    if sum_losses == 0:
        profit_factor = "inf" if sum_wins > 0 else "n/a"
    else:
        profit_factor = round(sum_wins / sum_losses, 2)

    return {
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate_pct, 2),
        "gross_pnl_usd": round(gross_pnl_usd, 2),
        "largest_winner": (
            {"symbol": largest_winner["symbol"], "pnl_usd": round(largest_winner["pnl_usd"], 2)}
            if largest_winner
            else None
        ),
        "largest_loser": (
            {"symbol": largest_loser["symbol"], "pnl_usd": round(largest_loser["pnl_usd"], 2)}
            if largest_loser
            else None
        ),
        "avg_winner": round(avg_winner, 2),
        "avg_loser": round(avg_loser, 2),
        "profit_factor": profit_factor,
    }


def build_notification_body(summary: dict) -> str:
    if summary["total_trades"] == 0:
        return "No closed trades today."

    pf = summary["profit_factor"]
    pf_str = str(pf)

    best = summary["largest_winner"]
    worst = summary["largest_loser"]
    best_sym = best["symbol"] if best else "n/a"
    best_amt = best["pnl_usd"] if best else 0.0
    worst_sym = worst["symbol"] if worst else "n/a"
    worst_amt = worst["pnl_usd"] if worst else 0.0

    body = (
        f"Trades: {summary['total_trades']} "
        f"({summary['wins']}W / {summary['losses']}L, {summary['win_rate_pct']}%)\n"
        f"P&L: ${summary['gross_pnl_usd']:+.2f}\n"
        f"Best: {best_sym} ${best_amt:+.2f}\n"
        f"Worst: {worst_sym} ${worst_amt:+.2f}\n"
        f"PF: {pf_str}"
    )
    return body


def _native(x):
    """Cast numpy scalar types to native Python types (Python 3.14 json
    encoder rejects numpy bool_/integer/floating)."""
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def load_open_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    try:
        data = json.loads(POSITIONS_PATH.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def load_entry_stops_by_symbol() -> dict[str, list[dict]]:
    """Entry-time stops per symbol, from the safety log's entry_opened
    events.

    This is the durable record of what each trade actually risked.
    smc_open_positions.json holds only OPEN positions, so by the time a
    daily summary runs, every pair it is scoring has already been removed
    from it -- the log is the only place the entry-time stop survives.
    smc_cycle deliberately keeps that stop distinct from
    current_stop_price so TP1's move to breakeven cannot overwrite it
    (see commit d94a548)."""
    if not SAFETY_LOG_PATH.exists():
        return {}
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    try:
        with SAFETY_LOG_PATH.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "entry_opened":
                    continue
                symbol, entry, stop = event.get("symbol"), event.get("entry_price"), event.get("stop")
                if symbol and entry is not None and stop is not None:
                    by_symbol[symbol].append({"entry_price": float(entry), "stop": float(stop)})
    except OSError:
        return {}
    return dict(by_symbol)


def get_initial_stop_for_pair(
    pair: dict, open_positions_by_symbol: dict, entry_stops_by_symbol: dict | None = None
) -> float | None:
    """Resolve the entry-time stop used to risk-size a closed pair, in
    priority order:
      1. A still-open position matching symbol + entry price.
      2. An entry_opened event in the safety log -- the usual hit, since a
         closed pair is already gone from smc_open_positions.json.
      3. None: the risk distance is genuinely unknown, so R is reported as
         unavailable rather than invented.

    There is deliberately no percentage-of-entry fallback. This used to
    return buy_price * 0.99, matching cycle.py's 1%-under-entry
    convention, but SMC's stops come off order-block structure and look
    nothing like 1%: KLAC on 2026-08-25 risked 0.88% and GS on 08-24 only
    0.08%, which that proxy would have understated by more than tenfold
    while looking perfectly plausible. A missing R beats a fabricated one.
    """
    symbol = pair["symbol"]
    buy_price = pair["buy_price"]

    for pos in open_positions_by_symbol.get(symbol, []):
        entry = pos.get("entry_price")
        # smc_cycle writes stop_price (entry-time, immutable) alongside
        # current_stop_price (moves to breakeven at TP1); risk sizing
        # wants the former. initial_stop is gap-and-go's older name.
        stop = pos.get("stop_price", pos.get("initial_stop"))
        if entry is not None and stop is not None and abs(float(entry) - buy_price) < 0.01:
            return float(stop)

    for opened in (entry_stops_by_symbol or {}).get(symbol, []):
        if abs(opened["entry_price"] - buy_price) < 0.01:
            return opened["stop"]

    return None


def compute_r_for_pairs(
    closed_pairs: list[dict], open_positions: list[dict], entry_stops_by_symbol: dict | None = None
) -> list[dict]:
    """Return closed_pairs with an added 'r_multiple' field (float or None
    if the risk distance is zero or the entry-time stop is unknown)."""
    open_positions_by_symbol: dict[str, list[dict]] = defaultdict(list)
    for pos in open_positions:
        symbol = pos.get("symbol")
        if symbol:
            open_positions_by_symbol[symbol].append(pos)

    enriched = []
    for pair in closed_pairs:
        pair = dict(pair)
        initial_stop = get_initial_stop_for_pair(pair, open_positions_by_symbol, entry_stops_by_symbol)
        risk_per_share = None if initial_stop is None else pair["buy_price"] - initial_stop
        # Catches an unknown stop and a zero risk distance alike.
        if not risk_per_share:
            pair["r_multiple"] = None
        else:
            pair["r_multiple"] = (pair["sell_price"] - pair["buy_price"]) / risk_per_share
        pair["initial_stop"] = initial_stop
        enriched.append(pair)
    return enriched


def r_bucket_label(r: float) -> str:
    if r <= -2:
        return R_BUCKET_LABELS[0]
    if r <= -1:
        return R_BUCKET_LABELS[1]
    if r <= 0:
        return R_BUCKET_LABELS[2]
    if r <= 1:
        return R_BUCKET_LABELS[3]
    if r <= 2:
        return R_BUCKET_LABELS[4]
    if r <= 3:
        return R_BUCKET_LABELS[5]
    return R_BUCKET_LABELS[6]


def build_r_histogram(pairs_with_r: list[dict]) -> dict:
    counts = {label: 0 for label in R_BUCKET_LABELS}
    for pair in pairs_with_r:
        r = pair.get("r_multiple")
        if r is None:
            continue
        counts[r_bucket_label(r)] += 1
    return counts


def get_last_price(symbol_ibkr: str) -> float | None:
    """Best-effort current/last price via yfinance, for unrealized R on
    open positions. Returns None on any failure - dashboard degrades
    gracefully rather than crashing on a network hiccup."""
    try:
        symbol_yahoo = symbol_ibkr.replace(" ", "-")
        hist = yf.Ticker(symbol_yahoo).history(period="1d", interval="1m")
        if hist is None or hist.empty:
            hist = yf.Ticker(symbol_yahoo).history(period="5d", interval="1d")
        if hist is None or hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])
        return price if price == price else None  # NaN guard
    except Exception:
        return None


def build_open_positions_view(open_positions: list[dict]) -> list[dict]:
    view = []
    for pos in open_positions:
        symbol = pos.get("symbol")
        entry = pos.get("entry_price")
        qty = pos.get("qty")
        stop = pos.get("current_stop_price", pos.get("initial_stop"))
        r_denom = pos.get("R")  # dollar risk per share, set at entry by cycle.py

        current_price = get_last_price(symbol) if symbol else None

        unrealized_r = None
        if current_price is not None and entry is not None and r_denom:
            unrealized_r = (current_price - float(entry)) / float(r_denom)

        view.append(
            {
                "symbol": symbol,
                "qty": _native(qty),
                "entry": _native(entry),
                "stop": _native(stop),
                "current_price": current_price,
                "unrealized_r": unrealized_r,
            }
        )
    return view


def get_last_cycle_info() -> dict:
    if not SAFETY_LOG_PATH.exists():
        return {"timestamp": None, "status": "no cycle data yet"}
    try:
        with SAFETY_LOG_PATH.open("r") as f:
            lines = [line for line in f if line.strip()]
        if not lines:
            return {"timestamp": None, "status": "no cycle data yet"}
        last = json.loads(lines[-1])
        return {
            "timestamp": last.get("timestamp_iso"),
            "status": last.get("event", "unknown"),
        }
    except (OSError, json.JSONDecodeError):
        return {"timestamp": None, "status": "no cycle data yet"}


def render_dashboard_html(
    summary: dict,
    pairs_with_r: list[dict],
    r_histogram: dict,
    open_positions_view: list[dict],
    last_cycle_info: dict,
    today_et_str: str,
) -> str:
    def esc(x) -> str:
        return html.escape(str(x)) if x is not None else "n/a"

    # --- Header strip ---
    last_cycle_ts = last_cycle_info.get("timestamp") or "n/a"
    last_cycle_status = last_cycle_info.get("status") or "n/a"

    # --- Card 1: today's P&L summary ---
    pnl = summary.get("gross_pnl_usd", 0.0)
    pnl_class = "text-success" if pnl > 0 else ("text-danger" if pnl < 0 else "")
    card1 = f"""
    <div class="card mb-4">
      <div class="card-header">Today's P&amp;L Summary</div>
      <div class="card-body">
        <h3 class="{pnl_class}">${pnl:+.2f}</h3>
        <p class="mb-1">Trades: {esc(summary.get('total_trades', 0))}</p>
        <p class="mb-1">Wins: {esc(summary.get('wins', 0))} &nbsp; Losses: {esc(summary.get('losses', 0))}</p>
        <p class="mb-1">Win rate: {esc(summary.get('win_rate_pct', 0))}%</p>
        <p class="mb-0">Profit factor: {esc(summary.get('profit_factor', 'n/a'))}</p>
      </div>
    </div>
    """

    # --- Card 2: R-multiple histogram (pure CSS bar chart) ---
    max_count = max(r_histogram.values()) if r_histogram.values() else 0
    max_count = max(max_count, 1)  # avoid div by zero
    bar_rows = []
    for i, label in enumerate(R_BUCKET_LABELS):
        count = r_histogram.get(label, 0)
        width_pct = (count / max_count * 100) if max_count else 0
        # Buckets 0-2 are <=0R (loss/breakeven side), 3-6 are >0R (win side).
        bar_color = "#dc3545" if i <= 2 else "#198754"
        bar_rows.append(
            f"""
            <div class="d-flex align-items-center mb-2">
              <div style="width:100px" class="text-end pe-2 small">{esc(label)}</div>
              <div class="flex-grow-1 bg-light rounded" style="height:20px;">
                <div style="width:{width_pct:.1f}%; height:20px; background-color:{bar_color};" class="rounded"></div>
              </div>
              <div style="width:30px" class="ps-2 small">{count}</div>
            </div>
            """
        )
    card2 = f"""
    <div class="card mb-4">
      <div class="card-header">R-Multiple Histogram</div>
      <div class="card-body">
        {''.join(bar_rows)}
      </div>
    </div>
    """

    # --- Card 3: open positions table ---
    if open_positions_view:
        pos_rows = []
        for p in open_positions_view:
            ur = p.get("unrealized_r")
            if ur is None:
                ur_str = "n/a"
                ur_class = ""
            else:
                ur_str = f"{ur:+.2f}R"
                ur_class = "text-success" if ur > 0 else ("text-danger" if ur < 0 else "")
            pos_rows.append(
                f"""
                <tr>
                  <td>{esc(p.get('symbol'))}</td>
                  <td>{esc(p.get('qty'))}</td>
                  <td>${esc(p.get('entry'))}</td>
                  <td>${esc(p.get('stop'))}</td>
                  <td class="{ur_class}">{ur_str}</td>
                </tr>
                """
            )
        pos_table_body = "".join(pos_rows)
    else:
        pos_table_body = '<tr><td colspan="5" class="text-center text-muted">No open positions</td></tr>'

    card3 = f"""
    <div class="card mb-4">
      <div class="card-header">Open Positions</div>
      <div class="card-body p-0">
        <table class="table table-sm mb-0">
          <thead>
            <tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Stop</th><th>Unrealized R</th></tr>
          </thead>
          <tbody>
            {pos_table_body}
          </tbody>
        </table>
      </div>
    </div>
    """

    # --- Card 4: recent closed trades (last 20, most recent first) ---
    recent = sorted(pairs_with_r, key=lambda p: p.get("sell_ts", ""), reverse=True)[:20]
    if recent:
        trade_rows = []
        for p in recent:
            r = p.get("r_multiple")
            if r is None:
                r_str = "n/a"
                r_class = ""
            else:
                r_str = f"{r:+.2f}R"
                r_class = "text-success" if r > 0 else ("text-danger" if r < 0 else "")
            pnl_usd = p.get("pnl_usd", 0.0)
            pnl_class = "text-success" if pnl_usd > 0 else ("text-danger" if pnl_usd < 0 else "")
            trade_rows.append(
                f"""
                <tr>
                  <td>{esc(p.get('symbol'))}</td>
                  <td>${p.get('buy_price', 0):.2f}</td>
                  <td>${p.get('sell_price', 0):.2f}</td>
                  <td class="{pnl_class}">${pnl_usd:+.2f}</td>
                  <td class="{r_class}">{r_str}</td>
                  <td>{p.get('hold_minutes', 0):.1f}m</td>
                </tr>
                """
            )
        trades_table_body = "".join(trade_rows)
    else:
        trades_table_body = '<tr><td colspan="6" class="text-center text-muted">No closed trades today</td></tr>'

    card4 = f"""
    <div class="card mb-4">
      <div class="card-header">Recent Closed Trades (last 20)</div>
      <div class="card-body p-0">
        <table class="table table-sm mb-0">
          <thead>
            <tr><th>Symbol</th><th>Buy</th><th>Sell</th><th>P&amp;L</th><th>R</th><th>Hold</th></tr>
          </thead>
          <tbody>
            {trades_table_body}
          </tbody>
        </table>
      </div>
    </div>
    """

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IBKR Paper-Trading Bot Dashboard</title>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.3/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
  <div class="bg-dark text-white p-3 mb-4">
    <div class="container">
      <h4 class="mb-1">Bot Status: ACTIVE</h4>
      <p class="mb-0 small">Last cycle: {esc(last_cycle_ts)} &mdash; {esc(last_cycle_status)}</p>
      <p class="mb-0 small text-secondary">Dashboard generated {esc(today_et_str)}</p>
    </div>
  </div>
  <div class="container pb-5">
    <div class="row">
      <div class="col-md-6">
        {card1}
        {card3}
      </div>
      <div class="col-md-6">
        {card2}
        {card4}
      </div>
    </div>
  </div>
</body>
</html>
"""
    return html_doc


def write_dashboard(html_doc: str) -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(html_doc, encoding="utf-8")


def main() -> None:
    today_et_str = datetime.now(ET).strftime("%Y-%m-%d")

    today_trades = load_today_trades()
    closed_pairs = pair_trades_fifo(today_trades)
    summary = aggregate(closed_pairs)

    print(json.dumps(summary))

    title = f"Daily Summary {today_et_str}"
    body = build_notification_body(summary)
    notify(title, body, "default")

    # --- HTML dashboard generation (additional side effect; failures here
    # must never take down the JSON/Telegram behavior above, which has
    # already completed by this point). ---
    try:
        open_positions = load_open_positions()
        pairs_with_r = compute_r_for_pairs(closed_pairs, open_positions, load_entry_stops_by_symbol())
        r_histogram = build_r_histogram(pairs_with_r)
        open_positions_view = build_open_positions_view(open_positions)
        last_cycle_info = get_last_cycle_info()

        html_doc = render_dashboard_html(
            summary, pairs_with_r, r_histogram, open_positions_view, last_cycle_info, today_et_str
        )
        write_dashboard(html_doc)
    except Exception as e:
        print(f"WARNING: dashboard generation failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with COMPUTE_PERF_ERRORS_LOG.open("a") as f:
            f.write(f"--- {datetime.now(timezone.utc).isoformat()} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        try:
            notify("Compute Perf CRASHED", str(exc)[:500], "high")
        except Exception:
            pass
        sys.exit(1)