"""Single-symbol dev-tool strategy gate.

IMPORTANT: This module intentionally does NOT implement the D1-D3 daily
filters or I1-I3 intraday filters from rules.json. Those use yfinance and
the prefilter built in a later chapter. This file only checks:
  (a) whether we're already in a position for the symbol, and
  (b) whether we're inside the configured entry time window.

Structure
---------
The module is split into two layers so the decision logic can be unit
tested without a broker connection or filesystem:

* Pure functions (no I/O, no side effects):
    - ``has_open_position``   -- position check against a list of positions
    - ``within_entry_window`` -- time-of-day gate on "HH:MM" strings
    - ``clean_price``         -- pick a usable price, handling NaN/None/<=0

* I/O layer (filesystem + broker):
    - ``_load_rules``   -- read rules.json from the current working directory
    - ``_log_result``   -- append the decision to logs/safety_log.jsonl
    - ``_fetch_price``  -- qualify the contract and sample a market price
    - ``evaluate``      -- thin orchestrator wiring the above together

Only ``evaluate`` is intended as public API for callers (bot.py); the pure
functions are also public so tests and future filter modules can reuse
them.

Note on paths: RULES_PATH and LOG_DIR are resolved relative to the current
working directory. All entry points are expected to run from the project
root (see README_RESTRUCTURE.md).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_async import Stock

RULES_PATH = Path("rules.json")
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "safety_log.jsonl"

ET_ZONE = ZoneInfo("America/New_York")
MKT_DATA_WAIT_SECS = 2


# ---------------------------------------------------------------------------
# Pure decision functions (no I/O -- unit test these directly)
# ---------------------------------------------------------------------------

def has_open_position(positions, symbol: str) -> bool:
    """Return True if ``positions`` contains an open long position in ``symbol``.

    Args:
        positions: Iterable of position objects as returned by
            ``ib_async.IB.positions()``. Each item must expose
            ``.contract.symbol`` (str) and ``.position`` (numeric; positive
            for long, negative for short).
        symbol: Ticker symbol to look for, e.g. ``"AAPL"``.

    Returns:
        True if any position matches the symbol with a size > 0
        (i.e. an existing long). Short positions (size < 0) and flat
        entries (size == 0) do not count as "in position" for this
        long-only strategy.
    """
    return any(
        p.contract.symbol == symbol and p.position > 0
        for p in positions
    )


def within_entry_window(now_hm: str, earliest: str, latest: str) -> bool:
    """Return True if ``now_hm`` falls inside the entry window (inclusive).

    All three arguments must be zero-padded 24-hour ``"HH:MM"`` strings
    (e.g. ``"09:35"``, not ``"9:35"``). With that format, lexicographic
    string comparison matches chronological order, which is what this
    function relies on. A ``ValueError`` is raised for malformed input
    rather than silently comparing wrong.

    Args:
        now_hm: Current time as ``"HH:MM"``.
        earliest: Window start as ``"HH:MM"`` (inclusive).
        latest: Window end as ``"HH:MM"`` (inclusive).

    Returns:
        True if ``earliest <= now_hm <= latest``.

    Raises:
        ValueError: If any argument is not a valid zero-padded
            ``"HH:MM"`` string.
    """
    for label, value in (("now_hm", now_hm), ("earliest", earliest), ("latest", latest)):
        if not _is_valid_hm(value):
            raise ValueError(
                f"{label} must be a zero-padded 24h 'HH:MM' string, got {value!r}"
            )
    return earliest <= now_hm <= latest


def clean_price(market_price, last) -> float:
    """Choose a usable positive price from a market price with fallback.

    ib_async's ``ticker.marketPrice()`` can return ``nan`` (no data yet)
    and ``ticker.last`` can be ``nan``, ``None``, ``0`` or negative in
    degenerate cases. This helper encapsulates the cleanup:

    1. If ``market_price`` is a real number > 0, use it.
    2. Otherwise, if ``last`` is a real number > 0, use that.
    3. Otherwise return ``0.0`` (caller treats 0.0 as "no price").

    Args:
        market_price: Primary price candidate (float or NaN/None).
        last: Fallback price candidate (float or NaN/None).

    Returns:
        A float > 0 when a usable price exists, else ``0.0``.
    """
    if _is_positive_number(market_price):
        return float(market_price)
    if _is_positive_number(last):
        return float(last)
    return 0.0


def _is_positive_number(value) -> bool:
    """True if ``value`` is a real (non-NaN, non-None) number greater than 0."""
    if value is None:
        return False
    try:
        return not math.isnan(value) and value > 0
    except TypeError:
        return False


def _is_valid_hm(value) -> bool:
    """True if ``value`` is a zero-padded 24h ``"HH:MM"`` string."""
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        return False
    hh, mm = value[:2], value[3:]
    if not (hh.isdigit() and mm.isdigit()):
        return False
    return 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59


# ---------------------------------------------------------------------------
# I/O layer (filesystem + broker)
# ---------------------------------------------------------------------------

def _load_rules() -> dict:
    """Read and parse rules.json from the current working directory."""
    with RULES_PATH.open("r") as f:
        return json.load(f)


def _log_result(result: dict) -> None:
    """Append one decision record to logs/safety_log.jsonl (JSON Lines).

    Every value is cast explicitly: Python 3.14's json encoder rejects
    numpy bools, and prices may arrive as numpy floats.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_result = {
        "pass": bool(result["pass"]),
        "reasons": list(result["reasons"]),
        "price": float(result["price"]),
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(safe_result) + "\n")


def _fetch_price(symbol: str, ib) -> float:
    """Sample a current market price for ``symbol`` via the broker.

    Qualifies the contract, requests streaming market data, waits
    ``MKT_DATA_WAIT_SECS`` for a tick, cleans the price via
    ``clean_price`` and always cancels the market data subscription.

    Args:
        symbol: Ticker symbol, e.g. ``"AAPL"``.
        ib: A connected ``ib_async.IB`` instance.

    Returns:
        A float > 0, or ``0.0`` when no usable price arrived in time.
    """
    stock = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(stock)
    contract = qualified[0] if qualified else stock

    ticker = ib.reqMktData(contract, "", False, False)
    try:
        ib.sleep(MKT_DATA_WAIT_SECS)
        return clean_price(ticker.marketPrice(), ticker.last)
    finally:
        ib.cancelMktData(contract)


def _fail(reason: str) -> dict:
    """Build, log and return a failing evaluation result."""
    result = {"pass": False, "reasons": [reason], "price": 0.0}
    _log_result(result)
    return result


def _pass(price: float, reasons: list[str]) -> dict:
    """Build, log and return a passing evaluation result."""
    result = {"pass": True, "reasons": list(reasons), "price": float(price)}
    _log_result(result)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(symbol: str, ib) -> dict:
    """Evaluate whether ``symbol`` is eligible for entry right now.

    Checks run in order, returning early on first failure:
      a) already in a long position          -> fail
      b) outside the configured entry window -> fail
      c) otherwise                           -> pass, with current market price

    Args:
        symbol: Ticker symbol, e.g. ``"AAPL"``.
        ib: A connected ``ib_async.IB`` instance.

    Returns:
        A dict with keys:
          - ``"pass"``    (bool):  True if the symbol may be entered now.
          - ``"reasons"`` (list[str]): Human-readable explanation(s).
          - ``"price"``   (float): Current market price on pass, else 0.0.
            Note: 0.0 can also occur on a *pass* when no usable price
            arrived within MKT_DATA_WAIT_SECS -- callers must not treat a
            pass as an implicit price guarantee.

    Side effects:
        Appends the decision to logs/safety_log.jsonl and briefly
        subscribes to market data for the symbol.
    """
    # a) Already in position
    if has_open_position(ib.positions(), symbol):
        return _fail("already in position")

    # b) Time window
    time_filter = _load_rules()["time_filter"]
    earliest = time_filter["earliest_entry_et"]
    latest = time_filter["latest_entry_et"]
    now_hm = datetime.now(ET_ZONE).strftime("%H:%M")

    if not within_entry_window(now_hm, earliest, latest):
        return _fail(f"outside entry window {earliest}-{latest}")

    # c) Time gate ok, no existing position -> get current market price
    price = _fetch_price(symbol, ib)
    return _pass(price, ["time gate ok", "no existing position"])
