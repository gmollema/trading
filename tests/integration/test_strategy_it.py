"""Integration test for trading_bot.strategy against a live paper TWS.

Unlike the unit tests, nothing is mocked: this exercises the real
qualifyContracts/reqMktData path and the real safety_log.jsonl write.
Because evaluate() depends on wall-clock time and current positions,
the test asserts *invariants* of the result rather than a fixed
pass/fail outcome.
"""

import json
from datetime import datetime

import pytest

from trading_bot import strategy


@pytest.mark.integration
def test_evaluate_against_paper_account(paper_client):
    symbol = "AAPL"

    log_lines_before = _read_log_lines()
    in_window_before = _in_window_now()

    result = strategy.evaluate(symbol, paper_client.ib)

    in_window_after = _in_window_now()

    # 1. Contract shape: keys, types, sane values
    assert set(result.keys()) == {"pass", "reasons", "price"}
    assert isinstance(result["pass"], bool)
    assert isinstance(result["reasons"], list) and result["reasons"]
    assert all(isinstance(r, str) for r in result["reasons"])
    assert isinstance(result["price"], float)
    assert result["price"] >= 0.0

    # 3. Exactly one new, valid JSON line in the safety log.
    #    (Checked before block 2, which may skip: the log invariant must
    #    hold on every run, boundary crossing or not.)
    log_lines_after = _read_log_lines()
    assert len(log_lines_after) == len(log_lines_before) + 1, (
        "evaluate() must append exactly one line to the safety log"
    )
    logged = json.loads(log_lines_after[-1])
    assert logged["pass"] == result["pass"]
    assert logged["reasons"] == result["reasons"]
    assert logged["price"] == result["price"]

    # 2. Reasons must be consistent with observable reality.
    #    Guard against the boundary race: if the clock crossed the window
    #    edge while evaluate() ran, we can't know which side it saw.
    if in_window_before != in_window_after:
        pytest.skip("clock crossed the entry-window boundary during the test")

    in_window = in_window_after
    in_position = strategy.has_open_position(paper_client.ib.positions(), symbol)

    if result["pass"]:
        assert in_window, "evaluate passed although the clock is outside the entry window"
        assert not in_position, "evaluate passed although a position already exists"
    elif "already in position" in result["reasons"]:
        assert in_position, "evaluate reported a position that does not exist"
    else:
        assert not in_window, (
            f"evaluate failed with {result['reasons']} but clock is inside "
            f"the window and no position exists"
        )


def _in_window_now() -> bool:
    """Independently compute window membership using the tested pure fn."""
    rules = strategy._load_rules()["time_filter"]
    now_hm = datetime.now(strategy.ET_ZONE).strftime("%H:%M")
    return strategy.within_entry_window(
        now_hm, rules["earliest_entry_et"], rules["latest_entry_et"]
    )


def _read_log_lines() -> list[str]:
    if not strategy.LOG_PATH.exists():
        return []
    return strategy.LOG_PATH.read_text(encoding="utf-8").splitlines()