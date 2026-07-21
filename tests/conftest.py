"""Shared pytest fixtures for the trading bot test suite."""

import os
import socket
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # tests/ -> project root

IBKR_HOST = "127.0.0.1"
IBKR_PAPER_PORT = 7497
IBKR_CLIENT_ID = 99  # separate id so tests don't clash with the live bot


@pytest.fixture(autouse=True)
def run_from_project_root():
    """All tests assume CWD = project root (rules.json, logs/, trades.csv)."""
    old = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield
    os.chdir(old)


def _tws_reachable() -> bool:
    """True if something is listening on the TWS paper port."""
    try:
        with socket.create_connection((IBKR_HOST, IBKR_PAPER_PORT), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture
def paper_client():
    """Connected IBKRClient against the local paper TWS.

    Skips (rather than aborts the whole session) when TWS/IB Gateway
    is not running, so unit tests can still run without a broker.
    """
    if not _tws_reachable():
        pytest.skip(f"TWS/IB Gateway not reachable on {IBKR_HOST}:{IBKR_PAPER_PORT}")

    # Import here so collecting/running unit tests never requires ib_async.
    from trading_bot.broker.ibkr_client import IBKRClient

    client = IBKRClient(IBKR_HOST, IBKR_PAPER_PORT, IBKR_CLIENT_ID)
    yield client
    client.disconnect()