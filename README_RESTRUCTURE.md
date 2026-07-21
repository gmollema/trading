# Restructured layout

    src/trading_bot/
        strategy.py        (was: /strategy.py)
        data/sp500_tickers.py  (was: /src/sp500_tickers.py)
        broker/            (unchanged)
        util/              (unchanged)
        cli/               (was: root scripts)
            bot.py cycle.py trade.py close_one.py
            morning_prefilter.py compute_perf.py rotate_logs.py
            setup_schedule.py cleanup_schedule.py
    tests/                 (unchanged layout; conftest import fixed)
    pyproject.toml         (new)

## One-time setup (from the project root, correct venv active!)

    pip install -e .

This makes `trading_bot` importable from anywhere. Without it, the cli
modules and tests cannot resolve imports (pytest still works via
pythonpath=src in pytest.ini, but the cli scripts need the install).

## Running things (from the project root)

    python -m trading_bot.cli.bot --symbol NVDA --check-only
    python -m trading_bot.cli.trade --symbol NVDA --side BUY --size 3
    python -m trading_bot.cli.cycle
    python -m trading_bot.cli.morning_prefilter --dry-run

or the short commands installed by pip:

    ht-bot --symbol NVDA --check-only
    ht-cycle

Run from the PROJECT ROOT: trades.csv, rules.json, logs/, watchlist.txt
are still resolved relative to the current working directory.

## Task Scheduler

Re-register the tasks (old ones point at the deleted root scripts):

    python -m trading_bot.cli.cleanup_schedule
    python -m trading_bot.cli.setup_schedule

The new tasks run `cmd /c cd /d <project> && python -m trading_bot.cli.<x>`,
so they now also get a correct working directory (the old version relied
on whatever CWD schtasks provided). Verify one task manually after setup.

## Not included in this zip

    .env    -> copy your existing .env into the project root yourself
    .venv   -> create/keep your venv; then run: pip install -r requirements.txt && pip install -e .
