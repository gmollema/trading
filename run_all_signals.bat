@echo off
REM Run all 3 strategies (RSI2, MA 30/90, Relative Strength) in one compact view.
REM Usage: run_all_signals.bat [capital] [spx_pct] [qqq_pct]
REM   e.g. run_all_signals.bat 1000 5 2.5   (defaults: 1000 5 2.5)

cd /d "%~dp0"
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1
echo Fetching prices...
".venv\Scripts\python.exe" -m trading_bot.cli.signals %*
pause
