@echo off
REM Master signal generator - Run all 3 strategies (RSI2, MA 30/90, Relative Strength)
REM Double-click or run from anywhere

setlocal
cd /d "%~dp0"

REM Use the project's virtual environment so trading_bot and yfinance are found
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Virtual environment not found at %PY%
    pause
    exit /b 1
)

REM Enable ANSI colors in the console
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1

echo.
echo ================================================================================
echo MORNING TRADING SIGNALS - %date% %time%
echo ================================================================================

echo.
echo ---------------------------- 1/3 RSI(2) MEAN REVERSION -------------------------
"%PY%" -m trading_bot.cli.rsi2_daily_signals
echo.
echo ---------------------------- 2/3 MA 30/90 TREND ---------------------------------
"%PY%" -m trading_bot.cli.ma_crossover_signals
echo.
echo ---------------------------- 3/3 RELATIVE STRENGTH ------------------------------
"%PY%" -m trading_bot.cli.relative_strength_signals

echo.
echo ================================================================================
echo POSITION MONITORS
echo ================================================================================
"%PY%" -m trading_bot.cli.rsi2_position_monitor
"%PY%" -m trading_bot.cli.ma_position_monitor
"%PY%" -m trading_bot.cli.rs_position_monitor

echo.
echo ================================================================================
echo SIGNALS COMPLETE
echo ================================================================================
echo.
pause
