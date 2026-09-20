@echo off
setlocal enabledelayedexpansion

cd /d C:\Users\gijs\PythonProjects\ibkr

REM Enable ANSI color codes in Windows 10+
for /f %%A in ('powershell -Command "[System.Environment]::OSVersion.Version.Major"') do set WIN_VER=%%A
if !WIN_VER! GEQ 10 (
    reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1
)

echo.
echo ======================================================================
echo RSI2 TRADING SYSTEM - Daily Check
echo ======================================================================
echo.

REM Check if positions file exists and has open positions
set HAS_POSITIONS=0
if exist rsi2_open_positions.json (
    for /f %%A in ('findstr /R /C:"\[" rsi2_open_positions.json') do (
        findstr /R /C:"{" rsi2_open_positions.json >nul
        if !errorlevel! equ 0 (
            set HAS_POSITIONS=1
        )
    )
)

if !HAS_POSITIONS! equ 1 (
    echo OPEN POSITIONS DETECTED - Showing position monitor only
    echo.
    python -m trading_bot.cli.rsi2_position_monitor
) else (
    echo NO OPEN POSITIONS - Showing entry signals only
    echo.
    python -m trading_bot.cli.rsi2_daily_signals
)

echo.
echo ======================================================================
echo Done! Act on any colored signals above
echo ======================================================================
echo.
pause
