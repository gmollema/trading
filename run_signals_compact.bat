@echo off
REM Compact signal generator - minimal output

echo [%date% %time%] Running signals...
echo.

python -m trading_bot.cli.rsi2_daily_signals
python -m trading_bot.cli.ma_crossover_signals
python -m trading_bot.cli.relative_strength_signals

echo.
echo [POSITIONS]
python -m trading_bot.cli.rsi2_position_monitor
python -m trading_bot.cli.ma_position_monitor
python -m trading_bot.cli.rs_position_monitor

echo Done.
