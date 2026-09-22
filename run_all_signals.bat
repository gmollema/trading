@echo off
REM Master signal generator - Run all 3 strategies

echo.
echo ================================================================================
echo MORNING TRADING SIGNALS - %date% %time%
echo ================================================================================
echo.

python -m trading_bot.cli.rsi2_daily_signals
python -m trading_bot.cli.ma_crossover_signals
python -m trading_bot.cli.relative_strength_signals

echo.
echo ================================================================================
echo POSITION MONITORS
echo ================================================================================
echo.

python -m trading_bot.cli.rsi2_position_monitor
python -m trading_bot.cli.ma_position_monitor
python -m trading_bot.cli.rs_position_monitor

echo.
echo ================================================================================
echo SIGNALS COMPLETE
echo ================================================================================
echo.
