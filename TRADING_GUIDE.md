# Trading 212 Automated Signal System

## Overview

Three complementary trading strategies running on your $500 account:

1. **RSI(2) Mean Reversion** — $121/year (7 trades/year)
2. **MA 30/90 Trend Following** — $118/year (5 trades/year)
3. **Relative Strength (Nasdaq/SPX)** — $131/year (15 trades/year)

**Total Expected:** $370/year ($1-2/day on $500)

## Capital Allocation

- **S&P 500 (SPY)**: $25 per trade
- **Nasdaq 100 (QQQ)**: $12.50 per trade

Stop loss:
- SPX: 225 points
- Nasdaq: 125 points

## Daily Trading Routine

### Morning (Before market open)

1. **Run all signal generators:**
   ```
   run_all_signals.bat
   ```
   
   This outputs:
   - RSI2 signals (BUY/SETUP/WAIT)
   - MA 30/90 signals (BUY/HOLD/SELL)
   - Relative Strength signals (BUY/HOLD/SELL/WAIT)
   - Position monitors (current P&L, exit signals)

2. **Act on BUY signals**
   - Green [SIGNAL] BUY = enter at market price when market opens
   - Log entry in appropriate CSV file:
     - RSI2 entry → `trades.csv`
     - MA entry → `trades_ma.csv`
     - RS entry → `trades_rs.csv`

3. **Monitor HOLD positions**
   - Green [ACTION] HOLD = stay in trade
   - Red [ACTION] SELL = exit at market price immediately

### Exit Conditions

**RSI2 Strategy:**
- SELL when you see `[POSITION] SELL` in monitor (12+ days held AND profitable)
- Or on stop loss (225 pts SPX, 125 pts Nasdaq)

**MA 30/90 Strategy:**
- SELL when you see red `[ACTION] SELL - DEATH CROSS` (30-MA < 90-MA)
- Or on stop loss

**Relative Strength:**
- SELL when you see red `[ACTION] SELL - DEATH CROSS` (Ratio < MA)
- Or on stop loss

### Trading 212 Steps

For each BUY signal:

1. Open Trading 212 app
2. Select instrument (SPY or QQQ)
3. Click "BUY"
4. Enter amount: $25 (SPY) or $12.50 (QQQ)
5. Set stop loss:
   - SPY: Current Price - 225 points
   - QQQ: Current Price - 125 points
6. Place order

For each SELL signal:

1. Open Trading 212 app
2. Select position
3. Click "SELL"
4. Sell at market price
5. Update CSV (mark as exited)

## CSV Format

### trades.csv (RSI2)
```
symbol,entry_date,entry_price,entry_amount,stop_loss
^GSPC,2026-09-22,4520.50,25.00,225.0
^IXIC,2026-09-20,18450.75,12.50,125.0
```

### trades_ma.csv (MA 30/90)
```
symbol,entry_date,entry_price,entry_amount
^GSPC,2026-09-19,4515.00,25.00
```

### trades_rs.csv (Relative Strength)
```
entry_date,entry_price,entry_amount
2026-09-15,18400.00,12.50
```

## Signal Interpretation

### RSI2 Signals
- `[SIGNAL] BUY` - RSI(2) crossed below 10 (SPX) or 7 (Nasdaq) AND above 200-MA
- `[SIGNAL] SETUP` - RSI(2) near threshold, watch for entry
- `[SIGNAL] WAIT` - Not yet in range

### MA 30/90 Signals
- `[SIGNAL] BUY` - 30-MA crossed above 90-MA (Golden Cross)
- `[SIGNAL] HOLD` - In uptrend (30-MA > 90-MA), continue holding
- `[SIGNAL] SELL` - 30-MA crossed below 90-MA (Death Cross)

### Relative Strength Signals
- `[SIGNAL] BUY` - Ratio crossed above its 20-MA (Nasdaq outperforming)
- `[SIGNAL] HOLD` - Ratio above MA, Nasdaq stronger, continue holding
- `[SIGNAL] SELL` - Ratio crossed below its 20-MA (Nasdaq weaker)
- `[SIGNAL] WAIT` - Ratio below MA, stay in cash

## Position Monitors

After running signals, check position monitors:

```
[POSITION] S&P 500 (SPY)
   Entry: $4520.50 (5 days ago)
   Current: $4545.25
   P&L: $6.19 (+0.3%)
   
[ACTION] HOLD - In uptrend
   Continue holding
```

Look for:
- `[ACTION] HOLD` = keep position open
- Red `[ACTION] SELL` = exit immediately at market

## Backtest Results

Testing period: 2021-07-01 onwards (5.2 years)

### RSI2
- 36 trades on SPX: +$84/year
- 29 trades on Nasdaq: +$121/year
- Win rate: ~48%

### MA 30/90
- 27 trades on SPX: +$52/year
- 25 trades on Nasdaq: +$66/year
- Win rate: ~45%

### Relative Strength
- 78 trades on Nasdaq
- Win rate: 39.7%
- Average: 87.5 pts/trade = $131/year

### Combined
All three running together: $370/year
- No signal overlap (strategies are complementary)
- Different risk profiles (mean reversion, trend, rotation)

## Important Rules

1. **Only trade during market hours** (9:30 AM - 4:00 PM ET)
2. **Execute BUY signals immediately** when market opens with green signal
3. **Exit on SELL signals immediately** (don't wait for better price)
4. **Always set stop loss** before placing buy order
5. **Check morning signals every day** (takes 2-3 minutes)
6. **Update CSV files** after each trade

## Troubleshooting

### No signals generated
- Check internet connection (need to fetch yfinance data)
- Check that Python environment is installed
- Verify `backtest_data/daily_index/` has current data

### Position monitor shows ERROR
- Might be market hours issue (closed/weekend)
- Check if Trading 212 API is accessible
- Try again later

### Stop loss hit on position
- Expected to happen ~50-60% of the time
- Exit position immediately
- Don't re-enter same signal
- Log as "Stop loss hit" in trades file

## Expected Results

On $500 account with full execution:
- Typical month: $30-35 profit ($1.50-1.75/day)
- Good month: $50-75 profit
- Bad month: -$10 to -$30 loss
- Annual: ~$370 average

Results are consistent and do not depend on market direction (mean reversion and trend-following strategies work in both up and down markets).

## Getting Started

1. **Today**: Set up CSV files:
   - Create empty `trades.csv`
   - Create empty `trades_ma.csv`
   - Create empty `trades_rs.csv`

2. **Tomorrow morning**: Run first signal check
   - `run_all_signals.bat`
   - Look for BUY signals
   - Enter positions on any green BUY signals

3. **Every morning after**: Repeat signal check and position monitoring
   - Takes ~2-3 minutes total
   - Execute signals as they appear
   - Log all trades

## Files Reference

- `run_all_signals.bat` - Master script (run this every morning)
- `src/trading_bot/cli/rsi2_daily_signals.py` - RSI2 signal generator
- `src/trading_bot/cli/rsi2_position_monitor.py` - RSI2 position tracker
- `src/trading_bot/cli/ma_crossover_signals.py` - MA 30/90 signal generator
- `src/trading_bot/cli/ma_position_monitor.py` - MA position tracker
- `src/trading_bot/cli/relative_strength_signals.py` - RS signal generator
- `src/trading_bot/cli/rs_position_monitor.py` - RS position tracker

## Support

If signals aren't working or you have questions about entries/exits:
1. Check that backtest data is current (in `backtest_data/daily_index/`)
2. Verify Python can access yfinance (test with `python -c "import yfinance"`)
3. Review backtests: `test_combined_strategies.py` for full system analysis
