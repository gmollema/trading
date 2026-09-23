# Trading 212 Signal System

## Overview

Three strategies, calculated each day on the US indexes and traded in Trading 212
on the UCITS ETFs that follow them:

| Strategy | Signal calculated on | Traded as |
|---|---|---|
| **RSI(2) mean reversion** | S&P 500 (`^GSPC`) and Nasdaq Composite (`^IXIC`) | SXR8 and SXRV |
| **MA 30/90 trend** | S&P 500 and Nasdaq Composite | SXR8 and SXRV |
| **Relative Strength** (Nasdaq vs S&P) | Ratio `^IXIC` / `^GSPC` | SXRV only |

- **SXR8** = iShares Core S&P 500 UCITS ETF (Acc), EUR, Xetra
- **SXRV** = iShares Nasdaq 100 UCITS ETF (Acc), EUR, Xetra

## What to expect (read this first)

At the default sizing ($1,000 capital, $50 per SXR8 trade, $25 per SXRV trade),
the three strategies together made roughly **$10-17 per year (1.0-1.7% of
capital)** in the backtest. Trading 212 charges no commission; the only cost
assumed is the bid-ask spread, **0.05% per round trip**. Profit scales with
capital: at $10,000 with the same percentages it would be about $100-170/year.

Earlier versions of this guide said **$370/year**. That figure was wrong: the
backtest scripts converted index points to dollars at $25 per 225 S&P points and
$12 per 125 Nasdaq points. That is the profit of a position sized so the stop
loss costs $25, which is roughly **$500-$850 per S&P trade and $1,500-$2,600 per
Nasdaq trade**, 30-100x more than the $25 / $12.50 the guide said to buy. The
scripts now use `trading_bot/backtest/position_sizing.py`, which multiplies the
amount actually bought by the percentage move.

## Backtest results (checked 2026-09-23)

Tested the way the system is traded: signal on the US index close, SXR8/SXRV
bought or sold on Xetra the next day at the close, 0.05% spread per round trip
(no commission on Trading 212).

### Profit per year at the default sizing

Profit is booked in the year a trade closes; fixed $50 / $25 per trade, not compounded.

| Year | RSI2 SXR8 | RSI2 SXRV | MA SXR8 | MA SXRV | RS SXRV | **Total** | % of $1,000 | Holding SXR8 / SXRV |
|---|---|---|---|---|---|---|---|---|
| 2011 | -3.45 | -0.22 | -9.04 | -7.03 | -4.74 | **-24.47** | -2.4% | +4.8% / +5.7% |
| 2012 | -0.50 | +1.19 | +6.09 | +2.36 | +1.92 | **+11.06** | +1.1% | +11.1% / +13.0% |
| 2013 | +3.46 | +3.28 | 0.00 | 0.00 | +1.59 | **+8.33** | +0.8% | +27.4% / +32.3% |
| 2014 | +2.01 | +2.76 | +18.74 | +9.76 | +6.72 | **+39.99** | +4.0% | +30.4% / +37.3% |
| 2015 | +2.96 | +1.83 | +7.38 | +3.45 | +1.31 | **+16.93** | +1.7% | +12.5% / +21.9% |
| 2016 | +0.88 | +1.53 | -1.13 | -3.01 | +0.86 | **-0.87** | -0.1% | +14.8% / +10.0% |
| 2017 | -0.77 | +1.11 | 0.00 | 0.00 | +2.70 | **+3.03** | +0.3% | +6.7% / +15.8% |
| 2018 | -0.36 | +1.50 | +4.03 | +10.34 | -2.92 | **+12.59** | +1.3% | -1.0% / +3.0% |
| 2019 | -2.59 | -1.28 | 0.00 | +3.55 | +1.64 | **+1.32** | +0.1% | +34.5% / +42.9% |
| 2020 | -0.01 | +0.41 | -2.40 | -1.54 | +4.86 | **+1.32** | +0.1% | +6.8% / +34.5% |
| 2021 | +6.18 | +0.65 | 0.00 | 0.00 | +5.95 | **+12.78** | +1.3% | +40.7% / +39.3% |
| 2022 | -1.27 | -0.49 | +17.84 | +7.72 | -0.42 | **+23.36** | +2.3% | -14.3% / -30.1% |
| 2023 | -1.61 | +2.01 | +4.47 | +4.48 | +7.10 | **+16.46** | +1.6% | +22.5% / +51.2% |
| 2024 | +6.69 | +2.01 | 0.00 | +3.50 | +4.25 | **+16.45** | +1.6% | +32.3% / +33.5% |
| 2025 | +2.30 | -2.71 | +11.45 | +0.11 | +3.99 | **+15.14** | +1.5% | +4.7% / +7.0% |

| Average per year | RSI2 SXR8 | RSI2 SXRV | MA SXR8 | MA SXRV | RS SXRV | **Total** | % of $1,000 |
|---|---|---|---|---|---|---|---|
| 2011-2025 | +0.93 | +0.91 | +3.83 | +2.25 | +2.32 | **+10.23** | 1.0% |
| 2018-2025 | +1.17 | +0.26 | +4.42 | +3.52 | +3.06 | **+12.43** | 1.2% |
| 2021-2025 | +2.46 | +0.29 | +6.75 | +3.16 | +4.17 | **+16.84** | 1.7% |

Best year 2014 (+$40), worst year 2011 (-$24); 2 of 15 full years lost money.
About 25-35 trades per year across all strategies.

### Edge compared with holding the ETF

**Edge** = extra return compared with an average stretch of the same length in
the ETF, i.e. whether the strategy picks better-than-average days. Positive is good.

| Strategy | ETF | Edge 2010-2017 | Edge 2018-2026 | Edge 2021-2026 | In market |
|---|---|---|---|---|---|
| RSI(2) | SXR8 | -16% | -3% | +8% | 27% |
| RSI(2) | SXRV | +31% | -17% | -12% | 15% |
| MA 30/90 | SXR8 | -36% | -6% | -9% | 75% |
| MA 30/90 | SXRV | -57% | -20% | -12% | 74% |
| Relative Strength | SXRV | -47% | +25% | +35% | 57% |

For comparison, holding the ETF over 2010-2026: SXR8 +689% (13.5%/yr),
SXRV +1471% (18.3%/yr). The strategies are only in the market part of the time
and with a small part of the capital, so they carry much less risk, and earn much less.

What the tests show:
- **MA 30/90** is profitable but below simply holding the ETF in every period.
- **RSI(2) on SXRV** has not worked since 2018. Calculating it on the Nasdaq-100
  (`^NDX`) instead of the Composite made it worse, and no entry-level/SMA
  combination was positive in both periods.
- **RSI(2) on SXR8** and **Relative Strength** were positive in 2021-2026 but
  negative before 2018, so they have not been consistent over time.
- Seven other strategies (turn of month, IBS, 3 down days, 200-day trend,
  Donchian, Bollinger breakout, support bounce) showed no reliable edge on the
  ETFs, even without costs. RSI(20) dip was tested earlier and abandoned (see
  `RSI20_DIP_FINDINGS.md`).

ETF data only starts in 2010, so each period has 40-130 trades and chance plays
a large part. Past results do not predict future results.

## Daily routine

### 1. Run the signals

```
run_all_signals.bat [capital] [spx_pct] [qqq_pct]
run_all_signals.bat 1000 5 2.5
```

Defaults: 1000 capital, 5% per SXR8 trade, 2.5% per SXRV trade. The output shows,
from top to bottom:

1. **Legends** for each strategy: what each state means and what to do
2. **SIGNALS**: today's state per strategy and ETF
3. **POSITIONS**: your open trades from the CSV files, with P&L and HOLD/SELL
4. **ACTIONS**: what to do today

If there are actions, the same lines are sent as one Telegram/ntfy message.

Running before the US open (9:15-9:25 AM ET = 15:15-15:25 CET) uses yesterday's
US close. Xetra is open until 17:30 CET, so you can act the same afternoon.

### 2. Act on ACTIONS

**BUY line**, for example:
```
BUY  SXR8  $50.00  stop loss -2.9%  log S&P-500 7707.95  (RSI2 -> trades.csv)
```
1. Buy SXR8 for the amount shown.
2. Set a stop loss in Trading 212 the shown % below the price you paid.
3. Add a row to the CSV in brackets, with the **index level** from the line as
   `entry_price` (not the ETF price you paid).

**SELL line**, for example:
```
SELL SXR8  today 2026-09-23  (RSI2 bought 2026-09-01: 12+ trading days & profitable)
```
1. Sell that position today at market.
2. Remove its row (the buy date identifies it) from the CSV.

A SELL keeps showing every day until the row is removed. The script never edits
the CSV files itself.

## Exit rules

| Strategy | Sells when |
|---|---|
| RSI(2) | Held 12+ **trading days** and in profit, or stop loss hit |
| MA 30/90 | 30-day average below the 90-day, or stop loss hit |
| Relative Strength | Ratio below its 20-day average, or held 60+ trading days, or stop loss hit |

Trading days skip weekends and NYSE holidays, as the backtests count them.

## CSV formats

`entry_price` is always the **index level** on the day you bought; `stop_loss`
is in index points (225 for S&P, 125 for Nasdaq).

### trades.csv (RSI2)
```
symbol,entry_date,entry_price,entry_amount,stop_loss
^GSPC,2026-09-24,7707.95,50.00,225
```

### trades_ma.csv (MA 30/90)
```
symbol,entry_date,entry_price,entry_amount
^IXIC,2026-09-24,26943.60,25.00
```

### trades_rs.csv (Relative Strength, SXRV only)
```
entry_date,entry_price,entry_amount
2026-09-24,26943.60,25.00
```

## Known limitations

- **SXRV stop loss is 0.46%** (125 Nasdaq points). That is smaller than a normal
  day's move, so it is easily triggered. It has not been re-checked against the
  backtest.
- **SXRV follows the Nasdaq-100**, while the signals use the Nasdaq Composite.
  They move closely but not identically.
- **Xetra closes at 17:30 CET**, mid US session, so ETF prices differ from the US
  closing prices the signals use.
- **Amounts are shown in `$`**, but SXR8 and SXRV trade in EUR. The amounts are
  simply the % of whatever capital you pass in.
- **Costs**: Trading 212 has no commission, but you still pay the bid-ask spread
  (about 0.02-0.10% per round trip on SXR8/SXRV, wider right after the 9:00 CET
  open). Keep the account in EUR to avoid Trading 212's 0.15% currency fee.

## Files

- `run_all_signals.bat` - run this every morning
- `src/trading_bot/cli/signals.py` - the combined signal view (legends, signals, positions, actions)
- `src/trading_bot/cli/rsi2_daily_signals.py`, `ma_crossover_signals.py`,
  `relative_strength_signals.py` - per-strategy signal logic used by `signals.py`
- `src/trading_bot/backtest/position_sizing.py` - dollar P&L at the real trade sizes
- `test_combined_strategies.py`, `test_relative_strength.py`, `test_ma_crossover.py`,
  `test_bollinger_squeeze.py`, `test_support_resistance.py` - index backtests, 2021-07 onwards

## Troubleshooting

- **"Could not download prices"**: check the internet connection; prices come from Yahoo Finance.
- **Window closes immediately**: run the batch from an open terminal to see the error.
- **Position shows SELL every day**: remove its row from the CSV after selling.
