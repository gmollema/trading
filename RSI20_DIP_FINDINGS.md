# RSI(20) Dip Strategy - Findings & Abandonment

**Status:** ⚠️ **DEPRECATED** - Do not use. Code retained for historical reference only.

## Summary

The RSI(20) @ 60/65 dip strategy, reconstructed from a TradingView webinar indicator, has been thoroughly backtested and **does not beat buy-and-hold on any instrument tested**. It was abandoned in favor of the RSI(2) mean reversion strategy, which demonstrates actual edge.

## Backtest Results (2008-2026, 18.7 years)

### S&P 500 (^GSPC)
```
Strategy:     +227% return, 6.55% CAGR, 16.6% max DD, 83 trades, 74.7% win
Buy-and-hold: +428% return, 9.33% CAGR, 53.3% max DD
Result:       ❌ LOSES by 2.78% CAGR
```

### ES Futures (E-mini S&P 500)
```
Strategy:     +208% return, 6.21% CAGR, 15.4% max DD, 83 trades, 73.5% win
Buy-and-hold: +425% return, 9.29% CAGR, 53.7% max DD
Result:       ❌ LOSES by 3.08% CAGR
```

### Nasdaq 100 Futures (NQ)
```
Strategy:     +502% return, 10.10% CAGR, 24.4% max DD, 91 trades, 76.9% win
Buy-and-hold: +1300% return, 15.19% CAGR, 49.9% max DD
Result:       ❌ LOSES by 5.09% CAGR
```

## Why It Failed

### What Looked Good
- ✅ High win rates: 73-76%
- ✅ Lower maximum drawdown: 15-24% vs 50%+ for buy-and-hold
- ✅ Better return/drawdown ratio: 0.39-0.41 vs 0.17-0.30
- ✅ Consistent entry/exit logic with clear mean reversion signals

### The Fatal Flaw
- ❌ **No absolute return edge** - Lower CAGR than simply holding the index
- ❌ **Leverage-dependent** - Only works with leverage; cannot beat buy-and-hold on its own
- ❌ **Misleading metrics** - High win rate and low drawdown created false confidence
- ❌ **Parameter optimization overfitted** - RSI(15) @ 60/65 seemed better but still underperformed

## Development History

1. **Webinar Strategy (Original)**: RSI(20) @ 60/65 entry/exit
2. **User Reconstruction**: Initially had inverted logic (momentum instead of mean reversion) = +34% vs reference's +158%
3. **Logic Fix**: Corrected inversion = +153% (closer to reference)
4. **Parameter Sweep**: Tested RSI(5-30), found RSI(15) @ 60/65 seemed optimal
5. **Full Validation**: Backtested across multiple instruments and time windows
6. **Result**: All versions lose to buy-and-hold
7. **Decision**: Abandoned in favor of RSI(2), which demonstrates real edge

## Key Insight: Return vs Risk-Adjusted Returns

The strategy excels at **risk-adjusted returns** (profit per unit of drawdown) but fails at **absolute returns**. This is only useful if leveraged, which introduces additional risks and requirements:
- Margin requirements
- Liquidation risk
- Leverage costs
- Regulatory constraints

## Why RSI(2) Works Better

- **Entry Selectivity**: RSI < 10 (S&P 500) is far more restrictive than RSI < 60
- **Trade Quality**: 67% win rate with genuine profitability edge
- **Absolute Return Edge**: Beats buy-and-hold without leverage required
- **Validated**: Backtest shows $1.1M profit on $10M capital (11% CAGR)

## Conclusion

The RSI(20) dip strategy, while intellectually sound as a mean reversion system with good risk metrics, **does not provide a tradeable edge on its own**. It serves as a historical example of how high win rates and low drawdown can be misleading when absolute returns lag the benchmark.

The lesson: Always compare against buy-and-hold. Relative returns without absolute edge are not sufficient.

---

**Code Status**: Files retained for historical validation. Marked as `@deprecated` in docstrings.
- `src/trading_bot/backtest/rsi20_dip_signals.py`
- `src/trading_bot/cli/rsi20_dip_backtest.py`
- `src/trading_bot/cli/rsi20_dip_cycle.py`
- `src/trading_bot/rsi20_dip_live.py`

**Active Strategy**: Use `rsi2_*` files instead.
