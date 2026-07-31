# Risk Report — Financial Risk Engine

## Portfolio

- Value: $10.0M | Expected return (ann): 2.1% | Volatility (ann): 8.4% | Sharpe (rf=3%): -0.1
- Assets: 10 classes | Daily observations: 1260

## Value-at-Risk (1-day, 95%)

| method | var | var_pct | cvar | cvar_pct |
|---|---|---|---|---|
| historical | 87,889.94 | 0.88 | 110,471.08 | 1.10 |
| parametric | 87,056.43 | 0.87 | 193,985.57 | 1.94 |
| monte_carlo | 86,052.37 | 0.86 | 108,404.81 | 1.08 |

## Backtest (historical VaR vs realized P&L)

- Exceptions: 63 of 1260 days (expected 63.0; rate 5.0000%)
- Kupiec POF LR = -0.00 -> model not rejected at 95%

## Stress scenarios

| scenario | total_pnl | total_pct |
|---|---|---|
| market_crash | -1,421,000.00 | -14.21 |
| covid_style | -1,084,000.00 | -10.84 |
| stagflation | -814,000.00 | -8.14 |
| rate_hike | -646,000.00 | -6.46 |
| currency_crisis | -535,000.00 | -5.35 |
| commodity_shock | -312,000.00 | -3.12 |

## Cash-Flow-at-Risk (annual)

- Mean annual EBITDA: $412.8M | p5: $342.3M | p1: $315.6M
- CFaR at 95%: $70.6M (14.1% of revenue) | P(negative EBITDA): 0.00%

## Charts

![Returns histogram](portfolio_returns_hist.png)
![Monte Carlo paths](monte_carlo_paths.png)
![VaR comparison](var_comparison.png)
![CFaR distribution](cfar_distribution.png)
![Stress impacts](stress_impacts.png)
