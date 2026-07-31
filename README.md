# Financial Risk Engine

Institutional-grade market & liquidity risk toolkit in pure Python.
Built to the depth expected of a 10-year risk management practice:
portfolio analytics, three-method VaR/CVaR, Basel-style backtesting,
macro stress scenarios, and Cash-Flow-at-Risk for operating liquidity.

## What's inside

| Module | What it does |
|---|---|
| `generate` | Deterministic synthetic market data: 10 asset classes, 5 years of daily correlated returns (Cholesky, PSD-enforced, seed 42) + correlated quarterly business drivers for CFaR |
| `portfolio` | Portfolio model: weighted returns, expected return, volatility, covariance, daily P&L, allocation, summary |
| `var` | VaR and CVaR (Expected Shortfall) by **historical**, **parametric** and **Monte Carlo** (GBM, 20k sims) methods, multi-day horizon scaling |
| `backtest` | Basel-style exception counting + **Kupiec POF** likelihood-ratio test (chi2, 3.841 threshold) |
| `stress` | 6 named macro scenarios (market crash, rate hike, commodity shock, currency crisis, stagflation, COVID-style) applied to the portfolio |
| `cfar` | Cash-Flow-at-Risk: 10k quarterly EBITDA simulations from correlated drivers, annual p5/p1, breach probability |
| `report` | Executive risk report: `docs/risk_report.md` + 5 PNG charts (returns histogram, MC paths, VaR comparison, CFaR distribution, stress impacts) |

## Quick start

```bash
pip install -r requirements.txt
pip install -e .          # optional
pytest                    # 40+ tests, deterministic (seed 42)
```

## CLI

```bash
python -m risk_engine.cli simulate          # portfolio summary + allocation
python -m risk_engine.cli var               # VaR/CVaR, 3 methods, 95% 1-day
python -m risk_engine.cli backtest          # rolling-window VaR + Kupiec test
python -m risk_engine.cli stress            # all 6 scenarios
python -m risk_engine.cli cfar              # CFaR: quarterly + annual table
python -m risk_engine.cli report            # docs/risk_report.md + 5 charts
python -m risk_engine.cli all               # everything end-to-end
```

## Interpreting the numbers

- **VaR** = worst loss you would not exceed at the given confidence on a
  normal day; **CVaR** = the average loss *inside* that tail. CVaR is
  coherent; VaR is not (it can miss tail concentration).
- **Kupiec POF**: LR > 3.841 rejects the VaR model at 95% confidence —
  too few *or* too many exceptions are both red flags.
- **Stress tests** show the portfolio's vulnerability to named macro
  shocks; the worst scenario identifies the biggest tail exposure.
- **CFaR** answers "how much can annual cash flow fall short of its
  expectation in a bad-but-not-extreme year (95%)?" — the liquidity
  twin of market VaR.

## Reproducibility

Every generator is seeded (default 42). Two runs produce identical
returns, VaRs and reports — every test, chart and number in
`docs/risk_report.md` is reproducible.
