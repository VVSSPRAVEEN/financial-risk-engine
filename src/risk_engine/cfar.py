"""Cash-Flow-at-Risk (CFaR) for operating liquidity planning.

Simulates quarterly EBITDA by drawing correlated business drivers
(revenue growth, COGS ratio, opex ratio, FX impact) and reports the
distribution of quarterly and annual cash flow:

    cash_flow_q = revenue_q * (1 - cogs_ratio - opex_ratio) * (1 + fx_impact)

CFaR is the shortfall at the chosen confidence level — the amount by
which cash flow can fall below the mean in a bad-but-not-extreme year.
Also reports probability of negative EBITDA (liquidity breach).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_engine.generate import generate_operating_drivers


class CashFlowAtRisk:
    def __init__(self, annual_revenue: float = 500_000_000.0,
                 seed: int = 42, n_sims: int = 10_000,
                 confidence: float = 0.95):
        self.annual_revenue = annual_revenue
        self.seed = seed
        self.n_sims = n_sims
        self.confidence = confidence
        self.drivers = generate_operating_drivers(seed=seed)

    # ------------------------------------------------------------- simulate
    def simulate(self) -> np.ndarray:
        """Simulated quarterly cash flows: shape (n_sims, 12)."""
        rng = np.random.default_rng(self.seed + 11)
        n, quarters = self.n_sims, len(self.drivers)

        mu = np.asarray(self.drivers.mean(), dtype=float)
        cov = np.asarray(self.drivers.cov(), dtype=float)
        chol = np.linalg.cholesky(cov)

        quarterly_revenue = self.annual_revenue / 4.0
        draws = mu + rng.standard_normal((n, quarters, 4)) @ chol.T

        growth = draws[:, :, 0]
        cogs = draws[:, :, 1]
        opex = draws[:, :, 2]
        fx = draws[:, :, 3]

        # Revenue compounds quarter over quarter
        revenue = quarterly_revenue * np.cumprod(1 + growth, axis=1)
        cash = revenue * (1 - cogs - opex) * (1 + fx)
        return cash

    # ------------------------------------------------------------- summary
    def quarterly_cfar(self) -> pd.DataFrame:
        cash = self.simulate()
        dates = [d.date() for d in self.drivers.index]
        p5 = np.percentile(cash, 5, axis=0)
        p50 = np.percentile(cash, 50, axis=0)
        p95 = np.percentile(cash, 95, axis=0)
        mean = cash.mean(axis=0)
        shortfall = mean - p5
        return pd.DataFrame({
            "quarter": dates, "mean": mean.round(2), "p5": p5.round(2),
            "p50": p50.round(2), "p95": p95.round(2),
            "cfar_95": shortfall.round(2),
        })

    def annual_cfar(self) -> dict:
        cash = self.simulate()
        annual = cash.sum(axis=1)
        mean = float(annual.mean())
        p5 = float(np.percentile(annual, 5))
        p1 = float(np.percentile(annual, 1))
        shortfall = mean - p5
        return {
            "annual_mean": round(mean, 2),
            "annual_p5": round(p5, 2),
            "annual_p1": round(p1, 2),
            "cfar_95": round(shortfall, 2),
            "cfar_95_pct_of_revenue": round(shortfall / self.annual_revenue * 100, 2),
            "prob_negative_ebitda": round(float((annual < 0).mean()), 4),
            "worst_case": round(float(annual.min()), 2),
            "n_sims": self.n_sims,
        }

    def breach_probability(self, threshold: float = 0.0) -> float:
        """Probability that annual cash flow falls below `threshold`."""
        annual = self.simulate().sum(axis=1)
        return round(float((annual < threshold).mean()), 4)
