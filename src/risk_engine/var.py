"""Value-at-Risk and CVaR (Expected Shortfall) — three estimation methods.

All methods return the LOSS amount in currency (positive number) at the
given confidence level, for the given horizon (default: 1 day), plus the
loss as % of portfolio value.

- historical_var : empirical quantile of daily portfolio P&L, rescaled
                   by sqrt(horizon) for multi-day horizons.
- parametric_var : normal assumption using sample mu/sigma.
- monte_carlo_var: GBM simulation with drift and covariance, seeded.
- cvar            : average loss beyond the VaR quantile (Expected
                   Shortfall) — the coherent risk measure recommended
                   alongside VaR.
"""
from __future__ import annotations

import numpy as np

from risk_engine.portfolio import Portfolio

DEFAULT_CONFIDENCE = 0.95


class VaRCalculator:
    def __init__(self, portfolio: Portfolio, confidence: float = DEFAULT_CONFIDENCE,
                 horizon: int = 1, seed: int = 42):
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if horizon < 1:
            raise ValueError("horizon must be >= 1 day")
        self.portfolio = portfolio
        self.confidence = confidence
        self.horizon = horizon
        self.seed = seed

    # ------------------------------------------------------------------ VaR
    def historical_var(self) -> dict:
        pnl = self.portfolio.daily_pnl()
        loss_quantile = float(np.percentile(pnl.to_numpy(), (1 - self.confidence) * 100))
        var_daily = max(-loss_quantile, 0.0)
        var = var_daily * np.sqrt(self.horizon)
        return self._result("historical", var)

    def parametric_var(self) -> dict:
        mu = self.portfolio.expected_return(annualized=False)
        sigma = self.portfolio.volatility(annualized=False)
        z = float(abs(np.percentile(np.random.default_rng(self.seed).normal(size=100_000),
                                    (1 - self.confidence) * 100)))
        # 1-day loss: -(mu*T - z*sigma*sqrt(T)) * V
        var = (z * sigma * np.sqrt(self.horizon) - mu * self.horizon) * self.portfolio.value
        return self._result("parametric", max(var, 0.0))

    def monte_carlo_var(self, n_sims: int = 20_000) -> dict:
        rng = np.random.default_rng(self.seed)
        weights = self.portfolio.weights_array()
        mu = self.portfolio.expected_return(annualized=False)
        sigma = self.portfolio.volatility(annualized=False)
        z = rng.standard_normal(n_sims)
        horizon = self.horizon
        # GBM terminal log-return
        log_r = (mu - 0.5 * sigma ** 2) * horizon + sigma * np.sqrt(horizon) * z
        losses = -self.portfolio.value * (np.exp(log_r) - 1.0)
        var = float(np.percentile(losses, self.confidence * 100))
        return self._result("monte_carlo", max(var, 0.0))

    # ------------------------------------------------------------------ CVaR
    def cvar(self, method: str = "historical", n_sims: int = 20_000) -> dict:
        """Expected shortfall: mean loss beyond the VaR threshold."""
        if method == "historical":
            pnl = self.portfolio.daily_pnl().to_numpy()
            q = float(np.percentile(pnl, (1 - self.confidence) * 100))
            tail = pnl[pnl <= q]
            es_daily = max(float(-tail.mean()), 0.0) if len(tail) else 0.0
            es = es_daily * np.sqrt(self.horizon)
        elif method == "parametric":
            var = self.parametric_var()["var"]
            sigma = self.portfolio.volatility(annualized=False)
            z = float(abs(np.percentile(np.random.default_rng(self.seed).normal(size=100_000),
                                        (1 - self.confidence) * 100)))
            phi = float(np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi))
            es = var + self.portfolio.value * sigma * np.sqrt(self.horizon) * (phi / (1 - self.confidence))
        elif method == "monte_carlo":
            rng = np.random.default_rng(self.seed + 1)
            weights = self.portfolio.weights_array()
            mu = self.portfolio.expected_return(annualized=False)
            sigma = self.portfolio.volatility(annualized=False)
            z = rng.standard_normal(n_sims)
            log_r = (mu - 0.5 * sigma ** 2) * self.horizon + sigma * np.sqrt(self.horizon) * z
            losses = -self.portfolio.value * (np.exp(log_r) - 1.0)
            var = float(np.percentile(losses, self.confidence * 100))
            es = float(losses[losses >= var].mean())
        else:
            raise ValueError(f"method must be historical|parametric|monte_carlo, got {method!r}")
        return self._result(f"cvar_{method}", max(es, 0.0))

    # ------------------------------------------------------------------ util
    def _result(self, method: str, var: float) -> dict:
        return {
            "method": method,
            "confidence": self.confidence,
            "horizon_days": self.horizon,
            "var": round(float(var), 2),
            "var_pct": round(float(var) / self.portfolio.value * 100, 4),
            "portfolio_value": self.portfolio.value,
        }

    def all_methods(self, n_sims: int = 20_000) -> list[dict]:
        """VaR from all three methods + CVaR for comparison tables."""
        results = [self.historical_var(), self.parametric_var(),
                   self.monte_carlo_var(n_sims)]
        for r in results:
            cvar = self.cvar(method=r["method"].replace("monte_carlo", "monte_carlo"),
                             n_sims=n_sims)
            r["cvar"] = cvar["var"]
            r["cvar_pct"] = cvar["var_pct"]
        return results
