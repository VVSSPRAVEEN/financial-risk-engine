"""Portfolio model built on the generated market data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_engine.generate import DEFAULT_WEIGHTS, generate_market_returns


class Portfolio:
    def __init__(self, returns: pd.DataFrame | None = None,
                 weights: dict[str, float] | None = None,
                 value: float = 10_000_000.0):
        self.returns = returns if returns is not None else generate_market_returns()
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.value = value
        self._validate()

    def _validate(self) -> None:
        missing = set(self.weights) - set(self.returns.columns)
        if missing:
            raise KeyError(f"Weights reference unknown assets: {sorted(missing)}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        if self.value <= 0:
            raise ValueError("Portfolio value must be positive")

    # ------------------------------------------------------------ statistics
    def weights_array(self) -> np.ndarray:
        return np.array([self.weights[a] for a in self.returns.columns])

    def portfolio_returns(self) -> pd.Series:
        """Daily portfolio return series (weighted average of asset returns)."""
        return self.returns @ self.weights_array()

    def expected_return(self, annualized: bool = True) -> float:
        """Historical mean daily return; annualized if requested."""
        mean = float(self.portfolio_returns().mean())
        return mean * 252 if annualized else mean

    def volatility(self, annualized: bool = True) -> float:
        std = float(self.portfolio_returns().std(ddof=1))
        return std * np.sqrt(252) if annualized else std

    def covariance(self) -> np.ndarray:
        return self.returns.cov().to_numpy()

    # ------------------------------------------------------------------ risk
    def daily_pnl(self) -> pd.Series:
        """Daily P&L in currency."""
        return self.portfolio_returns() * self.value

    def allocation(self) -> pd.DataFrame:
        """Weights and dollar allocation per asset."""
        df = pd.DataFrame({
            "weight": [self.weights[a] for a in self.returns.columns],
            "value": [self.weights[a] * self.value for a in self.returns.columns],
        }, index=self.returns.columns)
        return df.round(2)

    def summary(self) -> dict:
        return {
            "value": self.value,
            "expected_return_ann": round(self.expected_return(), 4),
            "volatility_ann": round(self.volatility(), 4),
            "sharpe": round((self.expected_return() - 0.03) / self.volatility(), 2),
            "n_assets": len(self.weights),
            "n_days": len(self.returns),
        }
