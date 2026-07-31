"""Deterministic synthetic market and operating data generation.

Market data: 10 asset classes, daily returns for 5 years (1,258 trading
days), correlated via a Cholesky-decomposed correlation matrix. Asset
classes mirror a diversified institutional portfolio:

    equities_us, equities_eu, equities_em, bonds_gov, bonds_corp,
    real_estate, commodities, gold, crypto, cash

Operating data: correlated quarterly business drivers for CFaR
(revenue growth, COGS ratio, opex ratio, FX impact), 12 quarters.

Everything is deterministic (numpy Generator, seed=42) so every run and
test reproduces identical figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
TRADING_DAYS = 252
HORIZON_DAYS = 5 * TRADING_DAYS

ASSET_CLASSES = [
    "equities_us", "equities_eu", "equities_em", "bonds_gov", "bonds_corp",
    "real_estate", "commodities", "gold", "crypto", "cash",
]
# (annualized expected return, annualized volatility)
ASSET_PARAMS = {
    "equities_us":  (0.095, 0.155),
    "equities_eu":  (0.080, 0.170),
    "equities_em":  (0.100, 0.225),
    "bonds_gov":    (0.035, 0.060),
    "bonds_corp":   (0.048, 0.085),
    "real_estate":  (0.065, 0.110),
    "commodities":  (0.050, 0.180),
    "gold":         (0.045, 0.150),
    "crypto":       (0.180, 0.550),
    "cash":         (0.030, 0.005),
}

# Correlation skeleton (order matches ASSET_CLASSES): base off-diagonal,
# with stronger links inside {equities}, {bonds} and {equities <-> crypto}.
_BASE_CORR = 0.18
_STRONG = 0.55
_MEDIUM = 0.30
CORRELATION = np.full((10, 10), _BASE_CORR)
np.fill_diagonal(CORRELATION, 1.0)
for i in (0, 1, 2):      # equities cluster
    for j in (0, 1, 2):
        if i != j:
            CORRELATION[i, j] = _STRONG
for i in (3, 4):         # bonds cluster
    for j in (3, 4):
        if i != j:
            CORRELATION[i, j] = _STRONG
CORRELATION[2, 8] = CORRELATION[8, 2] = _MEDIUM   # EM <-> crypto
CORRELATION[0, 8] = CORRELATION[8, 0] = _MEDIUM   # US eq <-> crypto
CORRELATION[6, 7] = CORRELATION[7, 6] = _MEDIUM   # commodities <-> gold
CORRELATION[5, 6] = CORRELATION[6, 5] = _MEDIUM   # RE <-> commodities

# Default portfolio: 60/40-style with alternatives sleeve
DEFAULT_WEIGHTS = {
    "equities_us": 0.24, "equities_eu": 0.12, "equities_em": 0.08,
    "bonds_gov": 0.22, "bonds_corp": 0.12, "real_estate": 0.07,
    "commodities": 0.04, "gold": 0.04, "crypto": 0.03, "cash": 0.04,
}


def _ensure_psd(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize and nudge eigenvalues to guarantee a PSD matrix."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-8, None)
    return (eigvecs * eigvals) @ eigvecs.T


def generate_market_returns(seed: int = SEED) -> pd.DataFrame:
    """Daily correlated returns per asset class (index = trading date)."""
    rng = np.random.default_rng(seed)
    mu = np.array([ASSET_PARAMS[a][0] / TRADING_DAYS for a in ASSET_CLASSES])
    sigma = np.array([ASSET_PARAMS[a][1] / np.sqrt(TRADING_DAYS) for a in ASSET_CLASSES])
    corr = _ensure_psd(CORRELATION)
    cov = np.diag(sigma) @ corr @ np.diag(sigma)
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((HORIZON_DAYS, len(ASSET_CLASSES)))
    returns = mu + z @ chol.T
    dates = pd.bdate_range("2021-01-04", periods=HORIZON_DAYS)
    return pd.DataFrame(returns, index=dates, columns=ASSET_CLASSES)


def generate_operating_drivers(seed: int = SEED) -> pd.DataFrame:
    """Quarterly correlated business drivers (12 quarters, 2023Q1..2025Q4).

    Columns: revenue_growth, cogs_ratio, opex_ratio, fx_impact.
    Used as the cash-flow engine for CFaR simulation.
    """
    rng = np.random.default_rng(seed + 7)
    n = 12
    means = np.array([0.02, 0.48, 0.30, -0.005])   # per-quarter
    sds = np.array([0.045, 0.030, 0.025, 0.015])
    corr = np.array([
        [1.00, -0.35, -0.20, 0.15],
        [-0.35, 1.00, 0.30, -0.10],
        [-0.20, 0.30, 1.00, -0.05],
        [0.15, -0.10, -0.05, 1.00],
    ])
    cov = np.diag(sds) @ _ensure_psd(corr) @ np.diag(sds)
    chol = np.linalg.cholesky(cov)
    draws = means + rng.standard_normal((n, 4)) @ chol.T
    df = pd.DataFrame(draws, columns=["revenue_growth", "cogs_ratio",
                                      "opex_ratio", "fx_impact"])
    df.index = pd.period_range("2023Q1", periods=n, freq="Q").to_timestamp()
    return df
