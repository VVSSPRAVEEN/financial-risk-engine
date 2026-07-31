"""Backtesting of VaR models (Basel-style validation).

`backtest_exceptions`: count the days where the realized loss exceeded
the forecast VaR (expected = horizon * (1 - confidence) exceptions for
an accurate model).

`kupiec_pof`: the Kupiec Proportion-of-Failures likelihood-ratio test —
LR ~ chi2(1); LR > 3.841 rejects the model at 95% confidence.
"""
from __future__ import annotations

import numpy as np

from risk_engine.portfolio import Portfolio
from risk_engine.var import VaRCalculator


def backtest_exceptions(portfolio: Portfolio, var_forecasts: np.ndarray,
                        confidence: float = 0.95) -> dict:
    """Compare realized daily P&L against a vector of VaR forecasts."""
    pnl = portfolio.daily_pnl().to_numpy()
    if len(var_forecasts) != len(pnl):
        raise ValueError("var_forecasts must have one entry per day")
    exceptions = (pnl < -var_forecasts).sum()
    expected = len(pnl) * (1 - confidence)
    return {
        "n_days": len(pnl),
        "exceptions": int(exceptions),
        "expected_exceptions": round(expected, 2),
        "exception_rate": round(float(exceptions) / len(pnl), 5),
        "model_ok": bool(exceptions <= expected * 1.5),
    }


def kupiec_pof(exceptions: int, n_days: int, confidence: float = 0.95) -> dict:
    """Kupiec Proportion-of-Failures test.

    LR = -2 * ln( ((1-p)^(n-x) * p^x) / ((1-x/n)^(n-x) * (x/n)^x) )
    Reject at 95% if LR > 3.841 (chi2_0.95 with 1 dof).
    """
    if exceptions == 0 or exceptions == n_days:
        return {"lr_stat": float("inf"), "rejected": True,
                "message": "Zero or 100% exceptions - degenerate"}
    p = 1 - confidence
    x, n = float(exceptions), float(n_days)
    lr = -2.0 * (np.log((1 - p) ** (n - x) * p ** x) -
                 np.log((1 - x / n) ** (n - x) * (x / n) ** x))
    rejected = lr > 3.841
    return {"lr_stat": round(float(lr), 4), "rejected": bool(rejected),
            "message": "reject model at 95%" if rejected else "model not rejected at 95%"}
