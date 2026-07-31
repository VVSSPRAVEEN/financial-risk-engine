"""Scenario-based stress testing.

`StressTester` applies named macro shocks to the portfolio and reports
the P&L impact per asset class plus the total. Scenarios mirror the
shocks risk committees actually model:

- market_crash   : equities -25%, EM -30%, crypto -40%, RE -15%
- rate_hike      : bonds -8%, equities -5%, cash +0 (no loss)
- commodity_shock: commodities -30%, gold +10% (flight to safety)
- currency_crisis: EM -20%, commodities -10%, gold +15%, US eq -5%
- stagflation    : broad -12% risk assets, gold +8%, bonds -4%
- covid_style    : equities -20%, bonds +3%, gold +12%, cash +0

Shocks are expressed as fractional price moves (negative = loss).
"""
from __future__ import annotations

import pandas as pd

from risk_engine.portfolio import Portfolio

SCENARIOS: dict[str, dict[str, float]] = {
    "market_crash": {"equities_us": -0.25, "equities_eu": -0.25, "equities_em": -0.30,
                     "bonds_gov": 0.02, "bonds_corp": -0.06, "real_estate": -0.15,
                     "commodities": -0.12, "gold": 0.05, "crypto": -0.40, "cash": 0.0},
    "rate_hike": {"equities_us": -0.05, "equities_eu": -0.05, "equities_em": -0.08,
                  "bonds_gov": -0.08, "bonds_corp": -0.10, "real_estate": -0.06,
                  "commodities": -0.04, "gold": -0.03, "crypto": -0.12, "cash": 0.0},
    "commodity_shock": {"equities_us": -0.03, "equities_eu": -0.03, "equities_em": -0.08,
                        "bonds_gov": 0.01, "bonds_corp": -0.02, "real_estate": -0.04,
                        "commodities": -0.30, "gold": 0.10, "crypto": -0.10, "cash": 0.0},
    "currency_crisis": {"equities_us": -0.05, "equities_eu": -0.10, "equities_em": -0.20,
                        "bonds_gov": 0.01, "bonds_corp": -0.05, "real_estate": -0.06,
                        "commodities": -0.10, "gold": 0.15, "crypto": -0.25, "cash": 0.0},
    "stagflation": {"equities_us": -0.12, "equities_eu": -0.12, "equities_em": -0.15,
                    "bonds_gov": -0.04, "bonds_corp": -0.08, "real_estate": -0.10,
                    "commodities": 0.05, "gold": 0.08, "crypto": -0.20, "cash": 0.0},
    "covid_style": {"equities_us": -0.20, "equities_eu": -0.22, "equities_em": -0.25,
                    "bonds_gov": 0.03, "bonds_corp": -0.04, "real_estate": -0.08,
                    "commodities": -0.15, "gold": 0.12, "crypto": -0.30, "cash": 0.0},
}


class StressTester:
    def __init__(self, portfolio: Portfolio, scenarios: dict | None = None):
        self.portfolio = portfolio
        self.scenarios = dict(SCENARIOS if scenarios is None else scenarios)

    def apply_shock(self, name: str) -> dict:
        """Apply one named scenario; returns per-asset impact + total P&L."""
        if name not in self.scenarios:
            raise KeyError(f"Unknown scenario '{name}'. Available: {sorted(self.scenarios)}")
        shock = self.scenarios[name]
        rows = []
        total = 0.0
        for asset, weight in self.portfolio.weights.items():
            impact = self.portfolio.value * weight * shock.get(asset, 0.0)
            total += impact
            rows.append({"asset": asset, "weight": weight,
                         "shock_pct": shock.get(asset, 0.0) * 100,
                         "pnl": round(impact, 2)})
        return {"scenario": name,
                "rows": pd.DataFrame(rows).sort_values("pnl"),
                "total_pnl": round(total, 2),
                "total_pct": round(total / self.portfolio.value * 100, 3)}

    def run_all(self) -> pd.DataFrame:
        """All scenarios in one table: total P&L and % per scenario."""
        rows = []
        for name in sorted(self.scenarios):
            result = self.apply_shock(name)
            rows.append({"scenario": name, "total_pnl": result["total_pnl"],
                         "total_pct": result["total_pct"]})
        return pd.DataFrame(rows).sort_values("total_pnl")

    def worst_scenario(self) -> str:
        table = self.run_all()
        return str(table.iloc[0]["scenario"])
