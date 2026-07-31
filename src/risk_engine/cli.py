"""Command-line interface for the Financial Risk Engine.

Usage (from repo root):
    python -m risk_engine.cli simulate
    python -m risk_engine.cli var [--confidence 0.95] [--horizon 1] [--n-sims 20000]
    python -m risk_engine.cli backtest
    python -m risk_engine.cli stress
    python -m risk_engine.cli cfar
    python -m risk_engine.cli report [--out docs]
    python -m risk_engine.cli all
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from risk_engine.backtest import backtest_exceptions, kupiec_pof  # noqa: E402
from risk_engine.cfar import CashFlowAtRisk  # noqa: E402
from risk_engine.portfolio import Portfolio  # noqa: E402
from risk_engine.report import build_risk_report  # noqa: E402
from risk_engine.stress import StressTester  # noqa: E402
from risk_engine.var import VaRCalculator  # noqa: E402


def _table(df: pd.DataFrame) -> None:
    print(df.to_string(index=False))


def cmd_simulate(args) -> int:
    portfolio = Portfolio()
    summary = portfolio.summary()
    print("Portfolio summary:")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print("\nAllocation:")
    _table(portfolio.allocation())
    return 0


def cmd_var(args) -> int:
    calculator = VaRCalculator(Portfolio(), confidence=args.confidence,
                               horizon=args.horizon)
    results = calculator.all_methods(n_sims=args.n_sims)
    print(f"VaR / CVaR ({args.confidence:.0%} confidence, {args.horizon}-day horizon):")
    _table(pd.DataFrame([{
        "method": r["method"], "var": r["var"], "var_pct": r["var_pct"],
        "cvar": r["cvar"], "cvar_pct": r["cvar_pct"]} for r in results]))
    return 0


def cmd_backtest(args) -> int:
    portfolio = Portfolio()
    calculator = VaRCalculator(portfolio, confidence=args.confidence)
    pnl = portfolio.daily_pnl().to_numpy()
    var_forecasts = pd.Series(pnl).rolling(args.window, min_periods=args.window).apply(
        lambda w: -float(np.percentile(w, (1 - args.confidence) * 100)), raw=True)
    var_forecasts = var_forecasts.fillna(calculator.historical_var()["var"]).to_numpy()
    bt = backtest_exceptions(portfolio, var_forecasts, args.confidence)
    kupiec = kupiec_pof(bt["exceptions"], bt["n_days"], args.confidence)
    print("Rolling-window VaR backtest:")
    for k, v in bt.items():
        print(f"  {k:22s} {v}")
    print(f"  Kupiec LR = {kupiec['lr_stat']} -> {kupiec['message']}")
    return 0


def cmd_stress(args) -> int:
    stress = StressTester(Portfolio())
    print("Stress scenarios (P&L impact):")
    _table(stress.run_all())
    print(f"\nWorst scenario: {stress.worst_scenario()}")
    return 0


def cmd_cfar(args) -> int:
    cfar = CashFlowAtRisk(annual_revenue=args.revenue, n_sims=args.n_sims)
    print("Quarterly cash-flow distribution (first 4 quarters):")
    _table(cfar.quarterly_cfar().head(4))
    annual = cfar.annual_cfar()
    print("\nAnnual CFaR:")
    for k, v in annual.items():
        print(f"  {k:24s} {v}")
    return 0


def cmd_report(args) -> int:
    portfolio = Portfolio()
    result = build_risk_report(portfolio, out_dir=args.out)
    print(f"Risk report written: {result['markdown']}")
    for p in result["screenshots"]:
        print("  " + p)
    return 0


def cmd_all(args) -> int:
    from argparse import Namespace
    cmd_simulate(Namespace())
    cmd_var(Namespace(confidence=0.95, horizon=1, n_sims=20_000))
    cmd_stress(Namespace())
    cmd_cfar(Namespace(revenue=500_000_000.0, n_sims=10_000))
    cmd_report(Namespace(out="docs"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Financial Risk Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="Portfolio summary and allocation")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("var", help="VaR / CVaR by all three methods")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--n-sims", type=int, default=20_000)
    p.set_defaults(func=cmd_var)

    p = sub.add_parser("backtest", help="Rolling-window VaR backtest + Kupiec test")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--window", type=int, default=252)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("stress", help="Run all stress scenarios")
    p.set_defaults(func=cmd_stress)

    p = sub.add_parser("cfar", help="Cash-Flow-at-Risk simulation")
    p.add_argument("--revenue", type=float, default=500_000_000.0)
    p.add_argument("--n-sims", type=int, default=10_000)
    p.set_defaults(func=cmd_cfar)

    p = sub.add_parser("report", help="Generate risk report (charts + markdown)")
    p.add_argument("--out", default="docs")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("all", help="simulate + var + stress + cfar + report")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
