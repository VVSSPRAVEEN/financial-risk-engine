"""Risk reporting: charts + markdown risk report.

`build_risk_report` produces docs/risk_report.md plus 4 PNGs:
- portfolio_returns_hist.png  : daily P&L distribution with VaR/CVaR lines
- monte_carlo_paths.png       : 200 sampled GBM portfolio paths
- var_comparison.png          : VaR by method + CVaR (bar chart)
- cfar_distribution.png       : annual EBITDA histogram with p5/p50 lines
- stress_impacts.png          : P&L impact per scenario (horizontal bars)
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from risk_engine.backtest import backtest_exceptions, kupiec_pof
from risk_engine.cfar import CashFlowAtRisk
from risk_engine.portfolio import Portfolio
from risk_engine.stress import StressTester
from risk_engine.var import VaRCalculator

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.35,
                     "axes.spines.top": False, "axes.spines.right": False})


def _fmt_money(x, _pos=None):
    return f"${x/1e6:,.0f}M"


def _chart_returns_hist(portfolio, var_results, out_dir: pathlib.Path) -> None:
    pnl = portfolio.daily_pnl().to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(pnl / 1e6, bins=80, color="#1f77b4", alpha=0.75)
    for r in var_results:
        ax.axvline(-r["var"] / 1e6, ls="--", lw=1.3,
                   label=f"{r['method']} VaR ${r['var']/1e6:,.1f}M")
    ax.axvline(-var_results[0]["cvar"] / 1e6, ls=":", lw=1.6, color="red",
               label=f"CVaR ${var_results[0]['cvar']/1e6:,.1f}M")
    ax.set_title("Daily portfolio P&L distribution with VaR / CVaR lines")
    ax.set_xlabel("Daily P&L ($M)"); ax.set_ylabel("days")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_fmt_money))
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "portfolio_returns_hist.png"); plt.close(fig)


def _chart_mc_paths(portfolio, out_dir: pathlib.Path, n_paths: int = 200) -> None:
    rng = np.random.default_rng(7)
    mu = portfolio.expected_return(annualized=False)
    sigma = portfolio.volatility(annualized=False)
    days = 252
    z = rng.standard_normal((n_paths, days))
    log_ret = (mu - 0.5 * sigma ** 2) + sigma * z
    paths = portfolio.value * np.exp(np.cumsum(log_ret, axis=1))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for path in paths[::20]:
        ax.plot(path / 1e6, lw=0.7, alpha=0.55)
    ax.set_title("Monte Carlo portfolio paths (1 year, 200 paths)")
    ax.set_xlabel("trading day"); ax.set_ylabel("portfolio value ($M)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_fmt_money))
    fig.tight_layout(); fig.savefig(out_dir / "monte_carlo_paths.png"); plt.close(fig)


def _chart_var_comparison(var_results, out_dir: pathlib.Path) -> None:
    methods = [f"{r['method']}\nVaR" for r in var_results] + ["CVaR\n(historical)"]
    values = [r["var"] / 1e6 for r in var_results] + [var_results[0]["cvar"] / 1e6]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(methods, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"${v:,.1f}M",
                ha="center", fontsize=9)
    ax.set_title("VaR by method and CVaR (1-day, 95%)")
    ax.set_ylabel("loss ($M)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_fmt_money))
    fig.tight_layout(); fig.savefig(out_dir / "var_comparison.png"); plt.close(fig)


def _chart_cfar(cfar: CashFlowAtRisk, out_dir: pathlib.Path) -> None:
    annual = cfar.simulate().sum(axis=1) / 1e6
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(annual, bins=70, color="#2ca02c", alpha=0.75)
    p5, p50, mean = np.percentile(annual, 5), np.percentile(annual, 50), annual.mean()
    for val, label, color in ((mean, f"mean ${mean:,.0f}M", "#1f77b4"),
                              (p50, f"p50 ${p50:,.0f}M", "#ff7f0e"),
                              (p5, f"p5 ${p5:,.0f}M", "#d62728")):
        ax.axvline(val, ls="--", color=color, lw=1.4, label=label)
    ax.set_title("Annual cash flow distribution (CFaR view)")
    ax.set_xlabel("annual EBITDA ($M)"); ax.set_ylabel("simulations")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_fmt_money))
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "cfar_distribution.png"); plt.close(fig)


def _chart_stress(stress: StressTester, out_dir: pathlib.Path) -> None:
    table = stress.run_all().sort_values("total_pct")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in table["total_pct"]]
    ax.barh(table["scenario"], table["total_pct"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Stress scenario impact on portfolio value (%)")
    ax.set_xlabel("P&L impact (% of portfolio)")
    for i, v in enumerate(table["total_pct"]):
        ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "stress_impacts.png"); plt.close(fig)


def _md_table(df: pd.DataFrame, float_fmt: str = "{:,.2f}") -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [float_fmt.format(v) if isinstance(v, (int, float)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_risk_report(portfolio: Portfolio, out_dir: str | pathlib.Path = "docs",
                      confidence: float = 0.95) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    calculator = VaRCalculator(portfolio, confidence=confidence)
    var_results = calculator.all_methods()
    stress = StressTester(portfolio)
    cfar = CashFlowAtRisk()

    _chart_returns_hist(portfolio, var_results, out)
    _chart_mc_paths(portfolio, out)
    _chart_var_comparison(var_results, out)
    _chart_cfar(cfar, out)
    _chart_stress(stress, out)

    # Backtest: compare historical 95% VaR forecast against realized P&L
    pnl = portfolio.daily_pnl().to_numpy()
    var_forecasts = np.full(len(pnl), calculator.historical_var()["var"])
    bt = backtest_exceptions(portfolio, var_forecasts, confidence)
    kupiec = kupiec_pof(bt["exceptions"], bt["n_days"], confidence)

    summary = portfolio.summary()
    var_table = pd.DataFrame([{
        "method": r["method"], "var": r["var"], "var_pct": r["var_pct"],
        "cvar": r["cvar"], "cvar_pct": r["cvar_pct"]} for r in var_results])
    stress_table = stress.run_all()
    quarterly = cfar.quarterly_cfar().head(4)
    annual = cfar.annual_cfar()

    md = [
        "# Risk Report — Financial Risk Engine",
        "",
        "## Portfolio",
        "",
        f"- Value: ${portfolio.value/1e6:,.1f}M | Expected return (ann): {summary['expected_return_ann']:.1%} "
        f"| Volatility (ann): {summary['volatility_ann']:.1%} | Sharpe (rf=3%): {summary['sharpe']}",
        f"- Assets: {summary['n_assets']} classes | Daily observations: {summary['n_days']}",
        "",
        "## Value-at-Risk (1-day, 95%)",
        "",
        _md_table(var_table),
        "",
        "## Backtest (historical VaR vs realized P&L)",
        "",
        f"- Exceptions: {bt['exceptions']} of {bt['n_days']} days "
        f"(expected {bt['expected_exceptions']}; rate {bt['exception_rate']:.4%})",
        f"- Kupiec POF LR = {kupiec['lr_stat']:.2f} -> {kupiec['message']}",
        "",
        "## Stress scenarios",
        "",
        _md_table(stress_table),
        "",
        "## Cash-Flow-at-Risk (annual)",
        "",
        f"- Mean annual EBITDA: ${annual['annual_mean']/1e6:,.1f}M | p5: ${annual['annual_p5']/1e6:,.1f}M "
        f"| p1: ${annual['annual_p1']/1e6:,.1f}M",
        f"- CFaR at 95%: ${annual['cfar_95']/1e6:,.1f}M "
        f"({annual['cfar_95_pct_of_revenue']:.1f}% of revenue) | "
        f"P(negative EBITDA): {annual['prob_negative_ebitda']:.2%}",
        "",
        "## Charts",
        "",
        "![Returns histogram](portfolio_returns_hist.png)",
        "![Monte Carlo paths](monte_carlo_paths.png)",
        "![VaR comparison](var_comparison.png)",
        "![CFaR distribution](cfar_distribution.png)",
        "![Stress impacts](stress_impacts.png)",
        "",
    ]
    (out / "risk_report.md").write_text("\n".join(md), encoding="utf-8")
    return {"markdown": str(out / "risk_report.md"),
            "screenshots": sorted(str(p) for p in out.glob("*.png"))}
