import pytest

from risk_engine.portfolio import Portfolio
from risk_engine.var import VaRCalculator


@pytest.fixture
def calculator():
    return VaRCalculator(Portfolio())


def test_historical_var_positive_and_bounded(calculator):
    r = calculator.historical_var()
    assert r["method"] == "historical"
    assert 0 < r["var"] < 0.05 * r["portfolio_value"]
    assert 0 < r["var_pct"] < 5.0


def test_parametric_var_same_ballpark_as_historical(calculator):
    h = calculator.historical_var()["var"]
    p = calculator.parametric_var()["var"]
    assert 0.5 * h < p < 2.0 * h


def test_monte_carlo_var_same_ballpark_as_historical(calculator):
    h = calculator.historical_var()["var"]
    m = calculator.monte_carlo_var(n_sims=20_000)["var"]
    assert 0.3 * h < m < 3.0 * h


@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
def test_cvar_exceeds_var(calculator, method):
    var = calculator.all_methods() if False else None
    v = {"historical": calculator.historical_var,
         "parametric": calculator.parametric_var,
         "monte_carlo": lambda: calculator.monte_carlo_var(10_000)}[method]()["var"]
    c = calculator.cvar(method=method, n_sims=10_000)["var"]
    assert c >= v - 1e-6


def test_all_methods_structure(calculator):
    results = calculator.all_methods(n_sims=5_000)
    assert len(results) == 3
    for r in results:
        assert set(r) >= {"method", "var", "var_pct", "cvar", "cvar_pct",
                          "confidence", "horizon_days", "portfolio_value"}


def test_horizon_scales_by_sqrt(calculator):
    h1 = VaRCalculator(Portfolio(), horizon=1).historical_var()["var"]
    h10 = VaRCalculator(Portfolio(), horizon=10).historical_var()["var"]
    assert abs(h10 / h1 - 10 ** 0.5) < 0.01


def test_higher_confidence_higher_var():
    low = VaRCalculator(Portfolio(), confidence=0.95).historical_var()["var"]
    high = VaRCalculator(Portfolio(), confidence=0.99).historical_var()["var"]
    assert high > low


def test_invalid_inputs():
    with pytest.raises(ValueError):
        VaRCalculator(Portfolio(), confidence=1.5)
    with pytest.raises(ValueError):
        VaRCalculator(Portfolio(), confidence=0.0)
    with pytest.raises(ValueError):
        VaRCalculator(Portfolio(), horizon=0)


def test_cvar_invalid_method(calculator):
    with pytest.raises(ValueError):
        calculator.cvar(method="quantum")
