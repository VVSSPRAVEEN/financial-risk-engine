import pytest

from risk_engine.cfar import CashFlowAtRisk


@pytest.fixture
def cfar():
    return CashFlowAtRisk(annual_revenue=500_000_000.0, n_sims=10_000)


def test_simulate_shape(cfar):
    cash = cfar.simulate()
    assert cash.shape == (cfar.n_sims, 12)
    assert (cash > -1e6).all()  # no absurd outliers


def test_quarterly_cfar_ordering(cfar):
    q = cfar.quarterly_cfar()
    assert len(q) == 12
    assert list(q.columns) == ["quarter", "mean", "p5", "p50", "p95", "cfar_95"]
    assert (q["p5"] < q["p50"]).all()
    assert (q["p50"] < q["p95"]).all()
    assert (q["cfar_95"] > 0).all()


def test_annual_cfar_sane(cfar):
    a = cfar.annual_cfar()
    assert a["annual_mean"] > 0
    assert 0 < a["cfar_95"] < a["annual_mean"]
    assert a["annual_p1"] < a["annual_p5"] < a["annual_mean"]
    assert a["prob_negative_ebitda"] < 0.05
    assert a["worst_case"] < a["annual_mean"]
    assert a["n_sims"] == 10_000


def test_cfar_deterministic():
    a = CashFlowAtRisk(seed=42).annual_cfar()
    b = CashFlowAtRisk(seed=42).annual_cfar()
    assert a["annual_mean"] == b["annual_mean"]
    assert a["cfar_95"] == b["cfar_95"]


def test_breach_probability_matches_annual_summary(cfar):
    assert cfar.breach_probability(0.0) == cfar.annual_cfar()["prob_negative_ebitda"]


def test_breach_threshold_larger_than_zero(cfar):
    assert cfar.breach_probability(0.0) <= cfar.breach_probability(100_000_000.0)
