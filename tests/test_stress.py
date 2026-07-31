import pandas as pd
import pytest

from risk_engine.portfolio import Portfolio
from risk_engine.stress import SCENARIOS, StressTester


@pytest.fixture
def stress():
    return StressTester(Portfolio())


def test_known_scenario_total_matches_manual_calc(stress):
    shock = SCENARIOS["market_crash"]
    p = stress.portfolio
    expected_pct = sum(p.weights[a] * s for a, s in shock.items()) * 100
    result = stress.apply_shock("market_crash")
    assert abs(result["total_pct"] - expected_pct) < 1e-9
    assert abs(result["total_pnl"] - expected_pct / 100 * p.value) < 0.01


def test_rows_sorted_ascending(stress):
    rows = stress.apply_shock("market_crash")["rows"]
    assert len(rows) == 10
    assert rows["pnl"].is_monotonic_increasing


def test_run_all_returns_all_scenarios_sorted(stress):
    table = stress.run_all()
    assert set(table["scenario"]) == set(SCENARIOS)
    assert table["total_pnl"].is_monotonic_increasing
    assert list(table.columns) == ["scenario", "total_pnl", "total_pct"]


def test_worst_scenario_is_market_crash(stress):
    assert stress.worst_scenario() == "market_crash"


def test_unknown_scenario_raises(stress):
    with pytest.raises(KeyError):
        stress.apply_shock("alien_invasion")


def test_custom_scenarios_override_defaults():
    custom = {"mini_dip": {"equities_us": -0.10}}
    stress = StressTester(Portfolio(), scenarios=custom)
    result = stress.apply_shock("mini_dip")
    assert result["scenario"] == "mini_dip"
    assert abs(result["total_pct"] + 0.10 * 0.24 * 100) < 1e-9
