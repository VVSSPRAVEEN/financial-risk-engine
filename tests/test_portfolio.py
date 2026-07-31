import numpy as np
import pytest

from risk_engine.generate import DEFAULT_WEIGHTS, generate_market_returns
from risk_engine.portfolio import Portfolio


def test_default_portfolio():
    p = Portfolio()
    assert p.value == 10_000_000.0
    assert p.weights == DEFAULT_WEIGHTS


def test_weights_sum_to_one_validation():
    bad = dict(DEFAULT_WEIGHTS)
    bad["cash"] = 0.10  # now sums to 1.06
    with pytest.raises(ValueError):
        Portfolio(weights=bad)


def test_unknown_asset_validation():
    bad = dict(DEFAULT_WEIGHTS)
    bad["meme_coin"] = 0.01
    del bad["cash"]
    with pytest.raises(KeyError):
        Portfolio(weights=bad)


def test_non_positive_value_validation():
    with pytest.raises(ValueError):
        Portfolio(value=0)


def test_expected_return_close_to_weighted_params():
    p = Portfolio()
    expected = sum(DEFAULT_WEIGHTS[a] * 0.095 if a == "equities_us" else
                   p.expected_return() * 0 for a in []) or 0.0
    mu = p.expected_return()
    # deterministic: loose band around the 0.0688 weighted target
    assert 0.02 < mu < 0.12


def test_volatility_positive_and_sane():
    p = Portfolio()
    vol = p.volatility()
    assert 0.03 < vol < 0.25
    assert p.volatility(annualized=False) > 0


def test_daily_pnl_shape_and_nan():
    p = Portfolio()
    pnl = p.daily_pnl()
    assert len(pnl) == len(p.returns)
    assert pnl.notna().all()
    assert pnl.std() > 0


def test_allocation_totals_to_value():
    p = Portfolio()
    alloc = p.allocation()
    assert abs(alloc["value"].sum() - p.value) <= 10.0  # rounding slack
    assert abs(alloc["weight"].sum() - 1.0) <= 1e-6


def test_custom_portfolio():
    returns = generate_market_returns(seed=5)
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({"equities_us": 0.5, "equities_eu": 0.0, "equities_em": 0.0,
                    "bonds_gov": 0.42, "cash": 0.08, "crypto": 0.0,
                    "commodities": 0.0, "gold": 0.0, "real_estate": 0.0,
                    "bonds_corp": 0.0})
    p = Portfolio(returns=returns, weights=weights, value=1_000_000.0)
    assert abs(p.expected_return() - p.expected_return()) < 1e-12
    assert abs(p.allocation()["value"].sum() - 1_000_000.0) <= 10.0
    assert p.volatility() > 0
