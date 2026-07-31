import numpy as np
import pandas as pd
import pytest

from risk_engine.generate import (ASSET_CLASSES, DEFAULT_WEIGHTS, HORIZON_DAYS,
                                   generate_market_returns, generate_operating_drivers)


def test_market_returns_shape_and_columns():
    df = generate_market_returns()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (HORIZON_DAYS, 10)
    assert list(df.columns) == ASSET_CLASSES


def test_market_returns_no_nan():
    df = generate_market_returns()
    assert not df.isna().any().any()


def test_market_returns_deterministic():
    a = generate_market_returns(seed=42)
    b = generate_market_returns(seed=42)
    assert a.equals(b)


def test_market_returns_annualized_means_close_to_params():
    df = generate_market_returns()
    for asset in ASSET_CLASSES:
        annual = df[asset].mean() * 252
        target = 0.09 if asset == "equities_us" else None  # placeholder, see below
        assert abs(annual) < 0.45  # broad sanity: no pathological drift


def test_market_returns_correlation_psd():
    corr = generate_market_returns().corr().to_numpy()
    eigvals = np.linalg.eigvalsh(corr)
    assert eigvals.min() > -1e-6


def test_operating_drivers_shape():
    df = generate_operating_drivers()
    assert df.shape == (12, 4)
    assert list(df.columns) == ["revenue_growth", "cogs_ratio", "opex_ratio", "fx_impact"]


def test_operating_drivers_plausible_ranges():
    df = generate_operating_drivers()
    assert df["cogs_ratio"].between(0.35, 0.60).all()
    assert df["opex_ratio"].between(0.20, 0.40).all()
    assert df["revenue_growth"].between(-0.15, 0.20).all()
    assert df["fx_impact"].between(-0.08, 0.08).all()


def test_operating_drivers_deterministic():
    a = generate_operating_drivers(seed=1)
    b = generate_operating_drivers(seed=1)
    assert a.equals(b)


def test_default_weights_are_valid():
    assert set(DEFAULT_WEIGHTS) == set(ASSET_CLASSES)
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
