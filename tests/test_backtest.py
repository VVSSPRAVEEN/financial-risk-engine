import numpy as np
import pytest

from risk_engine.backtest import backtest_exceptions, kupiec_pof
from risk_engine.portfolio import Portfolio


@pytest.fixture
def portfolio():
    return Portfolio()


def test_backtest_zero_forecasts_counts_loss_days(portfolio):
    zeros = np.zeros(len(portfolio.daily_pnl()))
    bt = backtest_exceptions(portfolio, zeros, confidence=0.95)
    negatives = int((portfolio.daily_pnl() < 0).sum())
    assert bt["exceptions"] == negatives
    assert bt["n_days"] == len(portfolio.daily_pnl())
    assert bt["expected_exceptions"] == round(len(portfolio.daily_pnl()) * 0.05, 2)


def test_backtest_huge_forecast_no_exceptions(portfolio):
    huge = np.full(len(portfolio.daily_pnl()), 1e12)
    bt = backtest_exceptions(portfolio, huge, confidence=0.95)
    assert bt["exceptions"] == 0
    assert bt["exception_rate"] == 0.0
    assert bt["model_ok"] is True


def test_backtest_length_mismatch_raises(portfolio):
    with pytest.raises(ValueError):
        backtest_exceptions(portfolio, np.zeros(10))


def test_kupiec_degenerate_zero_exceptions():
    r = kupiec_pof(0, 1000, confidence=0.95)
    assert r["rejected"] is True
    assert "degenerate" in r["message"].lower()


def test_kupiec_accepts_good_model():
    r = kupiec_pof(50, 1000, confidence=0.95)  # exactly 5%
    assert r["lr_stat"] == 0.0
    assert r["rejected"] is False
    assert "not rejected" in r["message"]


def test_kupiec_rejects_bad_model():
    r = kupiec_pof(200, 1000, confidence=0.95)  # 20% failure rate
    assert r["rejected"] is True
    assert r["lr_stat"] > 3.841


def test_kupiec_accepts_model_with_exact_exception_count():
    r = kupiec_pof(63, 1260, confidence=0.95)
    assert r["rejected"] is False
