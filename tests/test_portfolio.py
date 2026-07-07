import numpy as np
import pandas as pd

from src.portfolio import (
    annualized_covariance_matrix,
    annualized_expected_returns,
    optimize_max_sharpe,
    optimize_min_volatility,
    portfolio_return,
    portfolio_volatility,
)


def sample_returns():
    return pd.DataFrame(
        {
            "TSLA": [0.01, -0.02, 0.03, 0.01, -0.005],
            "BND": [0.001, 0.002, -0.001, 0.001, 0.0005],
            "SPY": [0.004, -0.003, 0.006, 0.002, -0.001],
        }
    )


def test_expected_returns_and_covariance():
    returns = sample_returns()
    expected = annualized_expected_returns(returns, tsla_forecast_return=0.2)
    cov = annualized_covariance_matrix(returns)
    assert expected.loc["TSLA"] == 0.2
    assert cov.shape == (3, 3)


def test_portfolio_metrics_positive_volatility():
    returns = sample_returns()
    expected = annualized_expected_returns(returns)
    cov = annualized_covariance_matrix(returns)
    weights = np.array([0.3, 0.4, 0.3])
    assert isinstance(portfolio_return(weights, expected), float)
    assert portfolio_volatility(weights, cov) >= 0


def test_optimization_weights_sum_to_one():
    returns = sample_returns()
    expected = annualized_expected_returns(returns)
    cov = annualized_covariance_matrix(returns)
    max_sharpe = optimize_max_sharpe(expected, cov)
    min_vol = optimize_min_volatility(expected, cov)
    assert abs(max_sharpe["weights"].sum() - 1) < 1e-6
    assert abs(min_vol["weights"].sum() - 1) < 1e-6
