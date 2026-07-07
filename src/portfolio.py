"""Modern Portfolio Theory optimization utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import RISK_FREE_RATE, TRADING_DAYS


def annualized_expected_returns(returns: pd.DataFrame, tsla_forecast_return: float | None = None) -> pd.Series:
    """Prepare annualized expected returns.

    BND and SPY use historical annualized average returns. TSLA can be overridden with the
    forecast-implied annualized return from Task 3.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    expected = returns.mean() * TRADING_DAYS
    if tsla_forecast_return is not None and "TSLA" in expected.index:
        expected.loc["TSLA"] = tsla_forecast_return
    return expected


def annualized_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized covariance matrix from daily returns."""
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    return returns.cov() * TRADING_DAYS


def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    """Expected annual portfolio return."""
    return float(np.dot(weights, expected_returns.values))


def portfolio_volatility(weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
    """Expected annual portfolio volatility."""
    cov = covariance_matrix.values
    return float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))


def portfolio_sharpe(weights: np.ndarray, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Expected Sharpe Ratio for a portfolio."""
    vol = portfolio_volatility(weights, covariance_matrix)
    if vol == 0:
        return np.nan
    return (portfolio_return(weights, expected_returns) - risk_free_rate) / vol


def _constraints(num_assets: int):
    return ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)


def _bounds(num_assets: int):
    return tuple((0.0, 1.0) for _ in range(num_assets))


def optimize_max_sharpe(expected_returns: pd.Series, covariance_matrix: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> dict[str, object]:
    """Find maximum Sharpe Ratio long-only portfolio."""
    num_assets = len(expected_returns)
    initial = np.repeat(1 / num_assets, num_assets)

    def objective(weights):
        return -portfolio_sharpe(weights, expected_returns, covariance_matrix, risk_free_rate)

    result = minimize(objective, initial, method="SLSQP", bounds=_bounds(num_assets), constraints=_constraints(num_assets))
    if not result.success:
        raise RuntimeError(f"Max Sharpe optimization failed: {result.message}")
    return summarize_weights(result.x, expected_returns, covariance_matrix, "Maximum Sharpe Portfolio", risk_free_rate)


def optimize_min_volatility(expected_returns: pd.Series, covariance_matrix: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> dict[str, object]:
    """Find minimum volatility long-only portfolio."""
    num_assets = len(expected_returns)
    initial = np.repeat(1 / num_assets, num_assets)

    def objective(weights):
        return portfolio_volatility(weights, covariance_matrix)

    result = minimize(objective, initial, method="SLSQP", bounds=_bounds(num_assets), constraints=_constraints(num_assets))
    if not result.success:
        raise RuntimeError(f"Min volatility optimization failed: {result.message}")
    return summarize_weights(result.x, expected_returns, covariance_matrix, "Minimum Volatility Portfolio", risk_free_rate)


def summarize_weights(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    label: str,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, object]:
    """Create a dictionary of weights and portfolio metrics."""
    weight_series = pd.Series(weights, index=expected_returns.index, name="weight")
    return {
        "label": label,
        "weights": weight_series,
        "expected_return": portfolio_return(weights, expected_returns),
        "volatility": portfolio_volatility(weights, covariance_matrix),
        "sharpe_ratio": portfolio_sharpe(weights, expected_returns, covariance_matrix, risk_free_rate),
    }


def efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    points: int = 50,
) -> pd.DataFrame:
    """Generate efficient frontier by minimizing volatility at target returns."""
    if points < 2:
        raise ValueError("points must be >= 2.")
    num_assets = len(expected_returns)
    initial = np.repeat(1 / num_assets, num_assets)
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), points)
    rows = []

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
            {"type": "eq", "fun": lambda weights, target=target: portfolio_return(weights, expected_returns) - target},
        )
        result = minimize(
            lambda weights: portfolio_volatility(weights, covariance_matrix),
            initial,
            method="SLSQP",
            bounds=_bounds(num_assets),
            constraints=constraints,
        )
        if result.success:
            vol = portfolio_volatility(result.x, covariance_matrix)
            rows.append(
                {
                    "target_return": target,
                    "volatility": vol,
                    "sharpe_ratio": (target - RISK_FREE_RATE) / vol if vol else np.nan,
                    **{f"weight_{ticker}": weight for ticker, weight in zip(expected_returns.index, result.x)},
                }
            )
    return pd.DataFrame(rows)


def recommendation_table(portfolio_result: dict[str, object]) -> pd.DataFrame:
    """Convert recommended portfolio result into a compact table."""
    weights = portfolio_result["weights"]
    rows = [{"metric": f"weight_{ticker}", "value": float(value)} for ticker, value in weights.items()]
    rows.extend(
        [
            {"metric": "expected_annual_return", "value": float(portfolio_result["expected_return"])},
            {"metric": "expected_annual_volatility", "value": float(portfolio_result["volatility"])},
            {"metric": "expected_sharpe_ratio", "value": float(portfolio_result["sharpe_ratio"])},
        ]
    )
    return pd.DataFrame(rows)
