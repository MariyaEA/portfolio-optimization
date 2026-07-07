"""Strategy backtesting utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BACKTEST_END_DATE, BACKTEST_START_DATE
from src.risk import performance_summary


def isolate_backtest_window(
    returns: pd.DataFrame,
    start_date: str = BACKTEST_START_DATE,
    end_date: str = BACKTEST_END_DATE,
) -> pd.DataFrame:
    """Select final-year held-out backtesting window."""
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    window = returns.sort_index().loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
    if window.empty:
        raise ValueError("Backtesting window is empty. Check dates and data coverage.")
    return window.dropna(how="any")


def normalize_weights(weights: pd.Series | dict[str, float], assets: list[str]) -> pd.Series:
    """Create aligned, normalized portfolio weights."""
    weight_series = pd.Series(weights, dtype=float).reindex(assets).fillna(0.0)
    total = weight_series.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return weight_series / total


def simulate_static_portfolio(returns: pd.DataFrame, weights: pd.Series | dict[str, float]) -> pd.Series:
    """Simulate a static buy-and-hold portfolio using daily asset returns."""
    aligned_weights = normalize_weights(weights, list(returns.columns))
    portfolio_returns = returns.dot(aligned_weights)
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns


def cumulative_returns(returns: pd.Series | pd.DataFrame):
    """Compute cumulative return curve."""
    return (1 + returns.fillna(0)).cumprod() - 1


def benchmark_60_40(returns: pd.DataFrame) -> pd.Series:
    """Define the required 60% SPY / 40% BND benchmark."""
    missing = [ticker for ticker in ["SPY", "BND"] if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Benchmark requires missing assets: {missing}")
    return simulate_static_portfolio(returns, {"SPY": 0.60, "BND": 0.40})


def backtest_strategy_vs_benchmark(returns: pd.DataFrame, strategy_weights: pd.Series | dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest model-driven strategy against 60/40 benchmark."""
    strategy = simulate_static_portfolio(returns, strategy_weights)
    strategy.name = "strategy"
    benchmark = benchmark_60_40(returns)
    benchmark.name = "benchmark_60_40"
    comparison_returns = pd.concat([strategy, benchmark], axis=1)
    metrics = pd.DataFrame(
        {
            "strategy": performance_summary(comparison_returns["strategy"]),
            "benchmark_60_40": performance_summary(comparison_returns["benchmark_60_40"]),
        }
    ).T
    return comparison_returns, metrics
