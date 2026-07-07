"""Risk and performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RISK_FREE_RATE, TRADING_DAYS


def calculate_var(returns: pd.Series | pd.DataFrame, confidence: float = 0.95) -> pd.Series:
    """Calculate historical Value at Risk as a positive loss number."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    quantile = returns.quantile(1 - confidence)
    return -quantile


def calculate_sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series | float:
    """Calculate annualized Sharpe Ratio for daily returns."""
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    excess_daily = returns - risk_free_rate / periods_per_year
    volatility = returns.std(ddof=1)
    if isinstance(volatility, pd.Series):
        volatility = volatility.replace(0, np.nan)
    elif volatility == 0:
        return float("nan")
    return np.sqrt(periods_per_year) * excess_daily.mean() / volatility


def annualized_return(returns: pd.Series | pd.DataFrame, periods_per_year: int = TRADING_DAYS):
    """Calculate annualized arithmetic mean return."""
    return returns.mean() * periods_per_year


def annualized_volatility(returns: pd.Series | pd.DataFrame, periods_per_year: int = TRADING_DAYS):
    """Calculate annualized volatility."""
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def max_drawdown(return_series: pd.Series) -> float:
    """Calculate maximum drawdown from daily returns."""
    if return_series.empty:
        raise ValueError("return_series cannot be empty.")
    wealth_index = (1 + return_series.fillna(0)).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = wealth_index / previous_peaks - 1
    return float(drawdowns.min())


def performance_summary(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> dict[str, float]:
    """Summarize common return/risk metrics for a daily portfolio return series."""
    if returns.empty:
        raise ValueError("returns cannot be empty.")
    clean_returns = returns.dropna()
    total_return = float((1 + clean_returns).prod() - 1)
    ann_return = float(annualized_return(clean_returns))
    sharpe = float(calculate_sharpe_ratio(clean_returns, risk_free_rate=risk_free_rate))
    mdd = max_drawdown(clean_returns)
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": float(annualized_volatility(clean_returns)),
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
    }


def calculate_risk_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Create VaR and Sharpe Ratio table for all assets."""
    return pd.DataFrame(
        {
            "VaR_95": calculate_var(returns, confidence=0.95),
            "Sharpe_Ratio": calculate_sharpe_ratio(returns),
            "Annualized_Return": annualized_return(returns),
            "Annualized_Volatility": annualized_volatility(returns),
        }
    )
