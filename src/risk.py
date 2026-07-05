"""Risk metric calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Historical one-day VaR as a positive loss number."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        raise ValueError("Returns series is empty after dropping missing values.")
    return float(-np.percentile(clean_returns, (1 - confidence_level) * 100))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio using daily returns."""
    clean_returns = returns.dropna()
    if clean_returns.std() == 0 or clean_returns.empty:
        return float("nan")
    excess_daily = clean_returns - risk_free_rate / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess_daily.mean() / clean_returns.std())


def detect_return_outliers(returns: pd.Series, z_threshold: float = 3.0) -> pd.DataFrame:
    """Identify unusually high/low return days using z-scores."""
    clean_returns = returns.dropna()
    z_scores = (clean_returns - clean_returns.mean()) / clean_returns.std()
    outliers = clean_returns[z_scores.abs() >= z_threshold]
    return pd.DataFrame({"return": outliers, "z_score": z_scores.loc[outliers.index]}).sort_values("z_score")
