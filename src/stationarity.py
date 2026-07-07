"""Stationarity testing utilities."""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, series_name: str) -> dict[str, object]:
    """Run Augmented Dickey-Fuller test and return documented results."""
    clean = series.dropna()
    if clean.empty:
        raise ValueError(f"{series_name} has no non-missing values for ADF test.")
    statistic, pvalue, used_lag, nobs, critical_values, icbest = adfuller(clean, autolag="AIC")
    interpretation = (
        "Stationary: reject the unit-root null hypothesis at 5%."
        if pvalue < 0.05
        else "Non-stationary: fail to reject the unit-root null hypothesis at 5%."
    )
    return {
        "series": series_name,
        "adf_statistic": statistic,
        "p_value": pvalue,
        "used_lag": used_lag,
        "n_observations": nobs,
        "critical_1pct": critical_values.get("1%"),
        "critical_5pct": critical_values.get("5%"),
        "critical_10pct": critical_values.get("10%"),
        "icbest": icbest,
        "interpretation": interpretation,
    }


def run_adf_for_prices_and_returns(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Run ADF tests for adjusted close prices and daily returns for each ticker."""
    rows = []
    for ticker in prices.columns:
        rows.append(adf_test(prices[ticker], f"{ticker}_adjusted_close"))
        rows.append(adf_test(returns[ticker], f"{ticker}_daily_return"))
    return pd.DataFrame(rows)
