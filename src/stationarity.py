"""Stationarity testing helpers."""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, series_name: str) -> dict[str, float | str | bool]:
    """Run Augmented Dickey-Fuller test and return a compact result dictionary."""
    clean_series = series.dropna()
    if len(clean_series) < 20:
        raise ValueError("ADF test requires at least 20 non-missing observations.")
    statistic, p_value, used_lag, n_obs, critical_values, _ = adfuller(clean_series)
    return {
        "series": series_name,
        "adf_statistic": float(statistic),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "stationary_at_5pct": bool(p_value < 0.05),
        "interpretation": "Stationary" if p_value < 0.05 else "Non-stationary",
        "critical_1pct": float(critical_values["1%"]),
        "critical_5pct": float(critical_values["5%"]),
        "critical_10pct": float(critical_values["10%"]),
    }
