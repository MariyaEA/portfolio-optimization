"""Cleaning and feature engineering utilities."""
from __future__ import annotations

import pandas as pd


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dtypes, remove duplicates, and handle missing values."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    cleaned = df.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]

    numeric_cols = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in cleaned]
    cleaned[numeric_cols] = cleaned[numeric_cols].apply(pd.to_numeric, errors="coerce")
    cleaned[numeric_cols] = cleaned[numeric_cols].interpolate(method="time").ffill().bfill()
    return cleaned


def add_return_features(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Add daily return, rolling mean, and rolling volatility columns."""
    if price_col not in df.columns:
        raise KeyError(f"{price_col} column is required.")
    out = df.copy()
    out["Daily_Return"] = out[price_col].pct_change()
    out["Rolling_Mean_30"] = out[price_col].rolling(window=30).mean()
    out["Rolling_Std_30"] = out["Daily_Return"].rolling(window=30).std()
    return out


def combine_adjusted_close(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a wide adjusted-close dataframe from a ticker dictionary."""
    return pd.concat({ticker: df["Adj Close"] for ticker, df in data.items()}, axis=1).dropna()
