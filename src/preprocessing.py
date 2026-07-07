"""Cleaning and preprocessing utilities."""
from __future__ import annotations

import pandas as pd


def inspect_data_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data quality summary for a price DataFrame."""
    if frame.empty:
        raise ValueError("Cannot inspect an empty DataFrame.")
    summary = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_count": frame.isna().sum(),
            "missing_pct": frame.isna().mean() * 100,
        }
    )
    return summary


def clean_price_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean one asset OHLCV frame.

    Missing price values are handled using time interpolation followed by forward/backward fill.
    Volume is forward-filled/back-filled because interpolation may create unrealistic fractional volume.
    """
    if frame.empty:
        raise ValueError("Cannot clean an empty DataFrame.")

    cleaned = frame.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()

    numeric_cols = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in cleaned]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    price_cols = [col for col in ["Open", "High", "Low", "Close", "Adj Close"] if col in cleaned]
    if price_cols:
        cleaned[price_cols] = cleaned[price_cols].interpolate(method="time").ffill().bfill()
    if "Volume" in cleaned:
        cleaned["Volume"] = cleaned["Volume"].ffill().bfill()

    if cleaned[numeric_cols].isna().any().any():
        raise ValueError("Missing values remain after cleaning. Inspect input data.")
    return cleaned


def clean_all_assets(asset_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean all ticker frames."""
    if not asset_frames:
        raise ValueError("asset_frames cannot be empty.")
    return {ticker: clean_price_data(frame) for ticker, frame in asset_frames.items()}


def calculate_daily_returns(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily percentage returns from a wide adjusted-close price table."""
    if price_frame.empty:
        raise ValueError("Cannot calculate returns on an empty DataFrame.")
    returns = price_frame.sort_index().pct_change().dropna(how="all")
    returns.index.name = "Date"
    return returns


def add_return_features(frame: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Add daily return, rolling mean, and rolling volatility features for one ticker."""
    if "Adj Close" not in frame.columns:
        raise ValueError("Input frame must contain 'Adj Close'.")
    enriched = frame.copy().sort_index()
    enriched["Daily_Return"] = enriched["Adj Close"].pct_change()
    enriched[f"Rolling_Mean_{window}"] = enriched["Adj Close"].rolling(window=window).mean()
    enriched[f"Rolling_Volatility_{window}"] = enriched["Daily_Return"].rolling(window=window).std()
    return enriched


def detect_return_outliers(returns: pd.DataFrame | pd.Series, threshold: float = 3.0) -> pd.DataFrame:
    """Identify unusually high or low daily returns using absolute z-scores.

    Returns a long-format DataFrame with Date, Ticker, Return, and ZScore.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(name=returns.name or "Return")
    if returns.empty:
        raise ValueError("returns cannot be empty.")

    zscores = (returns - returns.mean()) / returns.std(ddof=0)
    mask = zscores.abs() >= threshold
    rows = []
    for ticker in returns.columns:
        selected = returns.loc[mask[ticker].fillna(False), ticker]
        for date, value in selected.items():
            rows.append(
                {
                    "Date": pd.to_datetime(date),
                    "Ticker": ticker,
                    "Return": float(value),
                    "ZScore": float(zscores.loc[date, ticker]),
                }
            )
    return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def chronological_split(series: pd.Series, split_date: str) -> tuple[pd.Series, pd.Series]:
    """Split a time series chronologically around a split date.

    Training data includes observations strictly before split_date; test data includes observations
    on or after split_date.
    """
    if series.empty:
        raise ValueError("series cannot be empty.")
    series = series.sort_index()
    split_ts = pd.Timestamp(split_date)
    train = series.loc[series.index < split_ts]
    test = series.loc[series.index >= split_ts]
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    return train, test
