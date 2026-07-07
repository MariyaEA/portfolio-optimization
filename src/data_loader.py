"""Data extraction utilities for financial time series data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf

from src.config import END_DATE, START_DATE, TICKERS


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for Yahoo Finance downloads."""

    tickers: tuple[str, ...] = tuple(TICKERS)
    start: str = START_DATE
    end: str = END_DATE
    auto_adjust: bool = False


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def fetch_asset_data(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Fetch OHLCV data for one ticker from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Asset ticker, e.g. TSLA.
    start, end:
        Date range. The Yahoo Finance end date is exclusive.

    Returns
    -------
    pd.DataFrame
        DataFrame with Date index and OHLCV columns plus a ticker column.

    Raises
    ------
    RuntimeError
        If the API call fails or returns no data.
    ValueError
        If required columns are missing.
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as exc:  # pragma: no cover - depends on external API/network
        raise RuntimeError(f"Failed to download data for {ticker}: {exc}") from exc

    if data.empty:
        raise RuntimeError(f"No data returned for ticker {ticker}. Check ticker or date range.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data for {ticker} is missing columns: {missing}")

    data = data[REQUIRED_COLUMNS].copy()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    data["Ticker"] = ticker
    return data


def fetch_all_assets(config: DownloadConfig | None = None) -> dict[str, pd.DataFrame]:
    """Fetch all configured assets and return a dictionary keyed by ticker."""
    config = config or DownloadConfig()
    output: dict[str, pd.DataFrame] = {}
    for ticker in config.tickers:
        output[ticker] = fetch_asset_data(ticker=ticker, start=config.start, end=config.end)
    return output


def combine_assets(asset_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine individual ticker frames into a single long-format DataFrame."""
    if not asset_frames:
        raise ValueError("asset_frames cannot be empty.")
    combined = pd.concat(asset_frames.values(), axis=0).sort_index()
    combined = combined.reset_index().set_index(["Date", "Ticker"]).sort_index()
    return combined


def pivot_adjusted_close(asset_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a wide adjusted-close price table with dates as rows and tickers as columns."""
    if not asset_frames:
        raise ValueError("asset_frames cannot be empty.")
    adjusted = pd.concat({ticker: frame["Adj Close"] for ticker, frame in asset_frames.items()}, axis=1)
    adjusted.index = pd.to_datetime(adjusted.index)
    adjusted.index.name = "Date"
    return adjusted.sort_index()


def save_asset_data(asset_frames: dict[str, pd.DataFrame], output_dir) -> None:
    """Persist asset data to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ticker, frame in asset_frames.items():
        frame.to_csv(output_dir / f"{ticker.lower()}_prices.csv")
    pivot_adjusted_close(asset_frames).to_csv(output_dir / "adjusted_close_prices.csv")


def validate_tickers(tickers: Iterable[str]) -> list[str]:
    """Normalize and validate ticker input."""
    normalized = [ticker.strip().upper() for ticker in tickers]
    if not normalized or any(not ticker for ticker in normalized):
        raise ValueError("At least one non-empty ticker is required.")
    return normalized
