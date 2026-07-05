"""Data extraction utilities for Week 9 portfolio forecasting project."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-07-01"  # yfinance end is exclusive; includes 2026-06-30 when available


def fetch_asset_data(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV financial data from yfinance.

    Parameters
    ----------
    tickers:
        Asset tickers to download.
    start, end:
        Date range. The yfinance ``end`` argument is exclusive.
    output_dir:
        Optional folder where each ticker is saved as CSV.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary keyed by ticker with cleaned Date index.
    """
    results: dict[str, pd.DataFrame] = {}
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Failed to fetch data for {ticker}: {exc}") from exc

        if df.empty:
            raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {ticker}: {missing}")

        df = df[required].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df["Ticker"] = ticker
        results[ticker] = df

        if output_path:
            df.to_csv(output_path / f"{ticker}.csv")

    return results


def load_local_or_fetch(data_dir: str | Path = "data/processed") -> dict[str, pd.DataFrame]:
    """Load cached CSVs if present; otherwise fetch from yfinance."""
    data_path = Path(data_dir)
    local_files = {ticker: data_path / f"{ticker}.csv" for ticker in DEFAULT_TICKERS}
    if all(path.exists() for path in local_files.values()):
        return {
            ticker: pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            for ticker, path in local_files.items()
        }
    return fetch_asset_data(output_dir=data_path)
