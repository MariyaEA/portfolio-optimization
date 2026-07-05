"""Plotting helpers for EDA and forecasting."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_close_price_plot(price_df: pd.DataFrame, output_path: str | Path) -> None:
    ax = price_df.plot(figsize=(12, 6), title="Adjusted Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price (USD)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_daily_return_plot(returns_df: pd.DataFrame, output_path: str | Path) -> None:
    ax = returns_df.plot(figsize=(12, 6), title="Daily Percentage Change")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_rolling_plot(df: pd.DataFrame, output_path: str | Path, ticker: str = "TSLA") -> None:
    ax = df[["Adj Close", "Rolling_Mean_30"]].plot(figsize=(12, 6), title=f"{ticker} 30-Day Rolling Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_forecast_plot(train: pd.Series, test: pd.Series, forecast: pd.Series, output_path: str | Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test")
    plt.plot(test.index[: len(forecast)], forecast.values, label="ARIMA Forecast")
    plt.title("TSLA ARIMA Forecast vs Test Period")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
