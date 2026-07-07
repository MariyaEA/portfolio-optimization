"""Plotting utilities for EDA, forecasting, portfolio optimization, and backtesting."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_or_show(fig, output_path: str | Path | None = None):
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_closing_prices(prices: pd.DataFrame, output_path: str | Path | None = None):
    """Plot adjusted closing price trends for all assets."""
    fig, ax = plt.subplots(figsize=(11, 6))
    prices.plot(ax=ax)
    ax.set_title("Adjusted Closing Price Trend: TSLA, BND, and SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, output_path)


def plot_daily_returns(returns: pd.DataFrame, output_path: str | Path | None = None):
    """Plot daily percentage changes for all assets."""
    fig, ax = plt.subplots(figsize=(11, 6))
    (returns * 100).plot(ax=ax, alpha=0.8)
    ax.set_title("Daily Percentage Change / Return Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return (%)")
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, output_path)


def plot_rolling_statistics(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    ticker: str = "TSLA",
    window: int = 30,
    output_path: str | Path | None = None,
):
    """Plot rolling mean and rolling standard deviation for one ticker."""
    if ticker not in prices.columns or ticker not in returns.columns:
        raise ValueError(f"Ticker {ticker} not found in prices/returns.")
    rolling_mean = prices[ticker].rolling(window).mean()
    rolling_vol = returns[ticker].rolling(window).std() * np.sqrt(252)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(prices.index, prices[ticker], label=f"{ticker} Adjusted Close", alpha=0.6)
    ax1.plot(rolling_mean.index, rolling_mean, label=f"{window}-Day Rolling Mean")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(rolling_vol.index, rolling_vol, label=f"{window}-Day Annualized Volatility", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Annualized Volatility")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"{ticker}: Rolling Mean and Rolling Volatility")
    return _save_or_show(fig, output_path)


def plot_outliers(returns: pd.DataFrame, outliers: pd.DataFrame, ticker: str = "TSLA", output_path: str | Path | None = None):
    """Plot daily returns and highlight detected outliers."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(returns.index, returns[ticker] * 100, label=f"{ticker} Daily Return (%)", alpha=0.75)
    if not outliers.empty:
        ticker_outliers = outliers[outliers["Ticker"] == ticker]
        ax.scatter(ticker_outliers["Date"], ticker_outliers["Return"] * 100, label="Outlier Days", marker="o")
    ax.set_title(f"{ticker}: Return Outlier Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_or_show(fig, output_path)


def plot_model_forecasts(test: pd.Series, forecasts: dict[str, pd.Series], output_path: str | Path | None = None):
    """Plot test-period actuals against model forecasts."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(test.index, test.values, label="Actual TSLA Test Prices", linewidth=2)
    for label, forecast in forecasts.items():
        ax.plot(forecast.index, forecast.values, label=label, alpha=0.85)
    ax.set_title("TSLA Test-Period Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_or_show(fig, output_path)


def plot_future_forecast(
    historical: pd.Series,
    test: pd.Series,
    test_forecast: pd.Series,
    future_forecast: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """Plot historical data, test predictions, future forecast, and confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical.index, historical.values, label="Historical TSLA", alpha=0.65)
    ax.plot(test.index, test.values, label="Actual Test", linewidth=2)
    ax.plot(test_forecast.index, test_forecast.values, label="Test Prediction", linestyle="--")
    ax.plot(future_forecast.index, future_forecast["forecast"], label="6-Month Future Forecast", linewidth=2)
    ax.fill_between(
        future_forecast.index,
        future_forecast["lower_ci"].astype(float),
        future_forecast["upper_ci"].astype(float),
        alpha=0.2,
        label="Confidence Interval",
    )
    ax.set_title("TSLA Forecast: Historical, Test Prediction, and Future Confidence Interval")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_or_show(fig, output_path)


def plot_covariance_heatmap(covariance_matrix: pd.DataFrame, output_path: str | Path | None = None):
    """Plot covariance matrix heatmap using matplotlib."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(covariance_matrix.values)
    ax.set_xticks(range(len(covariance_matrix.columns)), covariance_matrix.columns)
    ax.set_yticks(range(len(covariance_matrix.index)), covariance_matrix.index)
    for i in range(len(covariance_matrix.index)):
        for j in range(len(covariance_matrix.columns)):
            ax.text(j, i, f"{covariance_matrix.iloc[i, j]:.4f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title("Annualized Covariance Matrix Heatmap")
    return _save_or_show(fig, output_path)


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    max_sharpe: dict[str, object],
    min_vol: dict[str, object],
    output_path: str | Path | None = None,
):
    """Plot efficient frontier and mark key portfolios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier["volatility"], frontier["target_return"], label="Efficient Frontier")
    ax.scatter(max_sharpe["volatility"], max_sharpe["expected_return"], marker="*", s=180, label="Maximum Sharpe")
    ax.scatter(min_vol["volatility"], min_vol["expected_return"], marker="o", s=100, label="Minimum Volatility")
    ax.set_title("Efficient Frontier with Key Portfolios")
    ax.set_xlabel("Expected Annual Volatility")
    ax.set_ylabel("Expected Annual Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_or_show(fig, output_path)


def plot_backtest_cumulative_returns(cumulative: pd.DataFrame, output_path: str | Path | None = None):
    """Plot cumulative returns for strategy and benchmark."""
    fig, ax = plt.subplots(figsize=(11, 6))
    (cumulative * 100).plot(ax=ax)
    ax.set_title("Backtest Cumulative Returns: Strategy vs 60/40 Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, output_path)
