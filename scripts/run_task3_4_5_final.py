"""Run Tasks 3-5: future forecasting, portfolio optimization, and backtesting."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.backtesting import backtest_strategy_vs_benchmark, cumulative_returns, isolate_backtest_window
from src.config import DATA_DIR, FIGURES_DIR, REPORTS_DIR, TEST_START_DATE
from src.data_loader import fetch_all_assets, pivot_adjusted_close
from src.forecasting import choose_future_horizon, forecast_future_arima, infer_forecast_return
from src.modeling import fit_arima_forecast
from src.portfolio import (
    annualized_covariance_matrix,
    annualized_expected_returns,
    efficient_frontier,
    optimize_max_sharpe,
    optimize_min_volatility,
    recommendation_table,
)
from src.preprocessing import calculate_daily_returns, clean_all_assets
from src.visualization import (
    plot_backtest_cumulative_returns,
    plot_covariance_heatmap,
    plot_efficient_frontier,
    plot_future_forecast,
)


def load_or_fetch_prices() -> pd.DataFrame:
    prices_path = DATA_DIR / "adjusted_close_prices.csv"
    if prices_path.exists():
        return pd.read_csv(prices_path, index_col="Date", parse_dates=True)
    assets = clean_all_assets(fetch_all_assets())
    prices = pivot_adjusted_close(assets)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(prices_path)
    return prices


def main() -> None:
    prices = load_or_fetch_prices()
    returns = calculate_daily_returns(prices)
    returns.to_csv(DATA_DIR / "daily_returns.csv")

    tsla = prices["TSLA"].dropna()
    arima_results, train, test, test_forecast, metrics = fit_arima_forecast(
        tsla,
        split_date=TEST_START_DATE,
        order=(1, 1, 1),
    )

    horizon_steps = choose_future_horizon(months=6)
    future_forecast = forecast_future_arima(arima_results, steps=horizon_steps)
    future_forecast.to_csv(REPORTS_DIR / "task3_future_forecast_ci.csv")

    plot_future_forecast(
        historical=train,
        test=test,
        test_forecast=test_forecast,
        future_forecast=future_forecast,
        output_path=FIGURES_DIR / "task3_future_forecast_with_ci.png",
    )

    tsla_expected_return = infer_forecast_return(
        current_price=float(tsla.iloc[-1]),
        forecast_price=float(future_forecast["forecast"].iloc[-1]),
        forecast_days=horizon_steps,
    )
    expected_returns = annualized_expected_returns(returns, tsla_forecast_return=tsla_expected_return)
    covariance = annualized_covariance_matrix(returns)
    expected_returns.to_csv(REPORTS_DIR / "task4_expected_returns.csv")
    covariance.to_csv(REPORTS_DIR / "task4_covariance_matrix.csv")

    frontier = efficient_frontier(expected_returns, covariance, points=60)
    max_sharpe = optimize_max_sharpe(expected_returns, covariance)
    min_vol = optimize_min_volatility(expected_returns, covariance)
    recommendation = max_sharpe

    frontier.to_csv(REPORTS_DIR / "task4_efficient_frontier.csv", index=False)
    recommendation_table(max_sharpe).to_csv(REPORTS_DIR / "task4_max_sharpe_portfolio.csv", index=False)
    recommendation_table(min_vol).to_csv(REPORTS_DIR / "task4_min_volatility_portfolio.csv", index=False)

    plot_covariance_heatmap(covariance, FIGURES_DIR / "task4_covariance_heatmap.png")
    plot_efficient_frontier(frontier, max_sharpe, min_vol, FIGURES_DIR / "task4_efficient_frontier.png")

    backtest_returns = isolate_backtest_window(returns)
    comparison_returns, backtest_metrics = backtest_strategy_vs_benchmark(backtest_returns, recommendation["weights"])
    comparison_returns.to_csv(REPORTS_DIR / "task5_backtest_daily_returns.csv")
    backtest_metrics.to_csv(REPORTS_DIR / "task5_backtest_performance_metrics.csv")
    cumulative = cumulative_returns(comparison_returns)
    cumulative.to_csv(REPORTS_DIR / "task5_backtest_cumulative_returns.csv")
    plot_backtest_cumulative_returns(cumulative, FIGURES_DIR / "task5_strategy_vs_benchmark.png")

    print("Tasks 3-5 complete.")
    print("Recommended portfolio:")
    print(recommendation_table(max_sharpe))
    print("Backtest metrics:")
    print(backtest_metrics)


if __name__ == "__main__":
    main()
