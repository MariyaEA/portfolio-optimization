import pandas as pd

from src.backtesting import backtest_strategy_vs_benchmark, benchmark_60_40, cumulative_returns, isolate_backtest_window


def test_backtest_strategy_vs_benchmark_outputs_metrics():
    dates = pd.date_range("2025-07-01", periods=10, freq="B")
    returns = pd.DataFrame(
        {
            "TSLA": [0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.01, -0.005, 0.004],
            "BND": [0.001] * 10,
            "SPY": [0.004, -0.003, 0.006, 0.002, -0.001, 0.003, 0.002, -0.002, 0.001, 0.002],
        },
        index=dates,
    )
    window = isolate_backtest_window(returns, "2025-07-01", "2025-07-31")
    comparison, metrics = backtest_strategy_vs_benchmark(window, {"TSLA": 0.3, "BND": 0.4, "SPY": 0.3})
    assert set(comparison.columns) == {"strategy", "benchmark_60_40"}
    assert "total_return" in metrics.columns


def test_benchmark_and_cumulative_returns():
    dates = pd.date_range("2025-07-01", periods=3, freq="B")
    returns = pd.DataFrame({"BND": [0.001, 0.001, 0.001], "SPY": [0.01, -0.01, 0.02]}, index=dates)
    benchmark = benchmark_60_40(returns)
    cumulative = cumulative_returns(benchmark)
    assert len(benchmark) == 3
    assert len(cumulative) == 3
