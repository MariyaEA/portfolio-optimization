import pandas as pd

from src.risk import calculate_sharpe_ratio, calculate_var, max_drawdown, performance_summary


def test_calculate_var_returns_positive_loss_number():
    returns = pd.Series([-0.05, -0.02, 0.01, 0.03, 0.04])
    var = calculate_var(returns, confidence=0.95)
    assert var > 0


def test_calculate_sharpe_ratio_is_float():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.005])
    sharpe = calculate_sharpe_ratio(returns)
    assert isinstance(float(sharpe), float)


def test_max_drawdown_negative_or_zero():
    returns = pd.Series([0.1, -0.2, 0.05])
    assert max_drawdown(returns) <= 0


def test_performance_summary_contains_required_metrics():
    returns = pd.Series([0.01, -0.01, 0.02, 0.005])
    summary = performance_summary(returns)
    for key in ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown"]:
        assert key in summary
