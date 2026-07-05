import pandas as pd

from src.risk import detect_return_outliers, sharpe_ratio, value_at_risk


def test_value_at_risk_returns_positive_loss():
    returns = pd.Series([-0.10, -0.02, 0.01, 0.03, 0.04])
    assert value_at_risk(returns, 0.95) > 0


def test_sharpe_ratio_numeric():
    returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.003])
    assert isinstance(sharpe_ratio(returns), float)


def test_detect_return_outliers():
    returns = pd.Series([0.01] * 20 + [0.50])
    outliers = detect_return_outliers(returns, z_threshold=2.0)
    assert not outliers.empty
