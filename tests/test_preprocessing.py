import numpy as np
import pandas as pd

from src.preprocessing import calculate_daily_returns, chronological_split, clean_price_data, detect_return_outliers


def test_clean_price_data_fills_missing_values():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [1, np.nan, 3, 4, 5],
            "High": [1, 2, np.nan, 4, 5],
            "Low": [1, 2, 3, np.nan, 5],
            "Close": [1, 2, 3, 4, np.nan],
            "Adj Close": [1, 2, np.nan, 4, 5],
            "Volume": [100, np.nan, 120, 130, 140],
        },
        index=dates,
    )
    cleaned = clean_price_data(frame)
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.index.is_monotonic_increasing


def test_calculate_daily_returns():
    prices = pd.DataFrame({"TSLA": [100, 110, 99]}, index=pd.date_range("2024-01-01", periods=3))
    returns = calculate_daily_returns(prices)
    assert round(float(returns.iloc[0, 0]), 4) == 0.1
    assert round(float(returns.iloc[1, 0]), 4) == -0.1


def test_detect_return_outliers():
    dates = pd.date_range("2024-01-01", periods=10)
    returns = pd.DataFrame({"TSLA": [0.01] * 9 + [0.5]}, index=dates)
    outliers = detect_return_outliers(returns, threshold=2.0)
    assert not outliers.empty
    assert outliers.iloc[0]["Ticker"] == "TSLA"


def test_chronological_split():
    series = pd.Series(range(10), index=pd.date_range("2024-01-01", periods=10))
    train, test = chronological_split(series, "2024-01-06")
    assert train.index.max() < pd.Timestamp("2024-01-06")
    assert test.index.min() >= pd.Timestamp("2024-01-06")
