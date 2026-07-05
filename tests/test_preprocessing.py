import pandas as pd

from src.preprocessing import add_return_features, clean_price_data


def test_clean_price_data_fills_missing_values():
    df = pd.DataFrame(
        {
            "Open": [1, None, 3],
            "High": [2, 3, 4],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1, 2, 3],
            "Adj Close": [1, None, 3],
            "Volume": [100, 200, 300],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    cleaned = clean_price_data(df)
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.index.is_monotonic_increasing


def test_add_return_features_creates_expected_columns():
    df = pd.DataFrame({"Adj Close": [100, 110, 121]}, index=pd.date_range("2020-01-01", periods=3))
    result = add_return_features(df)
    assert "Daily_Return" in result.columns
    assert round(result["Daily_Return"].iloc[1], 2) == 0.10
