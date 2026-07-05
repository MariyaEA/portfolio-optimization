"""Initial forecasting model implementation for TSLA."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def chronological_split(series: pd.Series, split_date: str = "2025-01-01") -> tuple[pd.Series, pd.Series]:
    """Split time series into train/test by date, preserving temporal order."""
    clean = series.dropna().sort_index()
    train = clean.loc[clean.index < split_date]
    test = clean.loc[clean.index >= split_date]
    if train.empty or test.empty:
        raise ValueError("Train or test set is empty. Check split date and data range.")
    return train, test


def fit_arima_forecast(
    train: pd.Series,
    steps: int,
    order: tuple[int, int, int] = (1, 1, 1),
) -> tuple[object, pd.Series]:
    """Fit ARIMA and forecast the requested number of future steps."""
    model = ARIMA(train.astype(float), order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return fitted, forecast


def forecast_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return MAE, RMSE and MAPE for aligned forecast values."""
    pred = pd.Series(predicted, index=actual.index[: len(predicted)]).astype(float)
    actual = actual.iloc[: len(pred)].astype(float)
    error = actual - pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    mape = np.mean(np.abs(error / actual)) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}
