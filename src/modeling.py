"""Forecasting model utilities for ARIMA/SARIMA and LSTM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.preprocessing import chronological_split


@dataclass(frozen=True)
class ForecastMetrics:
    """Forecast evaluation metrics."""

    model: str
    mae: float
    rmse: float
    mape: float

    def as_dict(self) -> dict[str, float | str]:
        return {"model": self.model, "mae": self.mae, "rmse": self.rmse, "mape": self.mape}


def calculate_forecast_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, model_name: str) -> ForecastMetrics:
    """Calculate MAE, RMSE, and MAPE."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have equal length.")
    mae = mean_absolute_error(true, pred)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    denominator = np.where(true == 0, np.nan, true)
    mape = float(np.nanmean(np.abs((true - pred) / denominator)) * 100)
    return ForecastMetrics(model=model_name, mae=float(mae), rmse=rmse, mape=mape)


def fit_arima_forecast(
    series: pd.Series,
    split_date: str,
    order: tuple[int, int, int] = (1, 1, 1),
) -> tuple[Any, pd.Series, pd.Series, pd.Series, ForecastMetrics]:
    """Fit ARIMA on training data and forecast the test period."""
    train, test = chronological_split(series.dropna(), split_date)
    try:
        model = ARIMA(train, order=order)
        results = model.fit()
    except Exception as exc:
        raise RuntimeError(f"ARIMA{order} failed to fit: {exc}") from exc
    forecast_values = results.forecast(steps=len(test))
    forecast = pd.Series(forecast_values.to_numpy(), index=test.index, name="ARIMA_Forecast")
    metrics = calculate_forecast_metrics(test, forecast, f"ARIMA{order}")
    return results, train, test, forecast, metrics


def fit_sarima_forecast(
    series: pd.Series,
    split_date: str,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 5),
) -> tuple[Any, pd.Series, pd.Series, pd.Series, ForecastMetrics]:
    """Fit SARIMA on training data and forecast the test period.

    For daily trading data, m=5 can represent a weekly trading-day seasonal pattern.
    """
    train, test = chronological_split(series.dropna(), split_date)
    try:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)
    except Exception as exc:
        raise RuntimeError(f"SARIMA{order}x{seasonal_order} failed to fit: {exc}") from exc
    forecast_values = results.forecast(steps=len(test))
    forecast = pd.Series(forecast_values.to_numpy(), index=test.index, name="SARIMA_Forecast")
    metrics = calculate_forecast_metrics(test, forecast, f"SARIMA{order}x{seasonal_order}")
    return results, train, test, forecast, metrics


def create_lstm_sequences(values: np.ndarray, window_size: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1D series into windowed LSTM sequences."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if len(values) <= window_size:
        raise ValueError("Not enough observations for the requested LSTM window_size.")
    x, y = [], []
    for idx in range(window_size, len(values)):
        x.append(values[idx - window_size : idx])
        y.append(values[idx])
    return np.asarray(x), np.asarray(y)


def build_lstm_model(window_size: int = 60, units: int = 64, learning_rate: float = 0.001):
    """Build an LSTM model with input, LSTM, dropout, and dense output layers."""
    try:
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:  # pragma: no cover - depends on optional TensorFlow install
        raise ImportError("TensorFlow is required for LSTM modeling. Install requirements.txt.") from exc

    model = Sequential(
        [
            Input(shape=(window_size, 1)),
            LSTM(units, return_sequences=True),
            Dropout(0.2),
            LSTM(units // 2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def fit_lstm_forecast(
    series: pd.Series,
    split_date: str,
    window_size: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
) -> tuple[Any, pd.Series, pd.Series, pd.Series, ForecastMetrics]:
    """Train an LSTM on windowed sequence data and forecast the test period."""
    clean_series = series.dropna().sort_index()
    train, test = chronological_split(clean_series, split_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    combined_for_test = pd.concat([train.iloc[-window_size:], test])
    combined_scaled = scaler.transform(combined_for_test.values.reshape(-1, 1))

    x_train, y_train = create_lstm_sequences(train_scaled, window_size=window_size)
    x_test, _ = create_lstm_sequences(combined_scaled, window_size=window_size)

    model = build_lstm_model(window_size=window_size)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    pred_scaled = model.predict(x_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).ravel()
    forecast = pd.Series(pred, index=test.index, name="LSTM_Forecast")
    metrics = calculate_forecast_metrics(test, forecast, f"LSTM_window_{window_size}")
    return model, train, test, forecast, metrics


def select_best_model(metrics_table: pd.DataFrame, metric: str = "rmse") -> str:
    """Select the best model name by lowest error metric."""
    if metric not in metrics_table.columns:
        raise ValueError(f"Metric '{metric}' not found in metrics table.")
    if metrics_table.empty:
        raise ValueError("metrics_table cannot be empty.")
    return str(metrics_table.sort_values(metric).iloc[0]["model"])
