import numpy as np
import pandas as pd

from src.modeling import calculate_forecast_metrics, create_lstm_sequences, select_best_model


def test_create_lstm_sequences_shapes():
    values = np.arange(100)
    x, y = create_lstm_sequences(values, window_size=10)
    assert x.shape == (90, 10, 1)
    assert y.shape == (90, 1)


def test_calculate_forecast_metrics():
    true = pd.Series([100, 110, 120])
    pred = pd.Series([101, 108, 119])
    metrics = calculate_forecast_metrics(true, pred, "test_model")
    assert metrics.model == "test_model"
    assert metrics.mae > 0
    assert metrics.rmse > 0
    assert metrics.mape > 0


def test_select_best_model_lowest_rmse():
    table = pd.DataFrame({"model": ["a", "b"], "rmse": [2.0, 1.0]})
    assert select_best_model(table, "rmse") == "b"
