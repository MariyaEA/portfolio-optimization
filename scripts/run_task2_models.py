"""Run Task 2: ARIMA/SARIMA and LSTM forecasting models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR, FIGURES_DIR, REPORTS_DIR, TEST_START_DATE
from src.data_loader import fetch_all_assets, pivot_adjusted_close
from src.modeling import fit_arima_forecast, fit_lstm_forecast, fit_sarima_forecast
from src.preprocessing import clean_all_assets
from src.visualization import plot_model_forecasts


def load_or_fetch_prices() -> pd.DataFrame:
    prices_path = DATA_DIR / "adjusted_close_prices.csv"
    if prices_path.exists():
        return pd.read_csv(prices_path, index_col="Date", parse_dates=True)
    assets = clean_all_assets(fetch_all_assets())
    prices = pivot_adjusted_close(assets)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(prices_path)
    return prices


def main(run_lstm: bool = True, lstm_epochs: int = 10) -> None:
    prices = load_or_fetch_prices()
    tsla = prices["TSLA"].dropna()

    arima_results, train, test, arima_forecast, arima_metrics = fit_arima_forecast(
        tsla,
        split_date=TEST_START_DATE,
        order=(1, 1, 1),
    )
    sarima_results, _, _, sarima_forecast, sarima_metrics = fit_sarima_forecast(
        tsla,
        split_date=TEST_START_DATE,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 5),
    )

    forecast_dict = {
        "ARIMA(1,1,1)": arima_forecast,
        "SARIMA(1,1,1)x(1,0,1,5)": sarima_forecast,
    }
    metrics = [arima_metrics.as_dict(), sarima_metrics.as_dict()]

    if run_lstm:
        lstm_model, _, _, lstm_forecast, lstm_metrics = fit_lstm_forecast(
            tsla,
            split_date=TEST_START_DATE,
            window_size=60,
            epochs=lstm_epochs,
            batch_size=32,
        )
        forecast_dict["LSTM window=60"] = lstm_forecast
        metrics.append(lstm_metrics.as_dict())

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(REPORTS_DIR / "task2_model_comparison.csv", index=False)

    task2_forecasts = pd.concat(forecast_dict, axis=1)
    task2_forecasts.to_csv(REPORTS_DIR / "task2_test_forecasts.csv")

    plot_model_forecasts(test, forecast_dict, FIGURES_DIR / "task2_forecast_comparison.png")
    print("Task 2 complete. Metrics:")
    print(metrics_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM if TensorFlow is not available.")
    parser.add_argument("--lstm-epochs", type=int, default=10, help="Number of LSTM training epochs.")
    args = parser.parse_args()
    main(run_lstm=not args.no_lstm, lstm_epochs=args.lstm_epochs)
