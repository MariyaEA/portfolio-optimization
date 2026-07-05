"""Run the interim Week 9 pipeline from the command line."""
from pathlib import Path

import pandas as pd

from src.data_loader import load_local_or_fetch
from src.modeling import chronological_split, fit_arima_forecast, forecast_metrics
from src.preprocessing import add_return_features, clean_price_data, combine_adjusted_close
from src.risk import detect_return_outliers, sharpe_ratio, value_at_risk
from src.stationarity import adf_test
from src.visualization import (
    save_close_price_plot,
    save_daily_return_plot,
    save_forecast_plot,
    save_rolling_plot,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    raw_data = load_local_or_fetch(PROCESSED)
    clean_data = {ticker: add_return_features(clean_price_data(df)) for ticker, df in raw_data.items()}

    prices = combine_adjusted_close(clean_data)
    returns = prices.pct_change().dropna()
    prices.to_csv(PROCESSED / "adj_close_prices.csv")
    returns.to_csv(PROCESSED / "daily_returns.csv")

    save_close_price_plot(prices, FIGURES / "closing_price_over_time.png")
    save_daily_return_plot(returns, FIGURES / "daily_percentage_change.png")
    save_rolling_plot(clean_data["TSLA"], FIGURES / "tsla_rolling_mean_volatility.png")

    stationarity = []
    risk_rows = []
    for ticker in ["TSLA", "BND", "SPY"]:
        stationarity.append(adf_test(clean_data[ticker]["Adj Close"], f"{ticker} adjusted close"))
        stationarity.append(adf_test(clean_data[ticker]["Daily_Return"], f"{ticker} daily return"))
        risk_rows.append(
            {
                "ticker": ticker,
                "VaR_95_daily": value_at_risk(clean_data[ticker]["Daily_Return"], 0.95),
                "Sharpe_Ratio_annualized": sharpe_ratio(clean_data[ticker]["Daily_Return"]),
            }
        )
        detect_return_outliers(clean_data[ticker]["Daily_Return"]).to_csv(PROCESSED / f"{ticker}_return_outliers.csv")

    pd.DataFrame(stationarity).to_csv(REPORTS / "stationarity_results.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(REPORTS / "risk_metrics.csv", index=False)

    train, test = chronological_split(clean_data["TSLA"]["Adj Close"], split_date="2025-01-01")
    fitted, forecast = fit_arima_forecast(train, steps=len(test), order=(1, 1, 1))
    forecast.index = test.index
    pd.DataFrame(forecast_metrics(test, forecast), index=["ARIMA(1,1,1)"]).to_csv(REPORTS / "model_metrics.csv")
    save_forecast_plot(train, test, forecast, FIGURES / "tsla_arima_forecast.png")
    print("Interim pipeline completed successfully.")


if __name__ == "__main__":
    main()
