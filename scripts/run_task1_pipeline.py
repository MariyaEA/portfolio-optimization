"""Run Task 1: data extraction, cleaning, EDA, stationarity, and risk metrics."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR, FIGURES_DIR, REPORTS_DIR
from src.data_loader import combine_assets, fetch_all_assets, pivot_adjusted_close, save_asset_data
from src.preprocessing import calculate_daily_returns, clean_all_assets, detect_return_outliers, inspect_data_quality
from src.risk import calculate_risk_table
from src.stationarity import run_adf_for_prices_and_returns
from src.visualization import plot_closing_prices, plot_daily_returns, plot_outliers, plot_rolling_statistics


def main() -> None:
    print("Task 1: downloading TSLA, BND, and SPY data...")
    raw_assets = fetch_all_assets()
    cleaned_assets = clean_all_assets(raw_assets)
    save_asset_data(cleaned_assets, DATA_DIR)

    combined = combine_assets(cleaned_assets)
    combined.to_csv(DATA_DIR / "combined_prices_long.csv")

    quality_tables = []
    for ticker, frame in cleaned_assets.items():
        q = inspect_data_quality(frame)
        q.insert(0, "Ticker", ticker)
        quality_tables.append(q.reset_index(names="Column"))
    import pandas as pd
    quality = pd.concat(quality_tables, ignore_index=True)
    quality.to_csv(REPORTS_DIR / "data_quality_summary.csv", index=False)

    prices = pivot_adjusted_close(cleaned_assets)
    returns = calculate_daily_returns(prices)
    returns.to_csv(DATA_DIR / "daily_returns.csv")

    outliers = detect_return_outliers(returns, threshold=3.0)
    outliers.to_csv(REPORTS_DIR / "return_outliers_zscore.csv", index=False)

    adf_results = run_adf_for_prices_and_returns(prices, returns)
    adf_results.to_csv(REPORTS_DIR / "adf_stationarity_results.csv", index=False)

    risk_table = calculate_risk_table(returns)
    risk_table.to_csv(REPORTS_DIR / "risk_metrics.csv")

    plot_closing_prices(prices, FIGURES_DIR / "task1_closing_prices.png")
    plot_daily_returns(returns, FIGURES_DIR / "task1_daily_returns.png")
    plot_rolling_statistics(prices, returns, ticker="TSLA", window=30, output_path=FIGURES_DIR / "task1_tsla_rolling_mean_volatility.png")
    plot_outliers(returns, outliers, ticker="TSLA", output_path=FIGURES_DIR / "task1_tsla_outliers.png")

    print("Task 1 complete. Outputs saved to data/processed, reports, and figures.")


if __name__ == "__main__":
    main()
