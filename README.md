# B9W9: Time Series Forecasting for Portfolio Management Optimization

Final GitHub submission for 10 Academy Week 9. The project supports **Guide Me in Finance (GMF) Investments**, a personalized portfolio advisory firm, by using time series forecasting and portfolio analytics to guide asset allocation decisions across **TSLA**, **BND**, and **SPY**.

The work follows the business reality that exact price prediction is difficult under the Efficient Market Hypothesis. Therefore, the models are treated as decision-support tools for understanding momentum, volatility, risk, and portfolio trade-offs rather than as standalone guarantees of future prices.

## Assets

| Ticker | Asset | Portfolio Role |
|---|---|---|
| TSLA | Tesla Inc. | High-risk, high-growth equity exposure |
| BND | Vanguard Total Bond Market ETF | Stability and income exposure |
| SPY | SPDR S&P 500 ETF Trust | Diversified U.S. equity market exposure |

## Final Task Coverage

### Task 1: Data Pipeline, Cleaning, EDA, and Risk Metrics

Implemented in `src/data_loader.py`, `src/preprocessing.py`, `src/stationarity.py`, `src/risk.py`, `src/visualization.py`, and `scripts/run_task1_pipeline.py`.

- Data extraction for TSLA, BND, and SPY using `yfinance`.
- Date coverage: **2015-01-01 to 2026-06-30**. The code uses `END_DATE = "2026-07-01"` because Yahoo Finance treats end dates as exclusive.
- Missing value and data type checks.
- Cleaning by time interpolation, forward fill, and backward fill.
- EDA visualizations:
  - adjusted closing price trend,
  - daily percentage change,
  - rolling mean and rolling volatility,
  - return outlier detection.
- Augmented Dickey-Fuller stationarity tests for adjusted closing prices and daily returns.
- Value at Risk and annualized Sharpe Ratio calculations.

### Task 2: Forecasting Models

Implemented in `src/modeling.py` and `scripts/run_task2_models.py`.

- Chronological train/test split to preserve temporal order.
- ARIMA model: documented default order **ARIMA(1,1,1)**.
- SARIMA model: documented default order **SARIMA(1,1,1)x(1,0,1,5)**, where `m=5` represents a weekly trading-day pattern.
- LSTM model:
  - windowed sequence data with a default 60-day lookback,
  - two LSTM layers,
  - dropout,
  - dense output layers,
  - trained on scaled TSLA prices.
- Model comparison output includes MAE, RMSE, and MAPE.

### Task 3: Future Market Forecasting

Implemented in `src/forecasting.py` and `scripts/run_task3_4_5_final.py`.

- Generates a 6-month future forecast using the selected classical baseline.
- Produces forecast visualization showing:
  - historical TSLA prices,
  - test actuals,
  - test predictions,
  - future forecast,
  - confidence interval bounds.

### Task 4: Portfolio Optimization

Implemented in `src/portfolio.py` and `scripts/run_task3_4_5_final.py`.

- Expected returns preparation:
  - TSLA uses forecast-implied annualized return,
  - BND and SPY use historical annualized average returns.
- Annualized covariance matrix from daily returns.
- Covariance heatmap visualization.
- Efficient Frontier generated with `scipy.optimize`.
- Maximum Sharpe Ratio Portfolio and Minimum Volatility Portfolio identified and marked.
- Final recommended weights and expected annual return, volatility, and Sharpe Ratio exported to CSV.

### Task 5: Strategy Backtesting

Implemented in `src/backtesting.py` and `scripts/run_task3_4_5_final.py`.

- Uses the final year of available data as a held-out backtesting window.
- Defines the benchmark portfolio as **60% SPY / 40% BND**.
- Simulates the optimized strategy as a static hold portfolio.
- Plots cumulative returns for strategy vs benchmark.
- Calculates total return, annualized return, annualized volatility, Sharpe Ratio, and maximum drawdown.

## Repository Structure

```text
portfolio-optimization/
├── .github/workflows/unittests.yml
├── .vscode/settings.json
├── data/processed/.gitkeep
├── figures/.gitkeep
├── notebooks/
│   ├── 01_task1_eda.ipynb
│   ├── 02_task2_forecasting_models.ipynb
│   └── 03_task3_4_5_forecast_portfolio_backtest.ipynb
├── reports/.gitkeep
├── scripts/
│   ├── run_all.py
│   ├── run_task1_pipeline.py
│   ├── run_task2_models.py
│   └── run_task3_4_5_final.py
├── src/
│   ├── backtesting.py
│   ├── config.py
│   ├── data_loader.py
│   ├── forecasting.py
│   ├── modeling.py
│   ├── portfolio.py
│   ├── preprocessing.py
│   ├── risk.py
│   ├── stationarity.py
│   └── visualization.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Tests

```bash
pytest -q
```

## Run the Full Final Workflow

```bash
python scripts/run_all.py
```

If TensorFlow installation is heavy or unavailable on your machine, run the classical pipeline first:

```bash
python scripts/run_all.py --no-lstm
```

Then install TensorFlow and run Task 2 with LSTM:

```bash
pip install tensorflow
python scripts/run_task2_models.py --lstm-epochs 10
```

## Generated Outputs

Running the scripts creates reproducible local outputs:

- `data/processed/adjusted_close_prices.csv`
- `data/processed/daily_returns.csv`
- `reports/adf_stationarity_results.csv`
- `reports/risk_metrics.csv`
- `reports/task2_model_comparison.csv`
- `reports/task3_future_forecast_ci.csv`
- `reports/task4_expected_returns.csv`
- `reports/task4_covariance_matrix.csv`
- `reports/task4_efficient_frontier.csv`
- `reports/task4_max_sharpe_portfolio.csv`
- `reports/task5_backtest_performance_metrics.csv`
- `figures/task1_closing_prices.png`
- `figures/task2_forecast_comparison.png`
- `figures/task3_future_forecast_with_ci.png`
- `figures/task4_efficient_frontier.png`
- `figures/task5_strategy_vs_benchmark.png`

Generated CSV and PNG outputs are ignored by Git because they are reproducible from source code.

## Git/GitHub Workflow Used

Recommended branch workflow for the final submission:

```bash
git checkout main
git pull origin main

git checkout -b task-2-forecasting-models
git add .
git commit -m "Complete ARIMA SARIMA and LSTM forecasting models"
git push -u origin task-2-forecasting-models
# Open PR to main and merge

git checkout main
git pull origin main
git checkout -b task-3-4-5-optimization-backtesting
git add .
git commit -m "Complete future forecasting portfolio optimization and backtesting"
git push -u origin task-3-4-5-optimization-backtesting
# Open PR to main and merge
```

For a faster final submission, one branch can be used:

```bash
git checkout main
git pull origin main
git checkout -b final-submission
git add .
git commit -m "Complete final portfolio optimization workflow"
git push -u origin final-submission
```

Then create a Pull Request from `final-submission` into `main`, merge it, and submit the final GitHub repository link.

## Notes on Modeling Decisions

- ARIMA/SARIMA are used as interpretable baselines that align well with technical and business review.
- LSTM is implemented to capture non-linear sequence patterns, but its performance should be compared carefully against simpler baselines.
- Portfolio optimization uses long-only weights constrained to sum to 1.
- Backtesting is a reality check, not a guarantee of future profitability. The final-year backtest compares the model-driven strategy against a simple 60/40 benchmark.
