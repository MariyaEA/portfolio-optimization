# B9W9: Time Series Forecasting for Portfolio Management Optimization

Interim submission for 10 Academy Week 9. The project analyzes TSLA, BND, and SPY from **2015-01-01 to 2026-06-30** using `yfinance`, performs cleaning and exploratory data analysis, calculates risk metrics, runs stationarity tests, and builds an initial ARIMA forecasting baseline for Tesla.

## Business Context

Guide Me in Finance (GMF) Investments wants to use historical financial data to support portfolio management decisions. Because price prediction is uncertain, the interim work focuses on clean data, volatility/risk understanding, and an interpretable baseline forecasting model before final portfolio optimization and backtesting.

## Assets

| Ticker | Asset | Role in Portfolio |
|---|---|---|
| TSLA | Tesla | High-risk, high-growth equity |
| BND | Vanguard Total Bond Market ETF | Stability and income |
| SPY | S&P 500 ETF | Broad market exposure |

## Repository Structure

```text
portfolio-optimization/
├── .github/workflows/unittests.yml
├── .vscode/settings.json
├── data/processed/
├── figures/
├── notebooks/
│   └── 01_task1_eda_and_task2_initial_forecast.ipynb
├── reports/
├── scripts/run_interim_pipeline.py
├── src/
│   ├── data_loader.py
│   ├── modeling.py
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
git clone <your-repo-url>
cd portfolio-optimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Interim Pipeline

```bash
python scripts/run_interim_pipeline.py
```

The script fetches data with `yfinance`, saves processed CSV outputs locally, generates EDA plots, writes stationarity/risk/model metric CSVs, and fits an initial `ARIMA(1,1,1)` model for TSLA. Generated CSV/PNG outputs are ignored by Git because they are reproducible.

## Interim Submission Coverage

### Task 1: Data Extraction, Cleaning, and EDA

- Downloads TSLA, BND, and SPY using `yfinance` for 2015-01-01 through 2026-06-30.
- Checks data types and missing values.
- Handles missing values using time interpolation plus forward/backward fill.
- Produces basic statistics and daily return features.
- Creates required visualizations:
  - adjusted closing price over time,
  - daily percentage change,
  - 30-day rolling mean and volatility.
- Detects unusually high/low return days using z-scores.
- Applies Augmented Dickey-Fuller tests to adjusted close prices and daily returns.
- Calculates 95% historical Value at Risk and annualized Sharpe Ratio.

### Task 2: Initial Forecasting Model Implementation

- Uses a chronological train/test split: training before 2025-01-01 and testing from 2025 onward.
- Implements an initial ARIMA model using `statsmodels` with documented order `(1,1,1)`.
- Generates forecasts for the full test period.
- Calculates MAE, RMSE, and MAPE.

## Notes on Interpretation

ADF testing is expected to show that raw adjusted close prices are commonly non-stationary, while daily returns are usually stationary. This supports differencing in ARIMA and explains why the initial ARIMA baseline uses `d=1`.

## Tests and CI

Run locally:

```bash
pytest -q
```

GitHub Actions runs the same unit tests on push and pull request through `.github/workflows/unittests.yml`.

## Data Source

Historical OHLCV market data is retrieved through the `yfinance` Python package. The end date in the code is set to `2026-07-01` because Yahoo/yfinance treats the `end` date as exclusive, so the last included trading date is expected to be on or before `2026-06-30` depending on market calendars and data availability.
