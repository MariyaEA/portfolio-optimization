"""Project constants and folder paths."""
from pathlib import Path

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
# yfinance end date is exclusive, so 2026-07-01 includes 2026-06-30.
END_DATE = "2026-07-01"
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
BACKTEST_START_DATE = "2025-07-01"
BACKTEST_END_DATE = "2026-06-30"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

for path in [DATA_DIR, FIGURES_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
