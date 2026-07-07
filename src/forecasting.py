"""Future forecasting utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def forecast_future_arima(results, steps: int = 126) -> pd.DataFrame:
    """Generate future ARIMA/SARIMA forecast with confidence intervals."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    forecast_result = results.get_forecast(steps=steps)
    mean_forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Use business days for visualization because market data is trading-day based.
    last_date = pd.to_datetime(results.data.dates[-1]) if results.data.dates is not None else pd.Timestamp.today()
    future_index = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=steps)

    output = pd.DataFrame(index=future_index)
    output["forecast"] = np.asarray(mean_forecast)
    output["lower_ci"] = np.asarray(conf_int.iloc[:, 0])
    output["upper_ci"] = np.asarray(conf_int.iloc[:, 1])
    output.index.name = "Date"
    return output


def infer_forecast_return(current_price: float, forecast_price: float, forecast_days: int) -> float:
    """Annualize expected return implied by current and forecast price."""
    if current_price <= 0 or forecast_price <= 0:
        raise ValueError("Prices must be positive.")
    period_return = forecast_price / current_price - 1
    return (1 + period_return) ** (TRADING_DAYS / forecast_days) - 1


def choose_future_horizon(months: int = 6) -> int:
    """Translate month horizon to approximate trading-day steps."""
    if months < 1 or months > 12:
        raise ValueError("Final task expects a 6-12 month horizon; choose a value between 1 and 12 if experimenting.")
    return int(round(TRADING_DAYS * months / 12))
