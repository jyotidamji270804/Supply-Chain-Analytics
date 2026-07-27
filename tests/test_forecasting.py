"""
forecasting.py
----------------
Demand Forecasting module.

  - Chronological train/test split (never shuffle time series data!)
  - Baseline: simple moving average
  - Robust model: ARIMA (statsmodels), with graceful fallback if it fails
    to converge on a particular category
  - Accuracy via MAPE and RMSE
  - evaluate_models(): comparison on held-out historical data
  - forecast_future(): genuine forward-looking forecast (the "next 90 days"
    view the dashboard needs)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.config import (
    TRAIN_TEST_SPLIT_RATIO,
    ARIMA_ORDER,
    ARIMA_ORDER_OVERRIDES,
    FORECAST_HORIZON_DAYS,
)


def get_arima_order(category: str = None):
    """Look up a per-category ARIMA order override, falling back to the default."""
    if category and category in ARIMA_ORDER_OVERRIDES:
        return ARIMA_ORDER_OVERRIDES[category]
    return ARIMA_ORDER


def train_test_split_series(series: pd.Series, ratio: float = TRAIN_TEST_SPLIT_RATIO):
    """Chronological split -- train is always the earlier portion."""
    split_idx = int(len(series) * ratio)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]
    return train, test


def moving_average_forecast(train: pd.Series, steps: int, window: int = 4) -> pd.Series:
    """Baseline forecast: project the last rolling average forward flat."""
    last_avg = train.tail(window).mean()
    freq = pd.infer_freq(train.index) or "W"
    future_index = pd.date_range(train.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.Series([last_avg] * steps, index=future_index)


def arima_forecast(train: pd.Series, steps: int, order=None, category: str = None) -> pd.Series:
    """Fit ARIMA on the training series and forecast `steps` ahead."""
    if order is None:
        order = get_arima_order(category)
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return forecast


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error, ignoring zero-actuals to avoid div/0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_models(series: pd.Series, ratio: float = TRAIN_TEST_SPLIT_RATIO, category: str = None) -> dict:
    """
    Fit both models on the train split, score both against the held-out test
    split, and return a comparison dict -- this is what proves the ARIMA
    model actually beats the naive baseline (or doesn't!).
    """
    train, test = train_test_split_series(series, ratio)
    steps = len(test)

    ma_pred = moving_average_forecast(train, steps)
    ma_pred.index = test.index

    try:
        arima_pred = arima_forecast(train, steps, category=category)
        arima_pred.index = test.index
    except Exception as e:
        arima_pred = ma_pred.copy()  # graceful fallback if ARIMA fails to converge
        print(f"[warning] ARIMA fit failed for category={category!r}, falling back to moving average: {e}")

    return {
        "train": train,
        "test": test,
        "moving_average_forecast": ma_pred,
        "arima_forecast": arima_pred,
        "moving_average_mape": mape(test, ma_pred),
        "moving_average_rmse": rmse(test, ma_pred),
        "arima_mape": mape(test, arima_pred),
        "arima_rmse": rmse(test, arima_pred),
    }


def forecast_future(series: pd.Series, horizon_days: int = FORECAST_HORIZON_DAYS,
                     freq: str = "W", category: str = None) -> pd.Series:
    """
    Produce a genuine future forecast (beyond the available data) using the
    full series as training data -- this powers the "next 90 days" view in
    the Streamlit app.
    """
    steps = max(1, horizon_days // 7) if freq == "W" else horizon_days
    try:
        return arima_forecast(series, steps, category=category)
    except Exception as e:
        print(f"[warning] ARIMA future forecast failed for category={category!r}, using moving average: {e}")
        return moving_average_forecast(series, steps)


if __name__ == "__main__":
    from src.preprocessing import load_raw_data, get_clean_weekly_series
    from src.config import PRODUCT_CATEGORIES

    df = load_raw_data()

    print("=== Smoke test: forecasting across all categories ===\n")
    for category in PRODUCT_CATEGORIES:
        weekly = get_clean_weekly_series(df, category)
        results = evaluate_models(weekly, category=category)
        future = forecast_future(weekly, category=category)
        winner = "ARIMA" if results["arima_mape"] < results["moving_average_mape"] else "Moving Average"
        print(f"{category:<20} MA MAPE={results['moving_average_mape']:.1f}%  "
              f"ARIMA MAPE={results['arima_mape']:.1f}%  -> winner: {winner}")