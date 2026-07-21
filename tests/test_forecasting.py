"""test_forecasting.py
---
Unit tests for src/forecasting.py.

Checks:
  - train_test_split_series never overlaps and stays chronological
  - moving_average_forecast produces the right number of future steps
  - arima_forecast produces the right number of future steps
  - mape/rmse behave correctly on known inputs (including edge cases)
  - evaluate_models returns a complete, well-formed result dict
  - forecast_future produces genuinely future dates beyond the input series"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.forecasting import (
    train_test_split_series,
    moving_average_forecast,
    arima_forecast,
    mape,
    rmse,
    evaluate_models,
    forecast_future,
)


@pytest.fixture
def sample_series():
    """A simple, well-behaved weekly series for fast, deterministic tests."""
    dates = pd.date_range("2022-01-01", periods=104, freq="W")
    values = 100 + np.arange(104) * 0.5 + 10 * np.sin(np.arange(104) * 2 * np.pi / 52)
    return pd.Series(values, index=dates)


def test_train_test_split_is_chronological_and_non_overlapping(sample_series):
    train, test = train_test_split_series(sample_series, ratio=0.8)
    assert train.index.max() < test.index.min(), "Train must end before test begins"
    assert len(train) + len(test) == len(sample_series)


def test_train_test_split_ratio_respected(sample_series):
    train, test = train_test_split_series(sample_series, ratio=0.85)
    expected_train_len = int(len(sample_series) * 0.85)
    assert len(train) == expected_train_len


def test_moving_average_forecast_length(sample_series):
    train, test = train_test_split_series(sample_series)
    forecast = moving_average_forecast(train, steps=len(test))
    assert len(forecast) == len(test)


def test_moving_average_forecast_is_flat(sample_series):
    """Baseline should predict the same value for every future step."""
    train, _ = train_test_split_series(sample_series)
    forecast = moving_average_forecast(train, steps=5)
    assert forecast.nunique() == 1, "Moving average baseline should be a flat line"


def test_arima_forecast_length(sample_series):
    train, test = train_test_split_series(sample_series)
    forecast = arima_forecast(train, steps=len(test))
    assert len(forecast) == len(test)


def test_mape_zero_when_perfect_prediction():
    actual = pd.Series([100, 200, 300])
    predicted = pd.Series([100, 200, 300])
    assert mape(actual, predicted) == pytest.approx(0.0)


def test_mape_ignores_zero_actuals_without_crashing():
    actual = pd.Series([0, 100, 200])
    predicted = pd.Series([5, 110, 190])
    result = mape(actual, predicted)
    assert not np.isnan(result), "MAPE should skip zero-actuals, not return NaN"


def test_rmse_zero_when_perfect_prediction():
    actual = pd.Series([50, 60, 70])
    predicted = pd.Series([50, 60, 70])
    assert rmse(actual, predicted) == pytest.approx(0.0)


def test_rmse_known_value():
    actual = pd.Series([0, 0, 0])
    predicted = pd.Series([3, 4, 0])
    # sqrt(mean([9, 16, 0])) = sqrt(8.33) ≈ 2.887
    assert rmse(actual, predicted) == pytest.approx(2.887, rel=1e-2)


def test_evaluate_models_returns_expected_keys(sample_series):
    results = evaluate_models(sample_series)
    expected_keys = {
        "train", "test", "moving_average_forecast", "arima_forecast",
        "moving_average_mape", "moving_average_rmse", "arima_mape", "arima_rmse",
    }
    assert expected_keys.issubset(results.keys())


def test_evaluate_models_metrics_are_non_negative(sample_series):
    results = evaluate_models(sample_series)
    assert results["moving_average_mape"] >= 0
    assert results["moving_average_rmse"] >= 0
    assert results["arima_mape"] >= 0
    assert results["arima_rmse"] >= 0


def test_forecast_future_produces_dates_beyond_input(sample_series):
    future = forecast_future(sample_series, horizon_days=90, freq="W")
    assert future.index.min() > sample_series.index.max(), "Forecast must be genuinely in the future"


def test_forecast_future_length_matches_horizon(sample_series):
    future = forecast_future(sample_series, horizon_days=90, freq="W")
    # 90 days / 7 ≈ 12 weeks
    assert 11 <= len(future) <= 13


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))