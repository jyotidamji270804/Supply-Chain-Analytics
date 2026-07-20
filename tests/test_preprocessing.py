"""
test_preprocessing.py
------------------------
Unit tests for src/preprocessing.py.

Checks:
  - build_category_series returns a continuous daily index with no gaps
  - build_category_series has no missing values after interpolation
  - resample_series correctly aggregates daily -> weekly totals
  - decompose_series returns trend/seasonal/resid components of matching length
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    load_raw_data,
    build_category_series,
    resample_series,
    decompose_series,
    get_clean_weekly_series,
)
from src.config import PRODUCT_CATEGORIES


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


def test_build_category_series_no_gaps(raw_df):
    series = build_category_series(raw_df, "Electronics")
    expected_range = pd.date_range(series.index.min(), series.index.max(), freq="D")
    assert len(series) == len(expected_range), "Series should have one row per calendar day"


def test_build_category_series_no_missing_values(raw_df):
    series = build_category_series(raw_df, "Electronics")
    assert not series.isna().any(), "Interpolation should leave no missing values"


def test_build_category_series_all_categories(raw_df):
    """Every category in config should be filterable without error."""
    for category in PRODUCT_CATEGORIES:
        series = build_category_series(raw_df, category)
        assert len(series) > 0, f"{category} produced an empty series"


def test_resample_series_weekly_sum_correct(raw_df):
    daily = build_category_series(raw_df, "Electronics")
    weekly = resample_series(daily, freq="W")

    # Spot check: first full week's weekly total should equal sum of its daily values
    first_week_end = weekly.index[0]
    daily_values_in_first_week = daily[daily.index <= first_week_end]
    assert weekly.iloc[0] == pytest.approx(daily_values_in_first_week.sum())


def test_resample_series_reduces_row_count(raw_df):
    daily = build_category_series(raw_df, "Electronics")
    weekly = resample_series(daily, freq="W")
    assert len(weekly) < len(daily), "Weekly resampling should produce fewer rows than daily"


def test_decompose_series_components_match_length(raw_df):
    weekly = get_clean_weekly_series(raw_df, "Electronics")
    decomposition = decompose_series(weekly, period=52)

    assert len(decomposition.trend) == len(weekly)
    assert len(decomposition.seasonal) == len(weekly)
    assert len(decomposition.resid) == len(weekly)


def test_decompose_series_handles_short_series():
    """Should not crash even if the series is shorter than 2x the seasonal period."""
    short_series = pd.Series(
        range(10),
        index=pd.date_range("2024-01-01", periods=10, freq="W"),
    )
    decomposition = decompose_series(short_series, period=52)
    assert len(decomposition.trend) == len(short_series)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))