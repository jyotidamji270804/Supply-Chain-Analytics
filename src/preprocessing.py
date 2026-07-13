"""
preprocessing.py
-----------------
Week 1: Time-Series Preprocessing module.

Responsibilities (per spec):
  - Load raw transactional data
  - Set a proper datetime index
  - Resample daily data into weekly/monthly aggregates
  - Handle missing dates via interpolation
  - Decompose the series into trend / seasonality / residual
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult

from src.config import RAW_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and parse dates."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def build_category_series(df: pd.DataFrame, category: str, value_col: str = "units_sold") -> pd.Series:
    """
    Filter to one product category and return a clean daily time series
    (datetime index, missing calendar days filled via interpolation).
    """
    sub = df[df["product_category"] == category].copy()
    sub = sub.sort_values("date").set_index("date")

    # Reindex to a continuous daily calendar to expose any missing dates
    full_range = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
    sub = sub.reindex(full_range)
    sub.index.name = "date"

    # Interpolate the numeric target column for any gaps introduced by reindexing
    sub[value_col] = sub[value_col].interpolate(method="linear").bfill().ffill()

    return sub[value_col]


def resample_series(series: pd.Series, freq: str = "W") -> pd.Series:
    """
    Resample a daily series into weekly ('W') or monthly ('M') aggregates.
    Sales/units are summed; use a different agg for level-type metrics if needed.
    """
    return series.resample(freq).sum()


def decompose_series(series: pd.Series, period: int = 52, model: str = "additive") -> DecomposeResult:
    """
    Decompose a series into trend, seasonal, and residual components.
    `period` should match the seasonality of the input (e.g. 52 for weekly
    data with yearly seasonality, 7 for daily data with weekly seasonality).
    """
    # seasonal_decompose needs at least 2 full periods of data
    if len(series) < 2 * period:
        period = max(2, len(series) // 2)
    result = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
    return result


def get_clean_weekly_series(df: pd.DataFrame, category: str, value_col: str = "units_sold") -> pd.Series:
    """Convenience wrapper: raw df -> clean daily series -> weekly resample."""
    daily = build_category_series(df, category, value_col)
    weekly = resample_series(daily, freq="W")
    return weekly


if __name__ == "__main__":
    df = load_raw_data()
    weekly = get_clean_weekly_series(df, "Electronics")
    print(weekly.head())
    print(f"\nWeekly series length: {len(weekly)}")

    decomposition = decompose_series(weekly, period=52)
    print("\nDecomposition components available:", ["trend", "seasonal", "resid", "observed"])
    