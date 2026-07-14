"""
forecasting.py
----------------
Week 3: Demand Forecasting module.

First piece: a chronological train/test split. This matters more than it
sounds -- shuffling time series data before splitting (like you might for a
generic ML dataset) leaks future information into training and gives you a
falsely optimistic accuracy score. Never shuffle time series data.

Moving average baseline, ARIMA, and accuracy metrics come in the next commits.
"""

import pandas as pd

from src.config import TRAIN_TEST_SPLIT_RATIO


def train_test_split_series(series: pd.Series, ratio: float = TRAIN_TEST_SPLIT_RATIO):
    """Chronological split -- train is always the earlier portion."""
    split_idx = int(len(series) * ratio)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]
    return train, test


if __name__ == "__main__":
    from src.preprocessing import load_raw_data, get_clean_weekly_series

    df = load_raw_data()
    weekly = get_clean_weekly_series(df, "Electronics")

    train, test = train_test_split_series(weekly)
    print(f"Total weeks: {len(weekly)}")
    print(f"Train weeks: {len(train)} ({train.index.min().date()} to {train.index.max().date()})")
    print(f"Test weeks:  {len(test)} ({test.index.min().date()} to {test.index.max().date()})")