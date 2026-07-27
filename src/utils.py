"""
utils.py
---------
Small shared helpers used across preprocessing, anomaly_detection,
forecasting, and the Streamlit app.
"""

import pandas as pd


def format_number(n: float) -> str:
    """Human-friendly number formatting for dashboard metrics (e.g. 12.4K)."""
    if n is None:
        return "-"
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def week_over_week_change(series: pd.Series) -> float:
    """Percentage change between the last two points of a series."""
    if len(series) < 2:
        return 0.0
    prev, latest = series.iloc[-2], series.iloc[-1]
    if prev == 0:
        return 0.0
    return (latest - prev) / prev * 100


def list_available_categories(df: pd.DataFrame) -> list:
    return sorted(df["product_category"].unique().tolist())