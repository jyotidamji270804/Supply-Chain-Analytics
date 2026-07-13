"""
eda.py
-------
Week 1: Exploratory Data Analysis helpers.

These are the checks you'd run in a Jupyter notebook before trusting the
data enough to build models on it -- missingness, duplicates, outlier
sniff-test, and basic category-level summary stats.
"""

import pandas as pd

from src.config import RAW_DATA_PATH


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Quick missing-value / dtype / uniqueness summary per column."""
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )
    return report


def check_duplicate_rows(df: pd.DataFrame, subset=("date", "product_category")) -> int:
    """Duplicate (date, category) combos would silently break resampling."""
    return int(df.duplicated(subset=list(subset)).sum())


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Total units sold, revenue, and avg inventory per product category."""
    return (
        df.groupby("product_category")
        .agg(
            total_units_sold=("units_sold", "sum"),
            total_revenue=("revenue", "sum"),
            avg_inventory_level=("inventory_level", "mean"),
            days_recorded=("date", "count"),
        )
        .round(1)
        .sort_values("total_revenue", ascending=False)
    )


def date_coverage_check(df: pd.DataFrame) -> dict:
    """Confirm the calendar range has no unexpected gaps."""
    expected_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    actual_days = pd.to_datetime(df["date"].unique())
    missing_days = expected_days.difference(actual_days)
    return {
        "start": df["date"].min(),
        "end": df["date"].max(),
        "expected_days": len(expected_days),
        "missing_days": len(missing_days),
    }


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    print("=== Data Quality Report ===")
    print(data_quality_report(df))
    print(f"\nDuplicate (date, category) rows: {check_duplicate_rows(df)}")
    print(f"\nDate coverage: {date_coverage_check(df)}")
    print("\n=== Category Summary ===")
    print(category_summary(df))