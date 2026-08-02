"""
test_validation.py
---------------------
Unit tests for src/validation.py -- confirms the validators actually catch
bad data (not just pass on good data, which would be a weak test).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.validation import (
    DataValidationError,
    validate_schema,
    validate_dtypes,
    validate_no_negative_values,
    validate_known_categories,
    validate_no_full_duplicate_rows,
    run_all_validations,
)


@pytest.fixture
def good_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "product_category": ["Electronics"] * 5,
        "units_sold": [10, 20, 30, 40, 50],
        "inventory_level": [100, 90, 80, 70, 60],
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
    })


def test_validate_schema_passes_on_good_data(good_df):
    validate_schema(good_df)


def test_validate_schema_catches_missing_column(good_df):
    bad_df = good_df.drop(columns=["revenue"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_schema(bad_df)


def test_validate_dtypes_catches_non_numeric_column(good_df):
    bad_df = good_df.copy()
    bad_df["units_sold"] = bad_df["units_sold"].astype(str)
    with pytest.raises(DataValidationError, match="not numeric"):
        validate_dtypes(bad_df)


def test_validate_no_negative_values_catches_negative_sales(good_df):
    bad_df = good_df.copy()
    bad_df.loc[0, "units_sold"] = -5
    with pytest.raises(DataValidationError, match="negative values"):
        validate_no_negative_values(bad_df)


def test_validate_known_categories_catches_unexpected_category(good_df):
    bad_df = good_df.copy()
    bad_df.loc[0, "product_category"] = "Totally New Category"
    with pytest.raises(DataValidationError, match="Unexpected product categories"):
        validate_known_categories(bad_df, expected_categories=["Electronics", "Groceries"])


def test_validate_no_duplicate_rows_catches_exact_duplicates(good_df):
    bad_df = pd.concat([good_df, good_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicated"):
        validate_no_full_duplicate_rows(bad_df)


def test_run_all_validations_all_pass_on_good_data(good_df):
    report = run_all_validations(good_df, expected_categories=["Electronics"])
    assert all(v == "PASS" for v in report.values())


def test_run_all_validations_reports_failures_without_raising(good_df):
    bad_df = good_df.copy()
    bad_df.loc[0, "units_sold"] = -5
    report = run_all_validations(bad_df)
    assert report["no_negative_values"].startswith("FAIL")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))