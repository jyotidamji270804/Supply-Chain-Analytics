"""
validation.py
---------------
Data validation layer: checks that incoming data meets the assumptions the
rest of the pipeline depends on, and fails loudly (with a clear message)
instead of silently producing garbage results downstream.
"""

import pandas as pd


class DataValidationError(Exception):
    """Raised when the raw dataset fails a required validation check."""
    pass


REQUIRED_COLUMNS = {
    "date", "product_category", "units_sold", "inventory_level", "revenue"
}


def validate_schema(df: pd.DataFrame) -> None:
    """Confirm all required columns are present."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_dtypes(df: pd.DataFrame) -> None:
    """Confirm date parses as datetime and numeric columns are actually numeric."""
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise DataValidationError("'date' column is not datetime type -- did you forget parse_dates?")

    for col in ["units_sold", "inventory_level", "revenue"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"'{col}' column is not numeric -- got {df[col].dtype}")


def validate_no_negative_values(df: pd.DataFrame) -> None:
    """Sales, inventory, and revenue should never be negative in a real dataset."""
    for col in ["units_sold", "inventory_level", "revenue"]:
        n_negative = (df[col] < 0).sum()
        if n_negative > 0:
            raise DataValidationError(f"'{col}' has {n_negative} negative values -- data quality issue")


def validate_known_categories(df: pd.DataFrame, expected_categories: list) -> None:
    """Catch typos or unexpected new categories that would silently break filtering."""
    unexpected = set(df["product_category"].unique()) - set(expected_categories)
    if unexpected:
        raise DataValidationError(f"Unexpected product categories found: {unexpected}")


def validate_no_full_duplicate_rows(df: pd.DataFrame) -> None:
    """Exact duplicate rows usually indicate an upstream ingestion bug (double-loaded file, etc.)."""
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        raise DataValidationError(f"Found {n_dupes} fully duplicated rows -- possible double-ingestion")


def run_all_validations(df: pd.DataFrame, expected_categories: list = None) -> dict:
    """
    Run every validation check and return a report instead of raising on the
    first failure -- useful for seeing the FULL picture of data quality
    issues in one pass, rather than fixing them one at a time.
    """
    checks = {
        "schema": validate_schema,
        "dtypes": validate_dtypes,
        "no_negative_values": validate_no_negative_values,
        "no_full_duplicate_rows": validate_no_full_duplicate_rows,
    }

    results = {}
    for name, check_fn in checks.items():
        try:
            check_fn(df)
            results[name] = "PASS"
        except DataValidationError as e:
            results[name] = f"FAIL: {e}"

    if expected_categories:
        try:
            validate_known_categories(df, expected_categories)
            results["known_categories"] = "PASS"
        except DataValidationError as e:
            results["known_categories"] = f"FAIL: {e}"

    return results


if __name__ == "__main__":
    from src.preprocessing import load_raw_data
    from src.config import PRODUCT_CATEGORIES

    df = load_raw_data(validate=False)
    report = run_all_validations(df, expected_categories=PRODUCT_CATEGORIES)

    print("=== Data Validation Report ===")
    for check, result in report.items():
        status_icon = "PASS" if result == "PASS" else "FAIL"
        print(f"[{status_icon}] {check}: {result}")

    all_passed = all(v == "PASS" for v in report.values())
    if not all_passed:
        raise DataValidationError("One or more validation checks failed -- see report above")
    print("\nAll checks passed. Data is safe to proceed with.")