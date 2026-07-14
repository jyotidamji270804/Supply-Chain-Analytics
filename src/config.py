"""
config.py
---------
Centralized configuration for the Supply Chain Analytics project.
Keeping constants here means every module (preprocessing, anomaly_detection,
forecasting, app) reads from a single source of truth.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "supply_chain_data.csv"
REPORTS_DIR = ROOT_DIR / "reports"

# ----------------------------------------------------------------------
# Data generation settings
# ----------------------------------------------------------------------
PRODUCT_CATEGORIES = [
    "Electronics",
    "Groceries",
    "Apparel",
    "Home & Kitchen",
    "Sports & Outdoors",
]

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
RANDOM_SEED = 42
# ----------------------------------------------------------------------
# Anomaly detection settings
# ----------------------------------------------------------------------
ZSCORE_THRESHOLD = 3.0       # standard deviations away from rolling mean
IQR_MULTIPLIER = 1.5         # standard Tukey fence multiplier
ISOLATION_FOREST_CONTAMINATION = 0.03  # expected proportion of anomalies
# ----------------------------------------------------------------------
# Forecasting settings
# ----------------------------------------------------------------------
FORECAST_HORIZON_DAYS = 90   # "next 90 days" requirement from the spec
TRAIN_TEST_SPLIT_RATIO = 0.85  # chronological split point