"""
generate_data.py
-----------------
Week 1 helper: acquiring a synthetic supply chain dataset.

Since we can't use a real proprietary supply chain feed, this script SIMULATES
one: daily sales for 5 product categories over 4 years, with realistic trend,
weekly + yearly seasonality, and random noise.

Anomaly injection and inventory simulation come in the next commit.

Run directly:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import PRODUCT_CATEGORIES, START_DATE, END_DATE, RANDOM_SEED, RAW_DATA_PATH


def _simulate_category(dates: pd.DatetimeIndex, base_level: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate one product category's daily sales series."""
    n = len(dates)
    t = np.arange(n)

    # --- Trend: slow linear growth with a slight curve ---
    trend = base_level + 0.015 * t + 0.00002 * t**2

    # --- Yearly seasonality (e.g., holiday bumps) ---
    yearly_season = base_level * 0.35 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2.2)

    # --- Weekly seasonality (weekend dips for B2B-style ordering) ---
    weekly_season = base_level * 0.12 * np.sin(2 * np.pi * t / 7)

    # --- Random noise ---
    noise = rng.normal(0, base_level * 0.06, size=n)

    sales = trend + yearly_season + weekly_season + noise
    sales = np.clip(sales, a_min=1, a_max=None)
    sales = np.round(sales).astype(int)

    return pd.DataFrame({"date": dates, "units_sold": sales})


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    base_levels = {
        "Electronics": 120,
        "Groceries": 480,
        "Apparel": 210,
        "Home & Kitchen": 150,
        "Sports & Outdoors": 90,
    }

    frames = []
    for category in PRODUCT_CATEGORIES:
        df = _simulate_category(dates, base_levels.get(category, 150), rng)
        df.insert(1, "product_category", category)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate_dataset()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated {len(df):,} rows across {df['product_category'].nunique()} categories")
    print(df.head())