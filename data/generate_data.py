"""
generate_data.py
-----------------
Week 1 helper: acquiring a synthetic supply chain dataset.

Since we can't use a real proprietary supply chain feed, this script SIMULATES
one: daily sales + inventory levels for 5 product categories over 4 years,
with realistic trend, weekly + yearly seasonality, random noise, and
DELIBERATELY injected anomalies (viral demand spikes, supplier delays /
stockouts) so the anomaly detection module in Week 2 has real signal to find.

Run directly:
    python data/generate_data.py

This writes data/supply_chain_data.csv (gitignored -- see README).
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import PRODUCT_CATEGORIES, START_DATE, END_DATE, RANDOM_SEED, RAW_DATA_PATH


def _simulate_category(dates: pd.DatetimeIndex, base_level: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate one product category's daily sales + inventory series."""
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

    # --- Inject anomalies ---
    anomaly_flags = np.zeros(n, dtype=bool)
    anomaly_type = np.array([""] * n, dtype=object)

    # 1) Viral demand spikes (2-4 random short bursts)
    n_spikes = rng.integers(2, 5)
    for _ in range(n_spikes):
        start = rng.integers(30, n - 14)
        length = rng.integers(2, 6)
        magnitude = rng.uniform(2.5, 4.5)
        idx = slice(start, start + length)
        sales[idx] *= magnitude
        anomaly_flags[idx] = True
        anomaly_type[idx] = "demand_spike"

    # 2) Supplier delay / stockout dips (sales crash toward zero)
    n_dips = rng.integers(2, 5)
    for _ in range(n_dips):
        start = rng.integers(30, n - 10)
        length = rng.integers(3, 8)
        idx = slice(start, start + length)
        sales[idx] *= rng.uniform(0.05, 0.2)
        anomaly_flags[idx] = True
        anomaly_type[idx] = "stockout"

    sales = np.round(np.clip(sales, a_min=0, a_max=None)).astype(int)

    # --- Inventory: starts high, drawn down by sales, replenished periodically ---
    inventory = np.zeros(n)
    stock = base_level * 25
    reorder_point = base_level * 8
    reorder_qty = base_level * 20
    for i in range(n):
        stock -= sales[i]
        if stock < reorder_point:
            if anomaly_type[i] == "stockout" and rng.random() < 0.5:
                pass  # replenishment delayed, stock stays low
            else:
                stock += reorder_qty * rng.uniform(0.9, 1.1)
        stock = max(stock, 0)
        inventory[i] = stock

    return pd.DataFrame(
        {
            "date": dates,
            "units_sold": sales,
            "inventory_level": np.round(inventory).astype(int),
            "is_injected_anomaly": anomaly_flags,
            "injected_anomaly_type": anomaly_type,
        }
    )


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

    full = pd.concat(frames, ignore_index=True)
    full["unit_price"] = full["product_category"].map(
        {
            "Electronics": 85.0,
            "Groceries": 6.5,
            "Apparel": 32.0,
            "Home & Kitchen": 28.0,
            "Sports & Outdoors": 45.0,
        }
    )
    full["revenue"] = (full["units_sold"] * full["unit_price"]).round(2)
    return full


if __name__ == "__main__":
    df = generate_dataset()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated {len(df):,} rows across {df['product_category'].nunique()} categories")
    print(f"Saved to: {RAW_DATA_PATH}")
    print(df.head())