"""
comparison.py
---------------
Cross-category comparison helpers for the dashboard's "Compare All Categories"
view. Keeps the (fairly heavy) looped computation out of app.py so it stays
readable, and cacheable as a single function.
"""

import pandas as pd

from src.preprocessing import get_clean_weekly_series
from src.forecasting import evaluate_models
from src.anomaly_detection import combine_anomaly_flags


def build_comparison_table(df: pd.DataFrame, categories: list, min_votes: int = 2) -> pd.DataFrame:
    """
    Run forecasting evaluation + anomaly detection across every category and
    return one summary row per category -- the "at a glance" view stakeholders
    actually want before drilling into any single category.
    """
    rows = []
    for category in categories:
        weekly = get_clean_weekly_series(df, category)

        eval_results = evaluate_models(weekly, category=category)
        anomalies = combine_anomaly_flags(weekly, min_votes=min_votes)

        best_model = "ARIMA" if eval_results["arima_mape"] < eval_results["moving_average_mape"] else "Moving Average"
        best_mape = min(eval_results["arima_mape"], eval_results["moving_average_mape"])

        rows.append({
            "product_category": category,
            "best_model": best_model,
            "best_mape_pct": round(best_mape, 1),
            "anomalies_flagged": int(anomalies["is_anomaly"].sum()),
            "total_weeks": len(anomalies),
            "anomaly_rate_pct": round(anomalies["is_anomaly"].mean() * 100, 1),
            "avg_weekly_demand": round(weekly.mean(), 0),
            "demand_volatility": round(weekly.std(), 0),
        })

    return pd.DataFrame(rows).set_index("product_category")


if __name__ == "__main__":
    from src.preprocessing import load_raw_data
    from src.config import PRODUCT_CATEGORIES

    df = load_raw_data()
    comparison = build_comparison_table(df, PRODUCT_CATEGORIES)
    print(comparison)