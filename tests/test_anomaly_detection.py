"""
test_anomaly_detection.py
---------------------------
Week 2 validation: how well does our majority-vote anomaly detector recover
the anomalies we KNOW we injected in generate_data.py?

This is only possible because our data is synthetic and self-labeled
(is_injected_anomaly). On real-world data you wouldn't have this luxury --
which is exactly why this kind of validation, when you CAN do it, is worth
doing thoroughly.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_raw_data, build_category_series, resample_series
from src.anomaly_detection import combine_anomaly_flags
from src.config import PRODUCT_CATEGORIES


def weekly_ground_truth(df: pd.DataFrame, category: str) -> pd.Series:
    """
    Aggregate the daily is_injected_anomaly flag up to weekly: a week counts
    as a 'true anomaly week' if ANY day in it was injected as one.
    """
    sub = df[df["product_category"] == category].copy()
    sub = sub.sort_values("date").set_index("date")
    full_range = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
    sub = sub.reindex(full_range)
    sub["is_injected_anomaly"] = sub["is_injected_anomaly"].fillna(False)
    weekly_flag = sub["is_injected_anomaly"].resample("W").max().astype(bool)
    return weekly_flag


def precision_recall_f1(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(precision, 2), "recall": round(recall, 2), "f1_score": round(f1, 2)}


if __name__ == "__main__":
    df = load_raw_data()

    print(f"{'Category':<20} {'Precision':>10} {'Recall':>8} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 60)

    for category in PRODUCT_CATEGORIES:
        weekly_series = resample_series(build_category_series(df, category), freq="W")
        detected = combine_anomaly_flags(weekly_series, min_votes=2)["is_anomaly"]
        truth = weekly_ground_truth(df, category)

        metrics = precision_recall_f1(truth, detected)
        print(f"{category:<20} {metrics['precision']:>10} {metrics['recall']:>8} "
              f"{metrics['f1_score']:>6} {metrics['true_positives']:>4} "
              f"{metrics['false_positives']:>4} {metrics['false_negatives']:>4}")