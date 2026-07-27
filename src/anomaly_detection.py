"""
anomaly_detection.py
----------------------
Week 2: Statistical Anomaly Detection module.

Implements three complementary detectors, as required by the spec:
  1. Rolling Z-Score        -> good for sudden local deviations
  2. IQR (Tukey fences)     -> robust to non-normal distributions
  3. Isolation Forest       -> catches multivariate / subtler anomalies

Each function returns a boolean mask aligned to the input series/frame so
results can be layered onto a chart directly.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import ZSCORE_THRESHOLD, IQR_MULTIPLIER, ISOLATION_FOREST_CONTAMINATION


def detect_zscore_anomalies(series: pd.Series, window: int = 14, threshold: float = ZSCORE_THRESHOLD) -> pd.Series:
    """
    Rolling Z-score: flags points that deviate more than `threshold` standard
    deviations from a rolling mean. Using a rolling (not global) mean lets
    the detector adapt to trend/seasonality instead of flagging everything
    during a genuine high-season period.
    """
    rolling_mean = series.rolling(window=window, min_periods=3, center=True).mean()
    rolling_std = series.rolling(window=window, min_periods=3, center=True).std().replace(0, np.nan)

    z_scores = (series - rolling_mean) / rolling_std
    anomalies = z_scores.abs() > threshold
    return anomalies.fillna(False)


def detect_iqr_anomalies(series: pd.Series, multiplier: float = IQR_MULTIPLIER) -> pd.Series:
    """
    Classic Tukey-fence IQR method on the full series distribution.
    Good baseline / sanity check alongside the rolling Z-score.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (series < lower_bound) | (series > upper_bound)


def detect_isolation_forest_anomalies(
    series: pd.Series,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = 42,
) -> pd.Series:
    """
    Isolation Forest on [value, day-over-day change, rolling volatility]
    features -- catches anomalies that single-point rules miss (e.g. a value
    that's "normal" in isolation but wildly inconsistent with recent momentum).
    """
    feat = pd.DataFrame(index=series.index)
    feat["value"] = series
    feat["diff"] = series.diff().fillna(0)
    feat["rolling_std"] = series.rolling(7, min_periods=1).std().fillna(0)

    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    preds = model.fit_predict(feat)  # -1 = anomaly, 1 = normal
    return pd.Series(preds == -1, index=series.index)


def combine_anomaly_flags(series: pd.Series, min_votes: int = 2) -> pd.DataFrame:
    """
    Run all three detectors and combine them by majority vote to reduce
    'alert fatigue' from any single method's false positives -- exactly the
    concern called out in the spec's KPI section.
    """
    z = detect_zscore_anomalies(series)
    iqr = detect_iqr_anomalies(series)
    iso = detect_isolation_forest_anomalies(series)

    votes = z.astype(int) + iqr.astype(int) + iso.astype(int)
    consensus = votes >= min_votes

    return pd.DataFrame(
        {
            "value": series,
            "zscore_flag": z,
            "iqr_flag": iqr,
            "isolation_forest_flag": iso,
            "vote_count": votes,
            "is_anomaly": consensus,
        }
    )


if __name__ == "__main__":
    from src.preprocessing import load_raw_data, get_clean_weekly_series

    df = load_raw_data()
    weekly = get_clean_weekly_series(df, "Electronics")
    result = combine_anomaly_flags(weekly)
    print(result[result["is_anomaly"]])
    print(f"\nTotal anomalies flagged (consensus >= 2 methods): {result['is_anomaly'].sum()} / {len(result)}")