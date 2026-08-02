"""
alerts.py
-----------
Simulates what happens when an anomaly is detected: formatting a clear,
actionable alert message and "sending" it (logged, not actually dispatched --
this models the notification layer without needing real Slack/email
credentials, which is exactly how you'd prototype this before wiring up
a real integration).
"""

from datetime import datetime

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


SEVERITY_THRESHOLDS = {
    "LOW": 1,       # 1 method flagged it
    "MEDIUM": 2,    # 2 methods agree
    "HIGH": 3,      # all 3 methods agree
}


def determine_severity(vote_count: int) -> str:
    """Map a detector's vote count to a human-readable severity level."""
    if vote_count >= 3:
        return "HIGH"
    elif vote_count == 2:
        return "MEDIUM"
    elif vote_count == 1:
        return "LOW"
    return "NONE"


def format_alert_message(category: str, week: pd.Timestamp, value: float, vote_count: int) -> str:
    """Build a clear, actionable alert message -- the kind a manager could
    act on immediately without needing to open the dashboard first."""
    severity = determine_severity(vote_count)
    week_str = week.strftime("%Y-%m-%d")

    return (
        f"[{severity}] Anomaly detected — {category}\n"
        f"  Week of: {week_str}\n"
        f"  Units sold: {value:.0f}\n"
        f"  Confidence: {vote_count}/3 detection methods agree\n"
        f"  Suggested action: Review inventory and recent orders for {category} "
        f"around {week_str} — check for stockouts, supplier delays, or demand spikes."
    )


def send_alert(message: str, channel: str = "console") -> dict:
    """
    Simulate dispatching an alert. In a real system, this function would
    call the Slack API or an email service -- here it logs the message and
    returns a record of what "would have" been sent, which is useful for
    testing the alert logic without needing real credentials.
    """
    timestamp = datetime.now().isoformat()
    logger.warning(f"ALERT DISPATCHED via {channel}:\n{message}")

    return {
        "timestamp": timestamp,
        "channel": channel,
        "message": message,
        "status": "simulated_sent",
    }


def process_anomalies_for_alerts(anomalies_df: pd.DataFrame, category: str, channel: str = "console") -> list:
    """
    Given a DataFrame from combine_anomaly_flags(), generate and "send" an
    alert for every flagged week. Returns a list of alert records -- useful
    for displaying an alert history in the dashboard.
    """
    flagged = anomalies_df[anomalies_df["is_anomaly"]]
    alerts = []

    for week, row in flagged.iterrows():
        message = format_alert_message(category, week, row["value"], row["vote_count"])
        alert_record = send_alert(message, channel=channel)
        alerts.append(alert_record)

    return alerts


if __name__ == "__main__":
    from src.preprocessing import load_raw_data, get_clean_weekly_series
    from src.anomaly_detection import combine_anomaly_flags

    df = load_raw_data()
    weekly = get_clean_weekly_series(df, "Electronics")
    anomalies = combine_anomaly_flags(weekly, min_votes=2)

    alerts = process_anomalies_for_alerts(anomalies, category="Electronics")
    logger.info(f"Generated {len(alerts)} alert(s) for Electronics")