"""
test_alerts.py
----------------
Unit tests for src/alerts.py.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.alerts import determine_severity, format_alert_message, send_alert, process_anomalies_for_alerts


def test_determine_severity_high():
    assert determine_severity(3) == "HIGH"


def test_determine_severity_medium():
    assert determine_severity(2) == "MEDIUM"


def test_determine_severity_low():
    assert determine_severity(1) == "LOW"


def test_determine_severity_none():
    assert determine_severity(0) == "NONE"


def test_format_alert_message_contains_key_info():
    week = pd.Timestamp("2024-06-16")
    message = format_alert_message("Electronics", week, 250.0, 3)

    assert "Electronics" in message
    assert "2024-06-16" in message
    assert "HIGH" in message
    assert "3/3" in message


def test_send_alert_returns_expected_structure():
    result = send_alert("test message", channel="console")

    assert result["status"] == "simulated_sent"
    assert result["channel"] == "console"
    assert "timestamp" in result


def test_process_anomalies_for_alerts_generates_one_per_flagged_week():
    anomalies_df = pd.DataFrame({
        "value": [100, 500, 110],
        "vote_count": [0, 3, 1],
        "is_anomaly": [False, True, False],
    }, index=pd.date_range("2024-01-01", periods=3, freq="W"))

    alerts = process_anomalies_for_alerts(anomalies_df, category="Test Category")
    assert len(alerts) == 1  # only the one row with is_anomaly=True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))