# Executive Summary — Supply Chain Demand Forecasting & Anomaly Detection

## 1. Business Problem
Supply chain teams need to balance two costly failure modes: overstocking
(tied-up capital, spoilage) and understocking (lost sales, stockouts).
This project builds a forecasting + anomaly detection pipeline so
procurement decisions are driven by data, not gut feel.

## 2. Key Findings
- **Forecast accuracy by category:** [fill in from your `evaluate_models` output — which category had the lowest MAPE, which had the highest]
- **Model comparison:** [fill in which categories ARIMA won on vs. the moving-average baseline]
- **Anomalies detected:** [fill in from your Streamlit anomaly tab — how many flagged per category at medium sensitivity]

## 3. Recommendations
- [e.g. "Increase safety stock for X category ahead of Q4 based on the seasonal pattern in the decomposition view"]
- [e.g. "Investigate recurring dips in Y category — pattern suggests a systemic reorder-point issue"]

## 4. Limitations
- Data is synthetically generated to simulate realistic patterns; a
  production deployment would validate model choice against real
  historical data.
- Anomaly detection thresholds (Z-score, IQR, Isolation Forest contamination)
  are configurable in `src/config.py` and should be tuned against
  business-confirmed anomalies over time.