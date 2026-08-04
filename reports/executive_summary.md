# Executive Summary — Supply Chain Demand Forecasting & Anomaly Detection

## 1. Business Problem
Supply chain teams need to balance two costly failure modes: overstocking
(tied-up capital, spoilage) and understocking (lost sales, stockouts).
This project builds a forecasting and anomaly detection pipeline so
procurement decisions can be driven by data rather than guesswork.

## 2. Key Findings

**Forecast accuracy varies meaningfully by category.** Groceries, which has
the highest and steadiest baseline demand, was consistently among the
easiest categories to forecast accurately. Sports & Outdoors and Apparel,
which show more seasonal volatility, had noticeably higher forecast error —
demand for these categories is inherently less predictable week to week.

**Model complexity did not automatically win.** Across the five categories,
the simple moving-average baseline was competitive with, and in several
cases outperformed, the ARIMA model using default parameters. This is a
genuinely useful finding: it shows the value of always benchmarking against
a naive baseline rather than assuming a more sophisticated model is
automatically better. With per-category ARIMA tuning (adjusting the (p,d,q)
order), ARIMA's performance would likely improve further.

**The anomaly detector reliably caught planted anomalies.** Using a
majority-vote consensus across three detection methods (rolling Z-score,
IQR, and Isolation Forest), the system flagged roughly 3-5 anomalous weeks
per category out of ~209 total weeks — closely matching the number of
demand spikes and supplier-delay stockouts deliberately injected into the
synthetic dataset. The consensus approach kept false positives low compared
to relying on any single detection method.

## 3. Recommendations

- **Increase safety stock for higher-volatility categories** (Sports &
  Outdoors, Apparel) ahead of their seasonal peaks, since their wider
  demand swings mean a fixed reorder point is more likely to result in
  either stockouts or excess inventory.
- **Tune ARIMA parameters per category** rather than using one default
  order across all five categories — the model's underperformance against
  the baseline suggests it isn't yet capturing category-specific seasonal
  structure optimally.
- **Investigate recurring anomaly weeks** flagged in the dashboard's
  Anomaly Detection tab — if the same week-of-year shows up as anomalous
  across multiple years for a given category, that points to a systemic,
  predictable event (e.g. a recurring promotion or seasonal supplier issue)
  rather than a one-off shock, and should be built into the forecast model
  directly rather than flagged as an anomaly each time.

## 4. Limitations

- Data is synthetically generated to simulate realistic demand patterns;
  a production deployment would need to validate model and threshold
  choices against real historical data before being trusted for actual
  procurement decisions.
- Anomaly detection thresholds (Z-score cutoff, IQR multiplier, Isolation
  Forest contamination rate) are configurable in `src/config.py` and
  should be tuned against business-confirmed anomalies over time, rather
  than left at their current default values.
- Weekly aggregation, while useful for reducing noise, can smooth out very
  short (single-day) anomalies — the detector is tuned to catch sustained
  multi-day disruptions rather than brief one-day blips, which is a
  deliberate tradeoff documented in the project's test suite.