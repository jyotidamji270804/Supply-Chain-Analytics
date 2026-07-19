# 📦 Supply Chain Analytics: Demand Forecasting & Anomaly Detection

> An end-to-end time-series analytics pipeline that forecasts product demand and automatically flags supply-chain anomalies — built as part of the Infotact Technical Internship Program (Project 3).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🎯 The Problem

Retailers and manufacturers lose money on both sides of the demand curve:
- **Overestimate demand** → bloated warehouses, tied-up capital, spoilage
- **Underestimate demand** → stockouts, lost sales, frustrated customers

On top of that, sudden anomalies — a viral product trend, a supplier delay —
often go unnoticed in raw sales data until the financial damage is already done.

**This project solves both problems**: a forecasting model that predicts the
next 90 days of demand, and an anomaly detection system that flags unusual
patterns before they become expensive.

---

## 🖥️ What It Does

An interactive Streamlit dashboard where a supply chain manager can:

| Feature | What it shows |
|---|---|
| 📈 **Forecast tab** | Historical demand + 90-day forecast with confidence interval, plus a head-to-head accuracy comparison of a moving-average baseline vs. an ARIMA model |
| 🚨 **Anomaly Detection tab** | Flagged weeks where sales deviated significantly from expected patterns, with an adjustable sensitivity control |
| 🔍 **Trend & Seasonality tab** | A decomposition view separating raw demand into trend, seasonal pattern, and residual noise |

All three tabs respond live to a product category selector in the sidebar.

---

## 🧠 How It Works
Raw daily sales/inventory data
│
▼
┌───────────────────┐
│  Preprocessing     │  → reindex to full calendar, interpolate gaps,
│  (preprocessing.py)│    resample to weekly, decompose trend/seasonality
└───────────────────┘
│
├──────────────────────────┐
▼                          ▼
┌───────────────────┐    ┌───────────────────────┐
│ Anomaly Detection  │    │ Forecasting            │
│(anomaly_detection.py)  │ (forecasting.py)        │
│                    │    │                        │
│ • Rolling Z-score  │    │ • Moving average        │
│ • IQR (Tukey fence)│    │   baseline               │
│ • Isolation Forest │    │ • ARIMA model            │
│                    │    │ • Chronological train/  │
│ Combined via        │   │   test split             │
│ majority vote       │   │ • MAPE / RMSE scoring    │
└───────────────────┘    └───────────────────────┘
│                          │
└──────────┬───────────────┘
▼
┌─────────────────────┐
│  Streamlit Dashboard │
│      (app.py)        │
└─────────────────────┘
**Why majority-vote anomaly detection?** Any single method has blind spots —
Z-score misfires on noisy series, IQR misses gradual shifts, Isolation Forest
can overfit to random noise. Requiring 2-out-of-3 methods to agree filters out
each one's individual false positives, directly addressing the "alert fatigue"
problem real ops teams face with over-sensitive monitoring systems.

**Why compare ARIMA against a naive baseline?** Because model complexity
should earn its keep. On this dataset, ARIMA doesn't automatically win — see
`reports/executive_summary.md` for the actual per-category results.

---

## 📁 Project Structure
├── data/
│   └── generate_data.py         # synthetic 4-year sales/inventory generator
├── src/
│   ├── config.py                 # centralized settings
│   ├── eda.py                    # data quality checks, category summaries
│   ├── preprocessing.py          # resampling, gap-filling, decomposition
│   ├── anomaly_detection.py     # Z-score, IQR, Isolation Forest, consensus
│   ├── forecasting.py            # moving average, ARIMA, MAPE/RMSE
│   └── utils.py                  # shared helpers
├── app/
│   └── app.py                     # Streamlit dashboard
├── tests/
│   └── test_anomaly_detection.py # precision/recall vs. injected ground truth
├── reports/
│   └── executive_summary.md      # business findings & recommendations
├── requirements.txt
└── .gitignore
---

## 🚀 Getting Started

```bash
# Clone and enter the repo
git clone <your-repo-url>
cd Supply-Chain-Analytics

# Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate the dataset (deterministic — same output every time via a fixed seed)
python data/generate_data.py

# Launch the dashboard
streamlit run app/app.py
```

---

## 🔒 A Note on the Data

The dataset is **synthetically generated**, not scraped or proprietary. Real
supply chain data is confidential, so `data/generate_data.py` simulates 4 years
of daily sales and inventory across 5 product categories — with realistic
trend, weekly and yearly seasonality, and *deliberately injected* anomalies
(demand spikes, supplier-delay stockouts) so the detection pipeline has real
signal to find.

Raw data files are excluded from version control (`.gitignore`); the
generator script is the single source of truth, seeded for reproducibility.

---

## ✅ Validation

Rather than just eyeballing the anomaly chart, `tests/test_anomaly_detection.py`
measures the detector's precision and recall against the dataset's *known*
injected anomalies — real accuracy numbers, not a vibe check.

---

## 📊 Key Findings

See [`reports/executive_summary.md`](reports/executive_summary.md) for the
full business write-up, including which product categories forecast most
reliably, where ARIMA beats the baseline (and where it doesn't), and
recommendations for inventory strategy.

---

## 🛠️ Built With

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `Streamlit` · `Plotly`

---

*Built for the Infotact Technical Internship Program — Data Analytics Track*