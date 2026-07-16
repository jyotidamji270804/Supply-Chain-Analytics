"""
app.py
-------
Week 4: Streamlit Deployment.

Colorful, info-rich dashboard foundation: custom theming, cached data
loading, and an expanded summary view. Sidebar controls and the
forecast/anomaly/trend tabs come next.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH
from src.preprocessing import load_raw_data
from src.eda import category_summary, data_quality_report


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Demand Forecasting & Anomaly Detection",
    page_icon="📦",
    layout="wide",
)

# ----------------------------------------------------------------------
# Custom color theme (injected CSS)
# ----------------------------------------------------------------------
PRIMARY = "#6C63FF"     # electric violet
ACCENT_1 = "#00C2A8"    # teal
ACCENT_2 = "#FF6B6B"    # coral
ACCENT_3 = "#FFB84C"    # amber
BG_CARD = "#1E1E2F"

st.markdown(f"""
    <style>
    .metric-card {{
        background: linear-gradient(135deg, {BG_CARD} 0%, #2A2A40 100%);
        border-left: 5px solid {PRIMARY};
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }}
    .metric-card h3 {{
        color: #AAAAAA;
        font-size: 13px;
        font-weight: 500;
        margin: 0 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .metric-card p {{
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
    }}
    .metric-card span {{
        font-size: 13px;
        font-weight: 500;
    }}
    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
    }}
    </style>
""", unsafe_allow_html=True)


def colored_metric(title, value, subtitle="", color=PRIMARY):
    st.markdown(f"""
        <div class="metric-card" style="border-left-color:{color};">
            <h3>{title}</h3>
            <p>{value}</p>
            <span style="color:{color};">{subtitle}</span>
        </div>
    """, unsafe_allow_html=True)


st.title("📦 Supply Chain Analytics")
st.caption("Demand Forecasting & Anomaly Detection")


# ----------------------------------------------------------------------
# Data loading (cached so the app doesn't reload from disk on every click)
# ----------------------------------------------------------------------
@st.cache_data
def get_data():
    if not RAW_DATA_PATH.exists():
        st.error("Dataset not found. Run `python data/generate_data.py` first.")
        st.stop()
    return load_raw_data()


df = get_data()
summary = category_summary(df)

# ----------------------------------------------------------------------
# Top-line colorful KPI row
# ----------------------------------------------------------------------
total_revenue = df["revenue"].sum()
total_units = df["units_sold"].sum()
avg_inventory = df["inventory_level"].mean()
n_categories = df["product_category"].nunique()

c1, c2, c3, c4 = st.columns(4)
with c1:
    colored_metric("Total Revenue", f"${total_revenue:,.0f}", "all categories, all-time", PRIMARY)
with c2:
    colored_metric("Total Units Sold", f"{total_units:,.0f}", "across 4 years", ACCENT_1)
with c3:
    colored_metric("Avg Inventory Level", f"{avg_inventory:,.0f} units", "daily average", ACCENT_3)
with c4:
    colored_metric("Product Categories", f"{n_categories}", "tracked in this dataset", ACCENT_2)

st.markdown("---")

st.subheader("📊 Category Performance Summary")
st.dataframe(
    summary.style.background_gradient(cmap="viridis", subset=["total_revenue"])
                 .background_gradient(cmap="Oranges", subset=["total_units_sold"]),
    use_container_width=True,
)

with st.expander("🔍 Data Quality Report (click to expand)"):
    st.dataframe(data_quality_report(df), use_container_width=True)