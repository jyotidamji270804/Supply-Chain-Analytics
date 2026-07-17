"""
app.py
-------
Week 4: Streamlit Deployment.

Adds sidebar controls (category selector, forecast horizon, anomaly
sensitivity) and a colorful revenue-by-category bar chart.

Forecast/Anomaly/Trend tabs come next.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH, FORECAST_HORIZON_DAYS
from src.preprocessing import load_raw_data, get_clean_weekly_series
from src.eda import category_summary, data_quality_report
from src.utils import list_available_categories


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
CATEGORY_COLORS = [PRIMARY, ACCENT_1, ACCENT_2, ACCENT_3, "#4EA8DE"]

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
st.caption("Demand Forecasting & Anomaly Detection — Infotact Data Analytics Internship, Project 3")


# ----------------------------------------------------------------------
# Data loading (cached so the app doesn't reload from disk on every click)
# ----------------------------------------------------------------------
@st.cache_data
def get_data():
    if not RAW_DATA_PATH.exists():
        st.error("Dataset not found. Run `python data/generate_data.py` first.")
        st.stop()
    return load_raw_data()


@st.cache_data
def get_weekly_series(category: str):
    df = get_data()
    return get_clean_weekly_series(df, category)


df = get_data()
categories = list_available_categories(df)
summary = category_summary(df)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("🎛️ Controls")
category = st.sidebar.selectbox("Product category", categories, index=0)

horizon_days = st.sidebar.slider(
    "Forecast horizon (days)", min_value=30, max_value=180, value=FORECAST_HORIZON_DAYS, step=15
)

sensitivity = st.sidebar.select_slider(
    "Anomaly detection sensitivity",
    options=["Low (3/3 methods agree)", "Medium (2/3 methods agree)", "High (any 1 method)"],
    value="Medium (2/3 methods agree)",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this app**\n\n"
    "Synthetic data simulating 4 years of daily sales + inventory across "
    "5 product categories, with realistic trend, seasonality, and injected "
    "anomalies (demand spikes & supplier-delay stockouts)."
)

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

# ----------------------------------------------------------------------
# Colorful revenue breakdown chart
# ----------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("💰 Revenue by Category")
    rev_by_cat = summary.reset_index()[["product_category", "total_revenue"]]
    fig = px.bar(
        rev_by_cat, x="product_category", y="total_revenue",
        color="product_category", color_discrete_sequence=CATEGORY_COLORS,
        text_auto=".2s",
    )
    fig.update_layout(
        showlegend=False, height=380,
        xaxis_title="", yaxis_title="Total Revenue ($)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📦 Units Sold Share")
    units_by_cat = summary.reset_index()[["product_category", "total_units_sold"]]
    fig2 = px.pie(
        units_by_cat, names="product_category", values="total_units_sold",
        color_discrete_sequence=CATEGORY_COLORS, hole=0.45,
    )
    fig2.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 Category Performance Summary")
st.dataframe(
    summary.style.background_gradient(cmap="viridis", subset=["total_revenue"])
                 .background_gradient(cmap="Oranges", subset=["total_units_sold"]),
    use_container_width=True,
)

with st.expander("🔍 Data Quality Report (click to expand)"):
    st.dataframe(data_quality_report(df), use_container_width=True)
