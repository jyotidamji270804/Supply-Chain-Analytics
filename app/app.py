"""app.py
-------
Streamlit dashboard for the Supply Chain Analytics project.

Four tabs:
  - Forecast: historical + forecasted demand, confidence interval,
    model accuracy comparison (Moving Average vs ARIMA)
  - Anomaly Detection: flagged points on the historical chart, adjustable
    sensitivity, table of flagged weeks
  - Trend & Seasonality: decomposition into trend / seasonal / residual
  - Inventory Planning: safety stock, reorder point, and order recommendations

Run with:
    streamlit run app/app.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH, FORECAST_HORIZON_DAYS, DEFAULT_LEAD_TIME_DAYS, DEFAULT_SERVICE_LEVEL
from src.preprocessing import load_raw_data, get_clean_weekly_series, build_category_series, decompose_series
from src.eda import category_summary, data_quality_report
from src.utils import list_available_categories, week_over_week_change, format_number
from src.forecasting import evaluate_models, forecast_future
from src.anomaly_detection import combine_anomaly_flags
from src.inventory_planning import calculate_reorder_point, recommend_order_quantity, SERVICE_LEVEL_Z_SCORES


# Page setup
st.set_page_config(
    page_title="Supply Chain Demand Forecasting & Anomaly Detection",
    page_icon="📦",
    layout="wide",
)

# Custom color theme (injected CSS)
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


st.title("Supply Chain Analytics")
st.caption("Demand Forecasting & Anomaly Detection")


# Data loading (cached)
@st.cache_data
def get_data():
    if not RAW_DATA_PATH.exists():
        from data.generate_data import generate_dataset
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(RAW_DATA_PATH, index=False)
    return load_raw_data()


@st.cache_data
def get_daily_series(category: str):
    df = get_data()
    return build_category_series(df, category)


@st.cache_data
def get_weekly_series(category: str):
    df = get_data()
    return get_clean_weekly_series(df, category)


@st.cache_data
def get_forecast_evaluation(category: str):
    series = get_weekly_series(category)
    return evaluate_models(series, category=category)


@st.cache_data
def get_future_forecast(category: str, horizon_days: int):
    series = get_weekly_series(category)
    return forecast_future(series, horizon_days=horizon_days, category=category)


@st.cache_data
def get_anomalies(category: str, min_votes: int):
    series = get_weekly_series(category)
    return combine_anomaly_flags(series, min_votes=min_votes)


df = get_data()
categories = list_available_categories(df)
summary = category_summary(df)

# Sidebar controls
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
min_votes_map = {
    "Low (3/3 methods agree)": 3,
    "Medium (2/3 methods agree)": 2,
    "High (any 1 method)": 1,
}
min_votes = min_votes_map[sensitivity]

st.sidebar.markdown("---")
st.sidebar.subheader("Inventory Planning Inputs")
lead_time_days = st.sidebar.number_input(
    "Supplier lead time (days)", min_value=1, max_value=60, value=DEFAULT_LEAD_TIME_DAYS
)
service_level = st.sidebar.selectbox(
    "Target service level", options=list(SERVICE_LEVEL_Z_SCORES.keys()),
    index=list(SERVICE_LEVEL_Z_SCORES.keys()).index(DEFAULT_SERVICE_LEVEL),
)
current_inventory_pct = st.sidebar.slider(
    "Simulated current inventory (% of reorder point)", min_value=20, max_value=150, value=80
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this app**\n\n"
    "Synthetic data simulating 4 years of daily sales + inventory across "
    "5 product categories, with realistic trend, seasonality, and injected "
    "anomalies (demand spikes & supplier-delay stockouts)."
)

# Top-line colorful KPI row
weekly_series = get_weekly_series(category)
cat_summary = summary.loc[category]
wow_change = week_over_week_change(weekly_series)
wow_color = ACCENT_1 if wow_change >= 0 else ACCENT_2

c1, c2, c3, c4 = st.columns(4)
with c1:
    colored_metric("Total Units Sold", format_number(cat_summary["total_units_sold"]), category, PRIMARY)
with c2:
    colored_metric("Total Revenue", f"${format_number(cat_summary['total_revenue'])}", category, ACCENT_1)
with c3:
    colored_metric("Avg Inventory", format_number(cat_summary["avg_inventory_level"]), category, ACCENT_3)
with c4:
    colored_metric("Week-over-Week", f"{wow_change:+.1f}%", "vs prior week", wow_color)

st.markdown("---")

# Colorful category overview (revenue + units share)
left, right = st.columns([2, 1])

with left:
    st.subheader("Revenue by Category")
    rev_by_cat = summary.reset_index()[["product_category", "total_revenue"]]
    fig_rev = px.bar(
        rev_by_cat, x="product_category", y="total_revenue",
        color="product_category", color_discrete_sequence=CATEGORY_COLORS,
        text_auto=".2s",
    )
    fig_rev.update_layout(
        showlegend=False, height=350,
        xaxis_title="", yaxis_title="Total Revenue ($)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_rev, use_container_width=True)

with right:
    st.subheader("Units Sold Share")
    units_by_cat = summary.reset_index()[["product_category", "total_units_sold"]]
    fig_pie = px.pie(
        units_by_cat, names="product_category", values="total_units_sold",
        color_discrete_sequence=CATEGORY_COLORS, hole=0.45,
    )
    fig_pie.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

with st.expander("Data Quality Report (click to expand)"):
    st.dataframe(data_quality_report(df), use_container_width=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast", "🚨 Anomaly Detection", "🔍 Trend & Seasonality", "🧮 Inventory Planning"
])

with tab1:
    st.subheader(f"Next {horizon_days} Days Forecasted Demand — {category}")

    future = get_future_forecast(category, horizon_days)
    eval_results = get_forecast_evaluation(category)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_series.index, y=weekly_series.values,
        mode="lines", name="Historical (weekly units sold)",
        line=dict(color=PRIMARY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=future.index, y=future.values,
        mode="lines+markers", name="Forecast",
        line=dict(color=ACCENT_3, dash="dash", width=2),
        marker=dict(size=6),
    ))

    resid_std = eval_results["test"].sub(eval_results["arima_forecast"]).std()
    if not np.isnan(resid_std):
        fig.add_trace(go.Scatter(
            x=list(future.index) + list(future.index[::-1]),
            y=list(future.values + 1.96 * resid_std) + list((future.values - 1.96 * resid_std)[::-1]),
            fill="toself", fillcolor="rgba(255,184,76,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="95% Confidence Interval",
            showlegend=True,
        ))

    fig.update_layout(
        height=450, xaxis_title="Week", yaxis_title="Units Sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Model Accuracy Comparison (held-out test set)")
    m1, m2 = st.columns(2)
    with m1:
        colored_metric("Moving Average — MAPE", f"{eval_results['moving_average_mape']:.1f}%", "baseline model", ACCENT_2)
        colored_metric("Moving Average — RMSE", f"{eval_results['moving_average_rmse']:.0f} units", "baseline model", ACCENT_2)
    with m2:
        colored_metric("ARIMA — MAPE", f"{eval_results['arima_mape']:.1f}%", "time-series model", ACCENT_1)
        colored_metric("ARIMA — RMSE", f"{eval_results['arima_rmse']:.0f} units", "time-series model", ACCENT_1)

    better_model = "ARIMA" if eval_results["arima_mape"] < eval_results["moving_average_mape"] else "Moving Average baseline"
    st.info(f"💡 On **{category}**, the **{better_model}** currently has the lower forecast error.")

with tab2:
    st.subheader(f"Detected Anomalies — {category}")
    st.caption(f"Sensitivity: {sensitivity}")

    anomalies_df = get_anomalies(category, min_votes)
    n_anomalies = int(anomalies_df["is_anomaly"].sum())

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=anomalies_df.index, y=anomalies_df["value"],
        mode="lines", name="Units Sold", line=dict(color=PRIMARY, width=2),
    ))
    flagged = anomalies_df[anomalies_df["is_anomaly"]]
    fig2.add_trace(go.Scatter(
        x=flagged.index, y=flagged["value"],
        mode="markers", name="Flagged Anomaly",
        marker=dict(color=ACCENT_2, size=13, symbol="x", line=dict(width=2, color="white")),
    ))
    fig2.update_layout(
        height=450, xaxis_title="Week", yaxis_title="Units Sold",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        colored_metric("Anomalies Flagged", f"{n_anomalies}", f"out of {len(anomalies_df)} weeks", ACCENT_2)
    with a2:
        colored_metric("Detection Rate", f"{n_anomalies / len(anomalies_df) * 100:.1f}%", "of all weeks", ACCENT_3)
    with a3:
        colored_metric("Consensus Threshold", f"{min_votes} / 3", "methods must agree", ACCENT_1)

    if n_anomalies > 0:
        st.markdown("**Flagged weeks — investigate: holiday? stockout? viral trend?**")
        display_df = flagged[["value", "zscore_flag", "iqr_flag", "isolation_forest_flag", "vote_count"]].copy()
        display_df.index.name = "week"
        st.dataframe(
            display_df.style.background_gradient(cmap="Reds", subset=["vote_count"]),
            use_container_width=True,
        )

        csv_data = display_df.to_csv().encode("utf-8")
        st.download_button(
            "⬇Download flagged anomalies as CSV",
            data=csv_data,
            file_name=f"{category.replace(' ', '_').lower()}_anomalies.csv",
            mime="text/csv",
        )
    else:
        st.success("No anomalies flagged at the current sensitivity level.")

with tab3:
    st.subheader(f"Time Series Decomposition — {category}")
    st.caption("Separating the raw signal into trend, seasonal pattern, and residual noise.")

    decomposition = decompose_series(weekly_series, period=52)

    fig3 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Trend", "Seasonality", "Residual (noise)"),
        vertical_spacing=0.08,
    )
    fig3.add_trace(go.Scatter(
        x=decomposition.trend.index, y=decomposition.trend,
        line=dict(color=ACCENT_1, width=2), name="Trend",
    ), row=1, col=1)
    fig3.add_trace(go.Scatter(
        x=decomposition.seasonal.index, y=decomposition.seasonal,
        line=dict(color=PRIMARY, width=2), name="Seasonality",
    ), row=2, col=1)
    fig3.add_trace(go.Scatter(
        x=decomposition.resid.index, y=decomposition.resid,
        line=dict(color=ACCENT_2, width=1.5), name="Residual",
    ), row=3, col=1)

    fig3.update_layout(
        height=700, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""
    **How to read this:**
    - **Trend** — the underlying long-term direction of demand for {category}, stripped of seasonal noise.
    - **Seasonality** — the repeating yearly pattern (e.g. holiday bumps, seasonal dips).
    - **Residual** — what's left after removing trend and seasonality; spikes here often line up with the anomalies flagged in the previous tab.
    """)

with tab4:
    st.subheader(f" Inventory Planning — {category}")
    st.caption("Translating demand variability into a concrete reorder decision.")

    daily_demand = get_daily_series(category)
    reorder_info = calculate_reorder_point(daily_demand, lead_time_days=lead_time_days, service_level=service_level)

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        colored_metric("Avg Daily Demand", f"{reorder_info['avg_daily_demand']:.0f} units", "historical average", PRIMARY)
    with i2:
        colored_metric("Safety Stock", f"{reorder_info['safety_stock_units']:.0f} units", f"{service_level} service level", ACCENT_3)
    with i3:
        colored_metric("Reorder Point", f"{reorder_info['reorder_point_units']:.0f} units", f"{lead_time_days}-day lead time", ACCENT_1)
    with i4:
        colored_metric("Service Level", service_level, "target stockout protection", ACCENT_2)

    st.markdown("---")

    future_weekly = get_future_forecast(category, horizon_days)
    simulated_inventory = reorder_info["reorder_point_units"] * (current_inventory_pct / 100)

    order_rec = recommend_order_quantity(
        forecast=future_weekly,
        current_inventory=simulated_inventory,
        reorder_point=reorder_info["reorder_point_units"],
    )

    st.subheader("Order Recommendation")
    st.caption(f"Based on a simulated current inventory of {current_inventory_pct}% of the reorder point — adjust in the sidebar.")

    if order_rec["should_reorder_now"]:
        st.error(
            f" **REORDER NOW** — current inventory ({order_rec['current_inventory']:.0f} units) "
            f"is at or below the reorder point ({order_rec['reorder_point']:.0f} units)."
        )
        colored_metric(
            "Recommended Order Quantity",
            f"{order_rec['recommended_order_quantity']:.0f} units",
            f"to cover forecasted demand over next {horizon_days} days",
            ACCENT_2,
        )
    else:
        st.success(
            f"🟢 **NO ACTION NEEDED** — current inventory ({order_rec['current_inventory']:.0f} units) "
            f"is above the reorder point ({order_rec['reorder_point']:.0f} units)."
        )

    # Visual: inventory level vs reorder point vs safety stock
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=["Current Inventory", "Reorder Point", "Safety Stock"],
        y=[simulated_inventory, reorder_info["reorder_point_units"], reorder_info["safety_stock_units"]],
        marker_color=[
            ACCENT_2 if order_rec["should_reorder_now"] else ACCENT_1,
            ACCENT_3,
            PRIMARY,
        ],
        text=[f"{simulated_inventory:.0f}", f"{reorder_info['reorder_point_units']:.0f}", f"{reorder_info['safety_stock_units']:.0f}"],
        textposition="outside",
    ))
    fig4.update_layout(
        height=380, yaxis_title="Units",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("ℹ️ How these numbers are calculated"):
        st.markdown(f"""
        - **Safety Stock** = Z-score({service_level}) × std(daily demand) × √(lead time in days)
        - **Reorder Point** = (avg daily demand × lead time) + safety stock
        - **Recommended Order Quantity** = forecasted demand over the horizon − current inventory (only if below reorder point)

        Raising the service level increases the Z-score, which increases safety stock —
        this is the classic tradeoff between stockout risk and holding cost.
        """)

st.markdown("---")
st.caption("Built for the Infotact Technical Internship Program —Supply Chain Analytics")